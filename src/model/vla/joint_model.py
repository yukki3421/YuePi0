import torch
from torch import nn
import math
from omegaconf import OmegaConf

from model.vla.mixture import Mixture


def _to_omega(obj):
    """把任意 config 对象转成 OmegaConf。
    支持: OmegaConf 自身 / dict / 普通 Python class 实例。
    """
    if OmegaConf.is_config(obj):
        return obj
    if isinstance(obj, dict):
        return OmegaConf.create(obj)
    # 普通 class 实例：把所有非下划线属性提取成 dict
    fields = {
        k: v for k, v in vars(type(obj)).items()
        if not k.startswith("_") and not callable(v)
    }
    # 也提取实例属性（如果有）
    fields.update({
        k: v for k, v in vars(obj).items()
        if not k.startswith("_") and not callable(v)
    })
    return OmegaConf.create(fields)


class JointModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.num_mixture = len(config.mixture)

        # 提取joint层的共享字段, 每个mixture都需要的
        shared_config = OmegaConf.create({
            "num_heads": config.num_heads,
            "num_kv_heads": config.num_kv_heads,
            "head_dim": config.head_dim,
            "rms_norm_eps": config.rms_norm_eps,
            "attention_bias": config.attention_bias,
            "attention_dropout": config.attention_dropout,
            "num_hidden_layers": config.num_hidden_layers,
        })

        # Mixtures: VLM, proprio, action
        self.mixtures = nn.ModuleDict()
        for mixture_name, mixture_config in config.mixture.items():
            mixture_config = _to_omega(mixture_config)
            merged = OmegaConf.merge(shared_config, mixture_config)
            self.mixtures[mixture_name] = Mixture(merged)
        self.mixture_names = list(self.mixtures.keys())

    def forward(self, attention_mask, position_ids_all, embeds_all):
        # 把 N 层 forward_mixture_layers 串起来，最后做 final norm
        active_mixture_names = list(embeds_all.keys())

        #  1. normalization
        # 输入 embedding 乘 sqrt(hidden_size)    [B, T, hidden_size]
        # 这是一个Attention 预缩放技巧，和 RMSNorm / 残差连接配合使用
        # Gemma / PaliGemma 会在 token embedding 进入 Transformer 层之前，
        # 把 embedding 乘以 sqrt(hidden_size)，让 embedding 的数值尺度和后续残差流更匹配。
        hidden_states = {}
        for name in active_mixture_names:
            hidden_size = embeds_all[name].shape[-1]
            normalizer = torch.tensor(
                hidden_size ** 0.5,
                dtype=embeds_all[name].dtype,
                device=embeds_all[name].device,
            )
            hidden_states[name] = embeds_all[name] * normalizer

        # 2. layer
        for layer_idx in range(self.num_hidden_layers):
            hidden_states = forward_mixture_layers(
                self.mixtures,
                attention_mask,
                position_ids_all,
                hidden_states,
                layer_idx,
            )
        # 3. norm
        hidden_states_all = {}
        for name in active_mixture_names:
            hidden_states_all[name] = self.mixtures[name].forward_norm( hidden_states[name] )
        return hidden_states_all

# 一层里的联合attention
def forward_mixture_attn(
        mixtures: nn.ModuleDict, # 三个Mixture的字典, key为'vlm', 'proprio', 'action'
        attention_mask: torch.FloatTensor, # 联合mask
        position_ids_all: dict[torch.LongTensor], # 每个expert的position_ids
        embeds_all: dict[torch.FloatTensor],
        layer_idx: int
) -> dict[torch.FloatTensor]:
    bsz = attention_mask.shape[0]
    q_lens = [embed.shape[1] for embed in embeds_all.values()]

    active_mixture_names = list(embeds_all.keys())

    q_all = {}
    k_all = {}
    v_all = {}
    # TODO : 阶段 1：每个 expert 各自算 q/k/v + RoPE + repeat_kv
    for name in active_mixture_names:
        # 1) 计算各自的 Q: (B, T, hidden) -> (B, num_heads, T, head_dim)
        # 计算各自的K, V： （B，T, hidden) -> (B, num_kv_heads, T, head_dim)
        '''
        利用Mixture的这个派发器
        def attn_func(self, method_name:str, layer_idx:int, *args) -> torch.FloatTensor:
            args = [arg for arg in args if arg is not None]
            return getattr(self.layers[layer_idx].self_attn, method_name)(*args)
        '''
        x = embeds_all[name]
        q = mixtures[name].attn_func(
            "forward_q_proj", layer_idx, x
        )
        # 计算各自的K
        k = mixtures[name].attn_func(
            "forward_k_proj", layer_idx, x
        )
        # 计算各自的V
        v = mixtures[name].attn_func(
            "forward_v_proj", layer_idx, x
        )
        # 2) RoPE
        cos, sin = mixtures[name].attn_func(
            "forward_rotary_emb", layer_idx, q, position_ids_all[name]
        )
        q_all[name] = mixtures[name].attn_func(
            "forward_apply_rotary_emb", layer_idx, q, cos, sin
        )
        k = mixtures[name].attn_func(
            "forward_apply_rotary_emb", layer_idx, k, cos, sin
        )
        # 3) 扩展 K, V
        k_all[name], v_all[name] = mixtures[name].attn_func(
            "repeat_kv", layer_idx, k, v
        )
    # TODO : 阶段 2：cat → 联合 attention → split → o_proj
    # (B, num_q_heads/num_kv_heads, T, head_dim), 按照T这个维度 拼接
    '''
    ⚠️ 两个隐性前提：
    - 三个 expert num_heads 必须一样（否则 H 维不对齐）
    - 三个 expert head_dim 必须一样（否则 D 维不对齐）
    '''
    query_states = torch.cat([q_all[name] for name in active_mixture_names], dim=-2)
    key_states = torch.cat([k_all[name] for name in active_mixture_names], dim=-2)
    value_states = torch.cat([v_all[name] for name in active_mixture_names], dim=-2)
    head_dim = query_states.shape[-1]
    attn_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(head_dim)
    # 加mask, 这里的attention_mask的形状 [B, 1, T, T]
    attn_scores = attn_scores + attention_mask
    attn_weights = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    # 加权求V
    attn_outputs = attn_weights @ value_states
    # （B, h, T, head_dim) -> (B, T, hidden_size)
    attn_outputs = attn_outputs.transpose(1, 2).contiguous().view(bsz, sum(q_lens), -1)
    # split回三段, 返回tuple
    attn_output_split = torch.split(attn_outputs, q_lens, dim=1)
    attn_outputs = dict(zip(active_mixture_names, attn_output_split))

    attn_outputs_final = {}
    for name in active_mixture_names:
        '''
        利用Mixture的这个派发器
        def attn_func(self, method_name:str, layer_idx:int, *args) -> torch.FloatTensor:
            args = [arg for arg in args if arg is not None]
            return getattr(self.layers[layer_idx].self_attn, method_name)(*args)
        '''
        attn_outputs_final[name] = mixtures[name].attn_func('forward_o_proj', layer_idx, attn_outputs[name])
    return attn_outputs_final


# 实现MixtureDecoderLayer的一层
def forward_mixture_layers(
        mixtures: nn.ModuleDict,
        attention_mask: torch.FloatTensor, # (B, h, T, T), 所有 image+text的
        position_ids_all: dict[torch.LongTensor],
        embeds_all: dict[torch.FloatTensor],
        layer_idx: int
) -> dict[torch.FloatTensor]:
    active_mixture_names = list(embeds_all.keys())

    residuals_pre_attn = embeds_all
    # 1)先做norm
    hidden_states_input_norm = {}
    for name in active_mixture_names:
        # 接收参数为 method_name:str, layer_idx:int
        hidden_states_input_norm[name] = mixtures[name].layer_func(
            "forward_norm",
            layer_idx,
            "input_layernorm",
            embeds_all[name]
        )
    hidden_states_pre_attn = hidden_states_input_norm
    ''' hidden_states_pre_attn['vlm'].shape
        torch.Size([2, 276, 2048])

        hidden_states_pre_attn['proprio'].shape
        torch.Size([2, 1, 1024])
        '''
    # 2)再做attn
    hidden_states_post_attn = forward_mixture_attn(
        mixtures,
        attention_mask,
        position_ids_all,
        hidden_states_pre_attn,
        layer_idx)

    # hidden_states_post_attn [B, T, hidden_size]
    # 3) 做残差连接
    hidden_states_post_res = {}
    for name in active_mixture_names:
        hidden_states_post_res[name] = residuals_pre_attn[name] + hidden_states_post_attn[name]
    residuals_pre_mlp = hidden_states_post_res

    # 4) 再做post_attn_layernorm + MLP
    hidden_states_post_norm = {}
    hidden_states_post_mlp = {}
    for name in active_mixture_names:
        hidden_states_post_norm[name] = mixtures[name].layer_func(
            "forward_norm", layer_idx, "post_attention_layernorm", hidden_states_post_res[name]
        )
        hidden_states_post_mlp[name] = mixtures[name].layer_func(
            "mlp", layer_idx, hidden_states_post_norm[name]
        )

    # 5) 残差合并
    hidden_states_final = {}
    for name in active_mixture_names:
        hidden_states_final[name] = residuals_pre_mlp[name] + hidden_states_post_mlp[name]

    return hidden_states_final
