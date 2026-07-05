import torch
from torch import nn
import math
from omegaconf import OmegaConf
from typing import Optional
from model.vla.mixture import Mixture
from model.kvcache import KVCache

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

        self.cache_names = [
            name for name in config.mixture if config.mixture[name].cache
        ]

        # 提取joint层的共享字段, 每个mixture都需要的
        shared_config = OmegaConf.create({
            "num_heads": config.num_heads,
            "num_kv_heads": config.num_kv_heads,
            "head_dim": config.head_dim,
            "rms_norm_eps": config.rms_norm_eps,
            "attention_bias": config.attention_bias,
            "attention_dropout": config.attention_dropout,
            "num_hidden_layers": config.num_hidden_layers,
            "time_hidden_size": config.time_hidden_size,
        })

        # Mixtures: VLM, proprio, action
        self.mixtures = nn.ModuleDict()
        for mixture_name, mixture_config in config.mixture.items():
            mixture_config = _to_omega(mixture_config)
            merged = OmegaConf.merge(shared_config, mixture_config)
            self.mixtures[mixture_name] = Mixture(merged)
        self.mixture_names = list(self.mixtures.keys())


    def forward(
        self,
        attention_mask: torch.Tensor,
        position_ids_all: dict[torch.Tensor],
        embeds_all: dict[torch.Tensor],
        time_cond: Optional[torch.FloatTensor] = None,
        kv_caches: dict[KVCache] = None,
        cache_mode: str = "no_append",
        return_caches: bool = False,
    ) -> dict[torch.FloatTensor]:

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
                mixtures = self.mixtures,
                attention_mask = attention_mask,
                position_ids_all = position_ids_all,
                embeds_all = hidden_states,
                layer_idx = layer_idx,
                time_cond = time_cond,
                kv_caches = kv_caches,
                cache_mode = cache_mode,
            )
        # 3. norm
        hidden_states_all = {}
        for name in active_mixture_names:
            hidden_states_all[name] = self.mixtures[name].forward_norm( hidden_states[name], time_cond=time_cond)
        if return_caches:
            return hidden_states_all, kv_caches

        return hidden_states_all

    def build_mixture_caches(self):
        return {name: KVCache() for name in self.cache_names}

# 一层里的联合attention
def forward_mixture_attn(
        mixtures: nn.ModuleDict, # 三个Mixture的字典, key为'vlm', 'proprio', 'action'
        attention_mask: torch.FloatTensor, # 联合mask
        position_ids_all: dict[torch.LongTensor], # 每个expert的position_ids
        embeds_all: dict[torch.FloatTensor],
        layer_idx: int,
        kv_caches: dict = {},
        cache_mode: str = "no_append", # 朴素推理/训练时, 不用kv cache
        attn_softclamp: float = 50.0,  # default in gemma
) -> dict[torch.FloatTensor]:
    bsz = attention_mask.shape[0]
    q_lens = [embed.shape[1] for embed in embeds_all.values()]
    assert cache_mode in ["no_append", "append", "append_non_active"]
    active_mixture_names = list(embeds_all.keys()) # 不一定vlm, proprio，action都在;有可能action不在
    if kv_caches is None:
        kv_caches = {}
    q_all = {}
    k_all = {}
    v_all = {}

    # ===== 阶段 0: 非 active 的 K/V 从 cache 读 (仅 append_non_active) =====
      # 必须先填非 active, 再填 active, 保证 cat 顺序 = [vlm, proprio, action]
      # 与 attention_mask 的列顺序对齐
      # pi0 去噪阶段，传入的embeds_all只有'action', 需要从kv cache中取vlm/proprio中的k v
    if cache_mode == "append_non_active":
        for name, kv_cache in kv_caches.items():
            if name not in active_mixture_names:
                k_all[name], v_all[name] = kv_cache.get(layer_idx)

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

        #  2) RoPE 前的准备cos , sin
        cos, sin = mixtures[name].attn_func(
            "forward_rotary_emb", layer_idx, x, position_ids_all[name]
        )
        # 计算Q, 并旋转Q进行相对位置编码
        q = mixtures[name].attn_func(
            "forward_q_proj", layer_idx, x
        )
        q_all[name] = mixtures[name].attn_func(
            "forward_apply_rotary_emb", layer_idx, q, cos, sin
        )
        # K/V 决策
        flag_cached_mixture = (name in kv_caches) \
                        and kv_caches[name].has_item(layer_idx) # 这个 mixture 的这一层，cache 里现在有没有 K/V？
        flag_calc_new_kv = (not flag_cached_mixture) or (cache_mode == "append") # 需要新算K/V吗？
        flag_to_cache_mixture = ( (name in kv_caches) and (not kv_caches[name].has_item(layer_idx)) ) or \
                                    (cache_mode == "append") # 这次算出的新 K/V 要写回 cache 吗?


        if flag_calc_new_kv:
            # 计算新的K, V
            k_new = mixtures[name].attn_func(
                "forward_k_proj", layer_idx, x
            )
            v_new = mixtures[name].attn_func(
                "forward_v_proj", layer_idx, x
            )
            # k RoPE旋转
            k_new = mixtures[name].attn_func(
                "forward_apply_rotary_emb", layer_idx, k_new, cos, sin
            )
            if flag_to_cache_mixture:
                kv_caches[name].update(k_new, v_new, layer_idx)
        else:
            k_new , v_new = None, None

        # 拼接最终用的K/V:是否需要拼接k_cached/ v_cached
        if flag_cached_mixture: # 之前就有这一层了, 说明是文本LLM推理时的 自回归的形式
            # [B, h, T, head_dim]
            k_cached, v_cached = kv_caches[name].get(layer_idx)
            k = torch.cat([k_cached, k_new], dim = -2) if k_new is not None else k_cached
            v = torch.cat([v_cached, v_new], dim = -2) if v_new is not None else v_cached
        else:
            k, v = k_new, v_new
        k_all[name] = k
        v_all[name] = v

    # 3) 由于使用GQA, 这里需要 扩展 K, V
    for name in k_all:
        k_all[name], v_all[name] = mixtures[name].attn_func(
            "repeat_kv", layer_idx, k_all[name], v_all[name]
        )

    # TODO : 阶段 2：cat → 联合 attention → split → o_proj
    # (B, num_q_heads/num_kv_heads, T, head_dim), 按照T这个维度 拼接
    '''
    ⚠️ 两个隐性前提：
    - 三个 expert num_heads 必须一样（否则 H 维不对齐）
    - 三个 expert head_dim 必须一样（否则 D 维不对齐）
    '''
    query_states = torch.cat([q_all[name] for name in q_all], dim=-2)
    key_states = torch.cat([k_all[name] for name in k_all], dim=-2)
    value_states = torch.cat([v_all[name] for name in v_all], dim=-2)
    head_dim = query_states.shape[-1]
    attn_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(head_dim)

    # 加soft capping
    attn_scores = attn_scores / attn_softclamp
    attn_scores = torch.tanh(attn_scores)
    attn_scores = attn_scores * attn_softclamp

    # 加mask, 这里的attention_mask的形状 [B, 1, T, T]
    attn_scores = attn_scores + attention_mask
    # 加 softmax
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
        layer_idx: int,
        time_cond: Optional[torch.FloatTensor] = None,
        kv_caches: dict[KVCache] = {},
        cache_mode: str = "no_append"
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
            embeds_all[name], time_cond
        )
    hidden_states_pre_attn = hidden_states_input_norm
    ''' hidden_states_pre_attn['vlm'].shape
        torch.Size([2, 276, 2048])

        hidden_states_pre_attn['proprio'].shape
        torch.Size([2, 1, 1024])
        '''
    # 2)再做attn
    hidden_states_post_attn = forward_mixture_attn(
        mixtures = mixtures,
        attention_mask = attention_mask,
        position_ids_all = position_ids_all,
        embeds_all = hidden_states_pre_attn,
        layer_idx = layer_idx,
        kv_caches = kv_caches,
        cache_mode = cache_mode,
        )

    # hidden_states_post_attn [B, T, hidden_size]
    # 3) 做残差连接: 先做adaptive_scale，再连接残差
    hidden_states_post_res = {}
    for name in active_mixture_names:
        hidden_states_post_attn[name] = mixtures[name].layer_func(
            "forward_adaptive_scale", layer_idx,
            "post_attn", hidden_states_post_attn[name], time_cond,
        )
        hidden_states_post_res[name] = residuals_pre_attn[name] + hidden_states_post_attn[name]
    residuals_pre_mlp = hidden_states_post_res

    # 4) 再做post_attn_layernorm + MLP
    hidden_states_post_norm = {}
    hidden_states_post_mlp = {}
    for name in active_mixture_names:
        hidden_states_post_norm[name] = mixtures[name].layer_func(
            "forward_norm", layer_idx, "post_attention_layernorm", hidden_states_post_res[name], time_cond,
        )
        hidden_states_post_mlp[name] = mixtures[name].layer_func(
            "mlp", layer_idx, hidden_states_post_norm[name]
        )

    # 5) 残差合并: 先把MLP的结果做一次adaptive_scale， 再连接残差
    hidden_states_final = {}
    for name in active_mixture_names:
        hidden_states_post_mlp[name] = mixtures[name].layer_func(
            "forward_adaptive_scale", layer_idx,
            "final", hidden_states_post_mlp[name], time_cond
        )
        hidden_states_final[name] = residuals_pre_mlp[name] + hidden_states_post_mlp[name]

    return hidden_states_final


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("config/yuepi0.yaml")
    model = JointModel(cfg.joint.config)

    # dummy inputs
    dummy_num_image_tokens = 7
    q_lens = [
        dummy_num_image_tokens,
        cfg.cond_steps,
        cfg.horizon_steps,
    ]  # not considering text (padding)
    total_len = sum(q_lens)
    inputs_embeds = torch.randn(
        1,
        dummy_num_image_tokens,
        cfg.mixture.vlm.hidden_size,
    )  # no history
    proprio_embeds = torch.randn(
        1,
        cfg.cond_steps,
        cfg.mixture.proprio.hidden_size,
    )
    action_embeds = torch.randn(
        1,
        cfg.horizon_steps,
        cfg.mixture.action.hidden_size,
    )
    time_cond = None
    if cfg.action_expert_adaptive_mode:
        time_cond = torch.randn(1, cfg.time_hidden_size)

    kv_caches = model.build_mixture_caches()
    position_ids_all = {
        "vlm": torch.arange(dummy_num_image_tokens)[None],
        "proprio": torch.arange(cfg.cond_steps)[None],
        "action": torch.arange(cfg.horizon_steps)[None],
    }  # add batch dim

    # block attention
    proprio_start = dummy_num_image_tokens
    proprio_end = dummy_num_image_tokens + 1
    action_start = proprio_end
    causal_mask = torch.full(
        (1, total_len, total_len),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
    )  # smallest value, avoid using inf for softmax nan issues with padding
    causal_mask[:, :dummy_num_image_tokens, :dummy_num_image_tokens] = (
        0  # image/text attend to itself
    )
    causal_mask[:, proprio_start:proprio_end, :dummy_num_image_tokens] = (
        0  # proprio attend to image/text
    )
    causal_mask[:, action_start:, :dummy_num_image_tokens] = (
        0  # action attend to image/text
    )
    causal_mask[:, proprio_start:proprio_end, proprio_start:proprio_end] = (
        0  # proprio attend to itself
    )
    causal_mask[:, action_start:, proprio_start:] = (
        0  # action attend to itself and proprio
    )

    # Add the head dimension
    # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
    causal_mask = causal_mask.unsqueeze(1)

    # dummy denoising - naive action inference
    print("Initial action embeds", action_embeds)
    num_step = 3
    for _step in range(num_step):
        print("running dummy denoising step", _step)
        action_embeds = model(
            attention_mask=causal_mask,
            position_ids_all=position_ids_all,
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
                "action": action_embeds,
            },
            kv_caches=kv_caches,
            time_cond=time_cond,
            cache_mode="no_append",
        )["action"]
        print("Updated action embeds", action_embeds)