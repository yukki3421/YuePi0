import torch
from torch import nn

from model.paligemma.gemma import GemmaMLP
from model.paligemma.modules import GemmaRoPE, GemmaRMSNorm
from model.utils import repeat_kv, apply_rotary_pos_emb

class MixtureAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        # Group Query Attention
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        # o_proj 是 attention 输出的最后一层，输入是 **拼回来的 attn 结果 (num_heads * head_dim)，输出回到 hidden_size
        # num_heads * head_dim 不一定等于hidden_size
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = GemmaRoPE(self.head_dim, theta=config.rope_theta)

    '''
    因为 attention 的"软糖中心" —— softmax(QKᵀ)·V 那一步 —— 不在 Mixture 里做，
    而是上层 JointModel 拿三个 expert 的 Q/K/V cat 起来一起做。
    所以 Mixture 只负责 "我自己这段 token 的投影 + 我自己这段的 RoPE + 我自己的 o_proj"，
    attention 的拼接交给上层。
    '''
    def forward_q_proj(self, x):
        B, T = x.shape[:2]
        query_state = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        return query_state

    def forward_k_proj(self, x):
        B, T = x.shape[:2]
        key_state = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return key_state

    def forward_v_proj(self, x):
        B, T = x.shape[:2]
        value_state = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return value_state

    def forward_o_proj(self, x):
        return self.o_proj(x)

    def forward_rotary_emb(self, x, position_ids):
        cos, sin = self.rotary_emb(x, position_ids)
        return cos, sin

    def forward_apply_rotary_emb(self, qk, cos, sin):
        qk_rot = apply_rotary_pos_emb(qk, cos, sin)
        return qk_rot

    def repeat_kv(self, k, v):
        key = repeat_kv(k, self.num_kv_groups)
        value = repeat_kv(v, self.num_kv_groups)
        return key, value


class MixtureDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MixtureAttention(config)
        self.mlp = GemmaMLP(config)
        # 暂时用GammaRMSNorm代替
        self.input_layernorm = GemmaRMSNorm(self.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(self.hidden_size, config.rms_norm_eps)

    def forward_norm(
        self,
        norm_name: str, # norm_name 应该为 input_layernorm or post_attention_layernorm
        x: torch.FloatTensor
    ) -> torch.FloatTensor:
        return getattr(self, norm_name)(x)


class Mixture(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            MixtureDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)

    @property
    def head_dim(self) -> int:
        return self.layers[0].self_attn.head_dim

    '''
        1. 反射式的方法派发器（reflection-based dispatcher）
        上层代码用"统一接口"调度任意一个 expert 的任意一个原子方法。

        拆解一下 getattr(obj, "method_name")(*args) 这个咒语

        Python 里这两行是等价的：
        mixture.layers[0].self_attn.forward_q_proj(x)         # 直接调
        getattr(mixture.layers[0].self_attn, "forward_q_proj")(x)  # 反射调用

        getattr(obj, "name") = "从 obj 里把名为 name 的属性/方法拿出来"，
        拿出来后加 (...) 就调用。关键是方法名变成了字符串参数，可以动态传。

        2. 注意第三个形参是 *args（前面带星号）。这不是"一个参数"，而是 "打包剩下所有位置参数到一个 tuple"。
        意思是：
        - method_name 和 layer_idx 各自捕获一个实参
        - *args 把剩下的所有实参打包成一个 tuple
    '''
    def layer_func(self, method_name:str, layer_idx:int, *args) -> torch.FloatTensor:
        args = [arg for arg in args if arg is not None]
        return getattr(self.layers[layer_idx], method_name)(*args)

    def attn_func(self, method_name:str, layer_idx:int, *args) -> torch.FloatTensor:
        args = [arg for arg in args if arg is not None]
        return getattr(self.layers[layer_idx].self_attn, method_name)(*args)

    # 整个expert最后的norm
    def forward_norm(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if hasattr(self, "norm"):
            args = [x]
            return self.norm(*args)
        return None

