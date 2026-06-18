import torch
from torch import nn
import math
from typing import Optional, Tuple
from .modules import GemmaRMSNorm, GemmaRoPE
from ..utils import apply_rotary_pos_emb, repeat_kv

class GemmaAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, rope_theta, layer_idx=0):
        super().__init__() 
        assert hidden_size == num_heads * head_dim
        assert num_heads > num_kv_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads # k v的个数, 也是分组的组数
        self.num_kv_groups = num_heads // num_kv_heads # 每组的头数

        self.q_proj = nn.Linear(hidden_size, num_heads*head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads*head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads*head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads*head_dim, hidden_size, bias=False)
        
        self.rotary_emb = GemmaRoPE(head_dim, theta=rope_theta)
        self.layer_idx = layer_idx # 第几层attention

    def forward(self, hidden_states, attention_mask, position_ids, kv_cache=None):
        # 输入形状：hidden_states （B, T, hidden_size)
        B, T, D = hidden_states.shape

        # 1. Q, K, V投影, reshape + transpose
        # (B, T, H*head_dim) -> (B, H, T, head_dim)
        query = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 2. RoPE旋转（cos, sin形状为(B, T, head_dim)
        cos, sin = self.rotary_emb(query, position_ids)
        query, key = apply_rotary_pos_emb(query, cos, sin), apply_rotary_pos_emb(key, cos, sin)

        # KV Cache
        if kv_cache is not None:
            # 调用实例的 update 方法                                                                           
            # 传入新的 key/value，更新 cache 内部状态                                                          
            # 返回完整的 key/value（旧的 + 新的拼接） 
            key, value = kv_cache.update(key, value, self.layer_idx)
        
        # 3. K, V扩展: 从num_kv_heads -> num_heads
        # 扩展倍数就是每组的注意力头数
        # key,value: (B, num_kv_heads, T, head_dim)  -> (B, num_heads, T, head_dim)
        key = repeat_kv(key, self.num_kv_groups)
        value = repeat_kv(value, self.num_kv_groups)

        # 4. 计算注意力
        # (B, num_heads, T, head_dim) @ (B, num_heads, head_dim, T)
        att_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
        # 加attention_mask，是一个三角矩阵, 右上角全为-inf的矩阵
        assert attention_mask is not None
        att_weights = att_weights + attention_mask
        # 再来softmax
        att_weights = nn.functional.softmax(att_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        
        # 5.输出output 
        # (B, num_heads, T, head_dim) -> (B, T, num_heads, head_dim) -> (B, T, hidden_size)
        # 再投影
        output = att_weights @ value
        output = output.transpose(1, 2).contiguous().view(B, T, self.num_heads*self.head_dim)
        att_output = self.o_proj(output)
        return att_output, att_weights

class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        x1 = nn.functional.gelu(self.gate_proj(x), approximate="tanh")
        x2 = self.up_proj(x)
        return self.down_proj(x1 * x2)

class GemmaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx:int):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.rope_theta = config.rope_theta

        self.self_attn = GemmaAttention(self.hidden_size, self.num_heads, self.num_kv_heads, self.head_dim, self.rope_theta, layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(self.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(self.hidden_size, config.rms_norm_eps)
    
    def forward(self, hidden_state, attention_mask, position_ids, kvcache=None):
        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)
        # Attention需要的参数：hidden_states, attention_mask, position_ids, kv_cache=None
        hidden_state, _ = self.self_attn(hidden_state, attention_mask, position_ids, kvcache)
        hidden_state = residual + hidden_state # 残差连接
        
        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state) + residual
    
        return hidden_state
# 主要为了 state_dict key 对齐 PaliGemma。
class GemmaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(
            config.vocab_size, 
            config.hidden_size,
            self.padding_idx
        )

        self.layers = nn.ModuleList([
            GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = GemmaRMSNorm(self.hidden_size, config.rms_norm_eps)

    def forward(self, attention_mask, position_ids, inputs_embedding, kvcache=None):
        # [B, T, hidden_size]
        hidden_state = inputs_embedding
        normalizer = torch.tensor(self.hidden_size ** 0.5, dtype=hidden_state.dtype)
        hidden_state = hidden_state * normalizer

        for layer in self.layers:
            # [B, T, hidden_size]
            hidden_state = layer(hidden_state, attention_mask, position_ids, kvcache=kvcache)
        hidden_state = self.norm(hidden_state)
        return hidden_state

# GemmaForCausalLM —— 包在 GemmaModel 外面，加一个 lm_head 输出 logits
class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model  = GemmaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, attention_mask, position_ids, inputs_embedding, kvcache=None):
        hidden_state = self.model(attention_mask, position_ids, inputs_embedding, kvcache)
        logits = self.lm_head(hidden_state)
        return {"logits": logits}

