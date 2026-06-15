import torch
from torch import nn
import math
from typing import Optional, Tuple
from model.vla.rope import RoPE, apply_rotary_pos_emb


class RoPE(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta
        
        # 计算出每对维度的 "基础频率"
        # 形状为(dim/2,)
        inv_freqs = 1.0 / (theta **(torch.arange(0, dim, 2).float() / dim))

        # 注册为buffer， 因为不是可学习参数.register_buffer() 是 PyTorch 中 nn.Module 的核心方法，
        # 专门用来给模型注册不需要梯度更新、
        # 但需要和模型一起保存 / 加载、且能在 GPU/CPU 之间自动迁移的张量。
        self.register_buffer("inv_freqs", inv_freqs, persistent=False)

        
    def forward(self, x, position_ids):
        # 输入x : (B, H, T, D)
        # 输入position_ids: (B, T)
        input_dtype = x.dtype
        B = position_ids.shape[0]
        inv_freqs_expanded = self.inv_freqs[None, :, None].expand(B, -1, 1) # 从(dim/2) 拓展到[B, dim/2, 1]维
        position_ids_expanded = position_ids[:, None, :] # 变成(B, 1, T)的维度
        # 相乘得到(B, dim/2, T的维度), 再交换1, 2两个维度, 变成 ( B, T, dim/2)
        angers = (inv_freqs_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        angers = torch.cat((angers, angers), dim=-1)
        cos = angers.cos() 
        sin = angers.sin()
        return cos, sin

def rotate_half(x):
    dim = x.shape[-1]
    x_2 = x[...,  dim//2:] # python的省略号切片, 就是前面所有维度全取
    x_1 = x[..., :dim//2]
    return torch.cat((-x_2, x_1), dim=-1)

# 输出旋转后的q, k
# 输入q, k 的形状: (B， H, T, D)
# cos，sin的形状：（B, T， D)
def apply_rotary_pos_emb(q, k, cos, sin):
    # 增加一个H维度
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    # 旋转公式
    q_rot = q*cos + rotate_half(q)*sin
    k_rot = k*cos + rotate_half(k)*sin
    return q_rot, k_rot


'''将(B, H_kv, T, D_h) 扩展成(B, H_Q, T, D_h), 其中H_Q = H_KV x G'''
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # expand只能扩展大小为1的维度，对于其他维度大小保持不变
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(
        batch,
        num_key_value_heads * n_rep,
        slen,
        head_dim,
    )
class GroupedQAttention(nn.Module):
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
        
        self.rotary_emb = RoPE(head_dim, theta=rope_theta)
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
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

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

def verify():
    from model.paligemma.gemma import GemmaAttention
    torch.manual_seed(0)
    B, T, D = 2, 8, 512
    num_heads = 32
    num_kv_heads = 8
    head_dim = 16
    rope_theta = 10000.0

    class FakeConfig:
        pass
    FakeConfig.hidden_size = D
    FakeConfig.num_attention_heads = num_heads
    FakeConfig.num_key_value_heads = num_kv_heads
    FakeConfig.head_dim = head_dim
    FakeConfig.rope_theta = rope_theta
    FakeConfig.attention_dropout = 0.0                                                                                 
    FakeConfig.attention_bias = False
    FakeConfig.rms_norm_eps = 1e-6 

    orig = GemmaAttention(FakeConfig(), layer_idx=0)
    mine = GroupedQAttention(D, num_heads, num_kv_heads, head_dim, rope_theta)
    
    # 拷贝原版的权重
    orig_sd = orig.state_dict()
    mine_sd = mine.state_dict()
    print("=== key 对比 ===")
    print(f"orig keys:  {list(orig_sd.keys())}")
    print(f"mine keys:  {list(mine_sd.keys())}")

    print("\n=== 检查哪些 key 对不上 ===")
    for k in orig_sd:
        if k not in mine_sd:
            print(f"  ❌ mine 缺少: {k}")
        elif orig_sd[k].shape != mine_sd[k].shape:
            print(f"  ⚠️   shape 不等: {k}  orig={orig_sd[k].shape} mine={mine_sd[k].shape}")

    mine.load_state_dict(orig.state_dict(), strict=False)

    # 构造输入
    x = torch.randn(B, T, D)
    position_ids = torch.arange(T).unsqueeze(0).expand(B, T)
    mask = torch.triu(torch.ones(T, T), diagonal=1)*(-1e9)

    y_orig = orig(x, attention_mask=mask, position_ids=position_ids)[0]
    y_mine = mine(x, attention_mask=mask, position_ids=position_ids)[0]
    
    ok = torch.allclose(y_mine, y_orig, atol=1e-5)
    diff = (y_mine - y_orig).abs().max().item()
    print(f"allclose={ok} max_diff={diff:.2e}")
    assert ok, "未对齐!"
if __name__ == "__main__":
    verify()