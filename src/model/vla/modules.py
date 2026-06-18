import torch
from torch import nn
import math

class AdaptiveRMSNorm:
    pass

class ActionEncoder(nn.Module):

    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.proj = nn.Linear(action_dim, hidden_size)
    
    def forward(self, actions):
        return self.proj(actions)


class TimeEncoder(nn.Module):
    # flow matching 训练时，每个 sample 都有一个 t ∈ [0, 1]，告诉模型 "现在是噪声多还是少"。
    # 输入t：（B）
    # 输出： time_emb  shape: (B, time_dim)
    # 实现是 sinusoidal positional embedding（跟 Transformer 位置编码同一个数学）
    def __init__(self, dim, max_period=10000.0):
        super().__init__()
        self.half_dim = dim // 2
        self.max_period = max_period
    
    # 把一个标量 t 映射成 dim 维的稠密向量。每个sample一个时间步
    def forward(self, t):
        i = torch.arange(self.half_dim, dtype=t.dtype, device=t.device)
        freqs = torch.exp( -math.log(self.max_period) * i / self.half_dim )
        angles = t.unsqueeze(-1) * freqs # 变成 (B, half_dim)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

class AdaptiveRMSNorm(nn.Module):
    def __init__(self, hidden_size, time_dim, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.time_dim = time_dim
        self.eps = eps
        self.scale_proj = nn.Linear(time_dim, hidden_size)
        self.shift_proj = nn.Linear(time_dim, hidden_size)

        # 初始化
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)
        
    def _norm(self, x):
        return x * torch.rsqrt( x.pow(2).mean(dim=-1, keepdim=True) + self.eps )
    
    def forward(self, x, time_emb):
        x_norm = self._norm(x)
        # out = x_normed * (1 + scale(time_emb)) + shift(time_emb)
        scale = self.scale_proj(time_emb).unsqueeze(1) #(B, 1, time_emb)
        shift = self.shift_proj(time_emb).unsqueeze(1)
        return x_norm * (1+scale) + shift
    


def build_blockwise_causal_mask(
    num_vlm_tokens: int, 
    num_proprio_tokens: int, 
    num_action_tokens: int,
    batch_size: int, 
    dtype = torch.float32,
    device = None
) -> torch.Tensor:
    '''返回(B, 1, T, T)的attention mask, 可见处为0, 屏蔽处为 -inf'''
    T_total = num_vlm_tokens + num_proprio_tokens + num_action_tokens
    mask_pre = torch.zeros(T_total, T_total, dtype=dtype)
    mask_pre[:num_vlm_tokens, num_vlm_tokens:] = torch.finfo(dtype).min
    # proprio屏蔽action
    mask_pre[num_vlm_tokens:num_vlm_tokens+num_proprio_tokens, num_vlm_tokens+num_proprio_tokens:] = torch.finfo(dtype).min
    # action内部加causal   
    A = num_action_tokens
    action_start = num_vlm_tokens + num_proprio_tokens
    action_sub = torch.zeros(A, A, dtype=dtype)
    upper = torch.triu(torch.ones(A, A), diagonal=1).bool()
    action_sub.masked_fill_(upper, torch.finfo(dtype).min)
    mask_pre[action_start:,action_start:] = action_sub
    # 拓展
    mask = mask_pre[None, None, :, :].expand(batch_size, 1, T_total, T_total)
    return mask.to(device)