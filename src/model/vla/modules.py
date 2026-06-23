import torch
from torch import nn
import math

'''
t 是 forward 的输入条件，不是参数。同一个   
  x_t 在 t=0.1 和 t=0.9 时，预测的 v 应该是不一样的（因为离终点的距离不同）。所以必须把 t注入到模型每一处需要"感知噪声水平"的地方。
time_cond 这个开关，就是控制 ActionEncoder这一层要不要也接收时间信号。
'''
class ActionEncoder(nn.Module):

    def __init__(self, action_dim, hidden_size, time_cond=False):
        super().__init__()
        # 编码动作
        self.linear_1 = nn.Linear(action_dim, hidden_size)
        self.time_cond = time_cond
        if time_cond:
            # 把[actio, time_emb] 拼起来再投影, 假设time_dim == action_dim
            self.linear_2 = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.nonlinearity = nn.SiLU()
        self.linear_3 = nn.Linear(hidden_size, hidden_size)        
    
    def forward(self, action, time_emb=None):
        # time_emb：[B, time_dim]
        # action: [B, seq_len, hidden_size]
        emb = self.linear_1(action)
        if self.time_cond:  #将time_emb拓展成(B, 1, time_dim)->(B, T, time_dim)
            time_emb_expand = time_emb.unsqueeze(1).expand(-1, action.size(1), -1)
            emb = torch.cat([emb, time_emb_expand], dim=-1)
        emb = self.nonlinearity(self.linear_2(emb))
        emb = self.linear_3(emb)

        return emb


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
    
class AdaptiveLayerscale(nn.Module):
    def __init__(self, dim: int, dim_cond: int, adaln_zero_bias_init_value: float=-2.0):
        super().__init__()
        adaln_zero_gamma_linear = nn.Linear(dim_cond, dim)
        nn.init.zeros_(adaln_zero_gamma_linear.weight)
        nn.init.constant_(adaln_zero_gamma_linear.bias, adaln_zero_bias_init_value) # 偏置初始化为一个常数
        self.to_adaln_zero_gamma = adaln_zero_gamma_linear

    def forward(self, x: torch.FloatTensor, cond: torch.FloatTensor) -> torch.FloatTensor:
        if cond.ndim == 2:
            cond = cond.unsqueeze(1)
        gamma = self.to_adaln_zero_gamma(cond)
        return x * gamma.sigmoid()


class ProprioEncoder(nn.Module):
    def __init__(self, proprio_dim, proprio_hidden_size):
        super().__init__()
        self.proj = nn.Linear(proprio_dim, proprio_hidden_size)

    def forward(self, proprios):
        return self.proj(proprios)

class ActionDecoder(nn.Module):
    def __init__(self, action_hidden_size, action_dim):
        super().__init__()
        self.proj = nn.Linear(action_hidden_size, action_dim)
    def forward(self, action_hidden):
        return self.proj(action_hidden)

def build_blockwise_causal_mask(
    num_vlm_tokens: int, 
    num_proprio_tokens: int, 
    num_action_tokens: int,
    batch_size: int, 
    dtype = torch.float32,
    device = None
) -> torch.Tensor:
    '''返回(B, 1, T, T)的attention mask, 可见处为0, 屏蔽处为 -inf'''
    """
        构造 Pi0 特有的 block-wise causal mask（不是普通的下三角因果 mask！）

        三段 token 的可见性规则（✓ 表示可以 attend 到）：
                     img/text img/text img/text (padding) proprio action action
            img/text    ✓        ✓        ✓                            (VLM 内部全互看，但看不到 proprio/action)
            img/text    ✓        ✓        ✓
            img/text    ✓        ✓        ✓
            (padding)                                                  (padding 行整行被屏蔽)
            proprio     ✓        ✓        ✓                 ✓          (proprio 看 VLM + 自己)
            action      ✓        ✓        ✓                 ✓     ✓     ✓ (action 看所有，并且 action 内部双向)
            action      ✓        ✓        ✓                 ✓     ✓     ✓

        注意：action 内部是 **双向** 的（不是自回归），因为 Flow Matching 一次性预测整段 horizon。

        输入：
            attention_mask: [B, max_image_text_tokens]  原始的 1/0 mask（1=有效 text/image，0=padding）
        输出：
            causal_mask:           [B, 1, total_tokens, total_tokens]  其中 0 表示可见，dtype.min 表示屏蔽（加在 softmax 前）
            vlm_position_ids:      [B, max_image_text_tokens]
            proprio_position_ids:  [B, num_proprio_tokens]
            action_position_ids:   [B, num_action_tokens]
        """

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