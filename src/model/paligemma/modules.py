from torch import nn
import torch

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim)) # 创建全0张量, 包装成可学习参数

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True)+self.eps)
    
    def forward(self, x):
        # 输入x 形状(B, T, D)
        input_dtype = x.dtype
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())

        return output.type_as(x)

# --------------------------------------------------------------

class GemmaRoPE(nn.Module):
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

        
    def forward(self, qk, position_ids):
        # 输入qk : (B, H, T, D), 输入必须是Q 和 K
        # 输入position_ids: (B, T)
        input_dtype = qk.dtype
        B = position_ids.shape[0]
        inv_freqs_expanded = self.inv_freqs[None, :, None].expand(B, -1, 1) # 从(dim/2) 拓展到[B, dim/2, 1]维
        position_ids_expanded = position_ids[:, None, :] # 变成(B, 1, T)的维度
        # 相乘得到(B, dim/2, T的维度), 再交换1, 2两个维度, 变成 ( B, T, dim/2)
        angers = (inv_freqs_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        angers = torch.cat((angers, angers), dim=-1)
        cos = angers.cos() 
        sin = angers.sin()
        return cos.to(qk.dtype), sin.to(qk.dtype)





