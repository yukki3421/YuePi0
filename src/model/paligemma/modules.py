from torch import nn
import torch

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.epsilon = eps
        self.weight = nn.Parameter(torch.zeros(dim)) # 创建全0张量, 包装成可学习参数

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True)+self.epsilon)
    
    def forward(self, x):
        # 输入x 形状(B, T, D)
        input_dtype = x.dtype
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())

        return output.type_as(x)