from torch import nn
import torch

class MyRMSNorm(nn.Module):
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

if __name__ == "__main__":
    x = torch.randn(2, 8, 64)
    rmsnorm = MyRMSNorm(dim=64)
    # rmsnorm.weight.data.fill_(1.0)
    y = rmsnorm(x)
    # 测试1： 归一化后的rms == 1
    y_rms = y.pow(2).mean(-1).sqrt()
    print(torch.allclose(y_rms, torch.ones_like(y_rms), atol=1e-5))
    # 测试2：缩放不变性
    y1 = rmsnorm(x)
    y2 = rmsnorm(10.0 * x)
    print(torch.allclose(y1, y2, atol=1e-5))
    # 测试3：weight全1, weight全0之间的倍数关系
    rmsnorm.weight.data.fill_(1.0)
    y3 = rmsnorm(x)
    rmsnorm.weight.data.fill_(0)
    y4 = rmsnorm(x)
    print(torch.allclose(y3, y4*2, atol=1e-5))
    # 测试4：梯度可传
    y.sum().backward() # 反向传播
    print(rmsnorm.weight.grad is not None)
    print(rmsnorm.weight.grad.shape == (64, ))

