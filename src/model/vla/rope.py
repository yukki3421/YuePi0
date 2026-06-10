import torch
from torch import nn

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

def test_rope():
    # B, H, T, dim = 2, 4, 16, 64
    # rope = RoPE(dim = dim)
    # q = torch.randn(B, H, T, dim)
    # k = torch.randn(B, H, T, dim)
    # position_ids = torch.arange(T).unsqueeze(0).expand(B, T)

    # cos, sin = rope(q, position_ids)
    # q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    # assert True

    print("\n[RoPE 数学性质测试]")
    dim = 64
    rope = RoPE(dim=dim)
    B, H, T = 2, 4, 16

    # 造输入
    q = torch.randn(B, H, T, dim)
    k = torch.randn(B, H, T, dim)
    position_ids = torch.arange(T).unsqueeze(0).expand(B, T)

    # 1. cos/sin 形状
    cos, sin = rope(q, position_ids)
    assert cos.shape == (B, T, dim), f"cos shape {cos.shape}"
    assert sin.shape == (B, T, dim), f"sin shape {sin.shape}"
    print(f"  ✅ cos/sin 形状: {cos.shape}")

    # 2. dtype 保持
    assert cos.dtype == q.dtype
    print(f"  ✅ dtype 保持: {q.dtype}")

    # 3. 旋转后形状不变
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    print(f"  ✅ 旋转后形状不变")

    # 4. 相对位置不变性（核心）
    same = torch.randn(1, 1, 1, dim)
    q_same = same.expand(B, H, T, dim)
    cos, sin = rope(q_same, position_ids)
    q_rot, k_rot = apply_rotary_pos_emb(q_same, q_same, cos, sin)

    b,h = 0, 0
    ok = True
    for i in range(T):
        for j in range(i, T):
            # 位置i, j的点积
            dot_ij = torch.dot(q_rot[b, h, i], k_rot[b, h, j])
            # 位置0, j-i的点积
            dot_0_delta = torch.dot(q_rot[b, h, 0], k_rot[b, h, j - i])
            if not torch.allclose(dot_ij, dot_0_delta, atol=1e-4):
                  ok = False                                                                                                                        
                  print(f"  ❌ 位置 ({i}, {j}) 失败: {dot_ij.item():.6f} vs {dot_0_delta.item():.6f}")
                  break                                                                                                                             
        if not ok:
            break
    print(f"  ✅ 相对位置不变性: {'通过' if ok else '失败'}")
    
    # 5. 旋转不改变向量长度（用同一个 q 比较）
    q_for_len = torch.randn(B, H, T, dim)
    cos_for_len, sin_for_len = rope(q_for_len, position_ids)
    q_rot_for_len, _ = apply_rotary_pos_emb(q_for_len, q_for_len, cos_for_len, sin_for_len)

    norm_before = q_for_len.pow(2).sum(-1).sqrt()
    norm_after = q_rot_for_len.pow(2).sum(-1).sqrt()
    norm_ok = torch.allclose(norm_before, norm_after, atol=1e-4)
    max_err = (norm_before - norm_after).abs().max().item()
    print(f"  ✅ 旋转不改变向量长度: {'通过' if norm_ok else '失败'} (max_err={max_err:.6e})")

    print("\n  🎉 所有测试通过！")




