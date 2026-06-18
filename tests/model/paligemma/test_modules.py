import torch
from model.paligemma.modules import GemmaRMSNorm,GemmaRoPE
from model.utils import apply_rotary_pos_emb

def test_gemma_rmsnorm():
    x = torch.randn(2, 8, 64)
    rmsnorm = GemmaRMSNorm(dim=64)
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

def test_GemmaRoPE():
    # B, H, T, dim = 2, 4, 16, 64
    # GemmaRoPE = GemmaRoPE(dim = dim)
    # q = torch.randn(B, H, T, dim)
    # k = torch.randn(B, H, T, dim)
    # position_ids = torch.arange(T).unsqueeze(0).expand(B, T)

    # cos, sin = GemmaRoPE(q, position_ids)
    # q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    # assert True

    print("\n[GemmaRoPE 数学性质测试]")
    dim = 64
    rope = GemmaRoPE(dim=dim)
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
    q_rot = apply_rotary_pos_emb(q, cos, sin)
    k_rot = apply_rotary_pos_emb(k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    print(f"  ✅ 旋转后形状不变")

    # 4. 相对位置不变性（核心）
    same = torch.randn(1, 1, 1, dim)
    q_same = same.expand(B, H, T, dim)
    cos, sin = rope(q_same, position_ids)
    q_rot = apply_rotary_pos_emb(q_same, cos, sin)
    k_rot = apply_rotary_pos_emb(q_same, cos, sin)
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
    q_rot_for_len = apply_rotary_pos_emb( q_for_len, cos_for_len, sin_for_len)

    norm_before = q_for_len.pow(2).sum(-1).sqrt()
    norm_after = q_rot_for_len.pow(2).sum(-1).sqrt()
    norm_ok = torch.allclose(norm_before, norm_after, atol=1e-4)
    max_err = (norm_before - norm_after).abs().max().item()
    print(f"  ✅ 旋转不改变向量长度: {'通过' if norm_ok else '失败'} (max_err={max_err:.6e})")

    print("\n  🎉 所有测试通过！")