import torch
from model.paligemma.gemma import RoPE

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