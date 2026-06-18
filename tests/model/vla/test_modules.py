import torch
import pytest
from model.vla.modules import build_blockwise_causal_mask, TimeEncoder, ActionEncoder, AdaptiveRMSNorm

# ========== 用例 1: 形状 + 区块值 ==========
def test_build_blockwise_causal_mask():
    m = build_blockwise_causal_mask(3, 2, 2, 2)
    assert m.shape == (2, 1, 7, 7)
    mat = m[0, 0]  # 7,7
    NEG = torch.finfo(torch.float32).min

    # 1) VLM → VLM 区块全是 0（双向可见）
    assert (mat[0:3, 0:3] == 0).all()

    # 2) VLM → Proprio + Action 全是 -inf（VLM 看不到右边）
    assert (mat[0:3, 3:7] == NEG).all()

    # 3) Proprio → VLM 全是 0（Proprio 能看 VLM）
    assert (mat[3:5, 0:3] == 0).all()

    # 4) Proprio → Proprio 全是 0
    assert (mat[3:5, 3:5] == 0).all()

    # 5) Proprio → Action 全是 -inf
    assert (mat[3:5, 5:7] == NEG).all()

    # 6) Action → VLM + Proprio 全是 0（Action 能看左边所有）
    assert (mat[5:7, 0:5] == 0).all()

    # 7) Action → Action 是下三角
    # A0 看 A0, 不看 A1：mat[5, 6] == NEG, mat[5, 5] == 0
    # A1 看 A0, A1：mat[6, 5] == 0, mat[6, 6] == 0
    assert mat[5, 5] == 0
    assert mat[5, 6] == NEG
    assert mat[6, 5] == 0
    assert mat[6, 6] == 0


# ========== 用例 2: Action 内部因果性（A 设大一点） ==========
def test_action_internal_causal():
    """Action 子方块: 严格上三角 = -inf, 下三角(含对角) = 0"""
    V, P, A, B = 2, 1, 4, 1
    m = build_blockwise_causal_mask(V, P, A, B)
    mat = m[0, 0]
    NEG = torch.finfo(torch.float32).min

    action_block = mat[V + P:, V + P:]  # (4, 4)

    upper_mask = torch.triu(torch.ones(A, A), diagonal=1).bool()
    lower_mask = ~upper_mask

    assert (action_block[upper_mask] == NEG).all(), "严格上三角应该全是 -inf"
    assert (action_block[lower_mask] == 0).all(), "下三角(含对角线)应该全是 0"


# ========== 用例 3: 批次一致性 ==========
def test_batch_consistency():
    """expand 出来的每个 batch 应该完全一致"""
    B = 4
    m = build_blockwise_causal_mask(3, 2, 2, B)
    assert m.shape == (B, 1, 7, 7)

    for i in range(1, B):
        assert torch.equal(m[0, 0], m[i, 0]), f"batch {i} 与 batch 0 不一致"


# ========== 用例 4: 端到端 softmax 验证 ⭐ ==========
def test_mask_with_softmax():
    """把 mask 喂给 softmax, 验证 attention 权重的屏蔽效果"""
    V, P, A = 3, 2, 2
    T = V + P + A
    m = build_blockwise_causal_mask(V, P, A, 1)
    mat = m[0, 0]  # (7, 7)

    # 模拟 attention scores (全 1)
    scores = torch.ones(T, T)
    scores_masked = scores + mat
    attn_weights = torch.softmax(scores_masked, dim=-1)

    # 1) 每行权重和 = 1 (softmax 性质)
    assert torch.allclose(
        attn_weights.sum(dim=-1), torch.ones(T), atol=1e-5
    ), "softmax 每行应该归一"

    # 2) row 0 (VLM) 对 col 3..7 (右侧) 的权重 ≈ 0
    assert (attn_weights[0, V:] < 1e-6).all(), "VLM 不应该 attend 到右侧"

    # 3) row 3 (Proprio) 对 col 5..7 (Action) 的权重 ≈ 0
    assert (attn_weights[V, V + P:] < 1e-6).all(), "Proprio 不应该 attend 到 Action"

    # 4) Action 内部 causal: row 5 对 col 6 的权重 ≈ 0
    assert attn_weights[V + P, V + P + 1] < 1e-6, "Action[0] 不应该 attend 到 Action[1]"

    # 5) Action[1] 能看到 Action[0]: 权重 > 0
    assert attn_weights[V + P + 1, V + P] > 0, "Action[1] 应该能看到 Action[0]"


# ========== 用例 5: 不同 dtype ==========
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_dtype_support(dtype):
    """不同 dtype 都能正确构造 mask"""
    m = build_blockwise_causal_mask(3, 2, 2, 1, dtype=dtype)
    assert m.dtype == dtype

    mat = m[0, 0]
    NEG = torch.finfo(dtype).min

    # 0 的位置
    assert mat[0, 0] == 0
    # -inf 的位置
    assert mat[0, 3] == NEG


# --------------------------------测试time_encoder-------------------------
def test_shape():
    enc = TimeEncoder(dim=64)
    t = torch.rand(4)  
# (B,)

    out = enc(t)
    assert out.shape == (4, 64)

def test_finite():
    enc = TimeEncoder(dim=64)
    t = torch.rand(4)
    out = enc(t)
    assert torch.isfinite(out).all()

def test_different_t_gives_different_emb():
    """不同的 t 应该产生不同的 embedding"""
    enc = TimeEncoder(dim=64)
    t = torch.tensor([0.1, 0.5, 0.9])
    out = enc(t)
    
# 任意两个 sample 不应该完全相同

    assert not torch.allclose(out[0], out[1])
    assert not torch.allclose(out[1], out[2])

def test_same_t_gives_same_emb():
    """相同的 t 应该产生完全一样的 embedding（确定性）"""
    enc = TimeEncoder(dim=64)
    t = torch.tensor([0.5, 0.5])
    out = enc(t)
    assert torch.allclose(out[0], out[1])

def test_value_range():
    """因为是 sin/cos，所有值应该在 [-1, 1]"""
    enc = TimeEncoder(dim=64)
    t = torch.linspace(0, 1, 100)
    out = enc(t)
    assert (out >= -1).all() and (out <= 1).all()

def test_no_learnable_params():
    """TimeEncoder 应该是无参数的纯函数"""
    enc = TimeEncoder(dim=64)
    assert sum(p.numel() for p in enc.parameters()) == 0


# -----------------------------测试Action Encoder---------------------------
def test_action_encoder_shape():                          
    """(B, H, action_dim) → (B, H, hidden_size)"""                                   
    enc = ActionEncoder(action_dim=7, hidden_size=1024)   
    actions = torch.randn(2, 4, 7)  # B=2, horizon=4, action_dim=7                   
    out = enc(actions)                                                               
    assert out.shape == (2, 4, 1024)                                                 
    assert torch.isfinite(out).all()

# -----------------------------测试AdaptiveRMSNorm---------------------------
def test_adaptive_rmsnorm_shape():
    norm = AdaptiveRMSNorm(hidden_size=64, time_dim=128)
    x = torch.randn(2, 8, 64)
    t_emb = torch.randn(2, 128)
    out = norm(x, t_emb)
    assert out.shape == (2, 8, 64)

def test_zero_init_behaves_like_rmsnorm():
    """初始化时（scale=0, shift=0），输出应该等于纯 RMSNorm 的结果"""
    norm = AdaptiveRMSNorm(hidden_size=64, time_dim=128)
    x = torch.randn(2, 8, 64)
    t_emb = torch.randn(2, 128)
    out = norm(x, t_emb)
    # 手动算 RMSNorm
    rms = x.pow(2).mean(-1, keepdim=True).sqrt()
    x_normed = x / (rms + 1e-6)

    assert torch.allclose(out, x_normed, atol=1e-5)