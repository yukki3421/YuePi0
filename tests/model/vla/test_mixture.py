import torch
import pytest
from model.vla.mixture import Mixture

class FakeConfig:
    hidden_size = 1024
    num_hidden_layers = 8
    num_heads = 16
    num_kv_heads = 4
    head_dim = 128
    intermediate_size = 4034
    rope_theta = 10000
    rms_norm_eps = 1e-6
    attention_bias = False
    use_final_norm = True

@pytest.fixture
def mix():
    return Mixture(FakeConfig())

def test_q_proj_shape(mix):
    B, T = 2, 281
    x = torch.randn(B, T, 1024)
    q = mix.attn_func("forward_q_proj", 0, x)
    assert q.shape == (B, 16, T, 128)

def test_o_proj_shape(mix):
    B, T = 2, 281
    x = torch.randn(B, T, 16*128)
    # forward_o_proj的输入是多头拼接的num_head * head_dim, 不是hidden_size
    o = mix.attn_func("forward_o_proj", 0, x)    
    assert o.shape != (B, T, 2048)
    assert o.shape == (B, T, 1024)

def test_kv_shape(mix):
    B, T = 2, 281
    x = torch.randn(B, T, 1024)
    k = mix.attn_func("forward_k_proj", 0, x)
    v = mix.attn_func("forward_v_proj", 0, x)
    assert k.shape == (B,4, T, 128)
    assert v.shape == (B, 4, T, 128)

def test_rotary_round_trip(mix):
        B, T = 2, 281
        x = torch.randn(B, T, 1024)
        pos = torch.arange(T).unsqueeze(0).expand(B, T)
    
        # 走完 q_proj → rotary → apply
        q = mix.attn_func("forward_q_proj", 0, x)         # (B, 16, T, 128)
        cos, sin = mix.attn_func("forward_rotary_emb", 0, q, pos)
        q_rot = mix.attn_func("forward_apply_rotary_emb", 0, q, cos, sin)
    
        # 形状不变
        assert q_rot.shape == q.shape
        # cos/sin 形状 (B, T, head_dim)
        assert cos.shape == (B, T, 128)
        assert sin.shape == (B, T, 128)
        # 旋转不应该把数值搞炸（NaN/Inf）
        assert torch.isfinite(q_rot).all()

def test_repeat_kv(mix):
        B, T = 2, 281
        k = torch.randn(B, 4, T, 128)   # (B, num_kv_heads, T, head_dim)
        v = torch.randn(B, 4, T, 128)
        k_rep, v_rep = mix.attn_func("repeat_kv", 0, k, v)
        # 4 → 16，扩展倍数 = num_heads / num_kv_heads = 4
        assert k_rep.shape == (B, 16, T, 128)
        assert v_rep.shape == (B, 16, T, 128)
        # 扩展后每组的 4 个头应该是同一个 KV 的复制
        assert torch.allclose(k_rep[:, 0], k_rep[:, 1])
        assert torch.allclose(k_rep[:, 0], k_rep[:, 2])
        assert torch.allclose(k_rep[:, 0], k_rep[:, 3])
        # 但下一组应该是不同的（来自 k 的下一个 kv head）
        assert not torch.allclose(k_rep[:, 0], k_rep[:, 4])

def test_norm(mix):
        B, T = 2, 281
        x = torch.randn(B, T, 1024)
        out = mix.norm(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


# test_mlp_via_layer_func —— 同时验证 mlp 和派发器
def test_mlp(mix):
    B, T = 2, 281
    x = torch.randn(B, T, 1024)
    out = mix.layer_func("mlp", 0, x)   # ← 经过 layer_func 派发
    assert out.shape == (B, T, 1024)
    assert torch.isfinite(out).all()