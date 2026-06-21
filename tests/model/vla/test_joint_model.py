import torch
import pytest
from torch import nn
from model.vla.joint_model import forward_mixture_attn, forward_mixture_layers, JointModel
from model.vla.mixture import Mixture

class FakeConfig:
    hidden_size = 256
    num_hidden_layers = 3
    num_heads = 8
    num_kv_heads = 4
    head_dim = 32
    intermediate_size = 512
    rope_theta = 10000
    rms_norm_eps = 1e-6
    attention_bias = False

@pytest.fixture
def mixtures():
    cfg = FakeConfig()
    return nn.ModuleDict({
        "vlm": Mixture(cfg),
        "proprio": Mixture(cfg),
        "action": Mixture(cfg)
    })

def make_full_causal_mask (B, T, dtype=torch.float32):
    '''最简版：整段当一段普通sequence的causal mask'''
    mask_2d = torch.full((T, T), torch.finfo(dtype).min, dtype=dtype)
    mask_2d = torch.triu(mask_2d, diagonal=1)
    return mask_2d[None, None, :, :].expand(B, 1, T, T).contiguous()

def test_forward_mixture_attn_shape(mixtures):
    B = 2
    T_vlm, T_prop, T_act = 8, 1, 4
    T_total = T_vlm + T_prop + T_act
    H = 256

    embeds = {
        "vlm": torch.randn(B, T_vlm, H),
        "proprio": torch.randn(B, T_prop, H),
        "action": torch.randn(B, T_act, H),
    }
    pos = {
        "vlm": torch.arange(T_vlm)[None].expand(B, -1),
        "proprio": torch.arange(T_prop)[None].expand(B, -1),
        "action": torch.arange(T_act)[None].expand(B, -1),
    }
    mask = make_full_causal_mask(B, T_total)
    out = forward_mixture_attn(mixtures, mask, pos, embeds, layer_idx=0)

    assert out["vlm"].shape == (B, T_vlm, H)
    assert out["proprio"].shape == (B, T_prop, H)
    assert out["action"].shape == (B, T_act, H)
    for name in ["vlm", "proprio", "action"]:
        assert torch.isfinite(out[name]).all(), f"{name} has NaN/Inf"

def test_forward_mixture_layers_shape(mixtures):
    B = 2
    T_vlm, T_prop, T_act = 8, 1, 4
    T_total = T_vlm + T_prop + T_act
    H = 256

    embeds = {
        "vlm": torch.randn(B, T_vlm, H),
        "proprio": torch.randn(B, T_prop, H),
        "action": torch.randn(B, T_act, H),
    }
    pos = {
        "vlm": torch.arange(T_vlm)[None].expand(B, -1),
        "proprio": torch.arange(T_prop)[None].expand(B, -1),
        "action": torch.arange(T_act)[None].expand(B, -1),
    }
    mask = make_full_causal_mask(B, T_total)
    out = forward_mixture_layers(mixtures, mask, pos, embeds, layer_idx=0)
    assert out["vlm"].shape == (B, T_vlm, H)
    assert out["proprio"].shape == (B, T_prop, H)
    assert out["action"].shape == (B, T_act, H)

    for name in ["vlm", "proprio", "action"]:
        assert torch.isfinite(out[name]).all(), f"{name} has NaN/Inf"



class FakeJointConfig:
    num_hidden_layers = 3
    num_heads = 8
    num_kv_heads = 4
    head_dim = 32
    rms_norm_eps = 1e-6
    attention_bias = False
    attention_dropout = 0.0
    mixture = {
        "vlm": FakeConfig(),
        "proprio": FakeConfig(),
        "action": FakeConfig(),
    }

def test_joint_model_forward_shape():
    B = 2
    T_vlm, T_prop, T_act = 8, 1, 4
    T_total = T_vlm + T_prop + T_act
    H = 256

    model = JointModel(FakeJointConfig())
    embeds = {
            "vlm": torch.randn(B, T_vlm, H),
            "proprio": torch.randn(B, T_prop, H),
            "action": torch.randn(B, T_act, H),
        }
    pos = {
        "vlm": torch.arange(T_vlm)[None].expand(B, -1),
        "proprio": torch.arange(T_prop)[None].expand(B, -1),
        "action": torch.arange(T_act)[None].expand(B, -1),
    }
    mask = make_full_causal_mask(B, T_total)
    mask = make_full_causal_mask(B, T_total)

    out = model(mask, pos, embeds)

    assert out["vlm"].shape == (B, T_vlm, H)
    assert out["proprio"].shape == (B, T_prop, H)
    assert out["action"].shape == (B, T_act, H)

    for name in ["vlm", "proprio", "action"]:
        assert torch.isfinite(out[name]).all(), f"{name} has NaN/Inf"

# 测试不同hidden_size
class SmallHiddenConfig:
    hidden_size = 128
    num_hidden_layers = 3
    num_heads = 8
    num_kv_heads = 4
    head_dim = 32
    intermediate_size = 256
    rope_theta = 10000
    rms_norm_eps = 1e-6
    attention_bias = False

class FakeJointConfigMixedHidden:
    num_hidden_layers = 3
    num_heads = 8
    num_kv_heads = 4
    head_dim = 32
    rms_norm_eps = 1e-6
    attention_bias = False
    attention_dropout = 0.0
    mixture = {
        "vlm": FakeConfig(),             # hidden_size = 256
        "proprio": SmallHiddenConfig(),  # hidden_size = 128
        "action": SmallHiddenConfig(),   # hidden_size = 128
    }

def test_joint_model_forward_with_different_hidden_sizes():
    B = 2
    T_vlm, T_prop, T_act = 8, 1, 4
    T_total = T_vlm + T_prop + T_act

    model = JointModel(FakeJointConfigMixedHidden())
    embeds = {
        "vlm": torch.randn(B, T_vlm, 256),
        "proprio": torch.randn(B, T_prop, 128),
        "action": torch.randn(B, T_act, 128),
    }
    pos = {
        "vlm": torch.arange(T_vlm)[None].expand(B, -1),
        "proprio": torch.arange(T_prop)[None].expand(B, -1),
        "action": torch.arange(T_act)[None].expand(B, -1),
    }
    mask = make_full_causal_mask(B, T_total)

    out = model(mask, pos, embeds)

    assert out["vlm"].shape == (B, T_vlm, 256)
    assert out["proprio"].shape == (B, T_prop, 128)
    assert out["action"].shape == (B, T_act, 128)

    for name in ["vlm", "proprio", "action"]:
        assert torch.isfinite(out[name]).all(), f"{name} has NaN/Inf"

