"""
对比测试：你的 GemmaDecoderLayer vs 原版 open-pi-zero
验证：load 原版 state_dict 到你的实现，相同输入下 allclose
"""
import sys
import torch
import pytest
from types import SimpleNamespace

# 把原版项目加入 PYTHONPATH（让 `from src.model.xxx` 能 import）
ORIG_PATH = "/home/cxy/projects/open-pi-zero"
if ORIG_PATH not in sys.path:
    sys.path.insert(0, ORIG_PATH)

# 原版（重命名避免冲突）
from src.model.paligemma.gemma import GemmaDecoderLayer as OrigDecoderLayer

# 你的实现
from model.paligemma.gemma import GemmaDecoderLayer as MyDecoderLayer


def make_config():
    """构造一个同时兼容原版和你的实现的 config"""
    # SimpleNamespace` 是 Python 标准库提供的"轻量对象"，
    # 可以用 `obj.attr` 访问属性，不需要写一个完整的 class。
    return SimpleNamespace(
        # ===== 共用字段 =====
        hidden_size=64,
        intermediate_size=128,
        head_dim=32,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        # ===== 原版命名 =====
        num_attention_heads=2,
        num_key_value_heads=1,
        attention_dropout=0.0,
        attention_bias=False,
        # ===== 你的命名 =====
        num_heads=2,
        num_kv_heads=1,
    )


def test_decoder_layer_allclose():
    torch.manual_seed(42)
    config = make_config()

    # 1. 实例化原版（eval 模式关掉 dropout）
    orig = OrigDecoderLayer(config, layer_idx=0).eval()

    # 2. 实例化你的实现，load 原版 state_dict
    mine = MyDecoderLayer(config, layer_idx=0).eval()
    # strict=False: 忽略 RoPE buffer 的命名差异（inv_freq vs inv_freqs）
    missing, unexpected = mine.load_state_dict(orig.state_dict(), strict=True)
    print(f"\nmissing keys: {missing}")
    print(f"unexpected keys: {unexpected}")

    # 3. 构造相同输入
    B, T = 1, 4
    x = torch.randn(B, T, config.hidden_size)
    # causal mask: 上三角 -inf
    causal = torch.full((T, T), float("-inf"))
    causal = torch.triu(causal, diagonal=1)
    attention_mask = causal[None, None, :, :]  # (1, 1, T, T)
    position_ids = torch.arange(T).unsqueeze(0)  # (1, T)

    # 4. 两边 forward
    with torch.no_grad():
        y_orig = orig(
            hidden_states=x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=None,
        )
        # 原版 forward 返回的是 hidden_states 单值（看 line 196）
        if isinstance(y_orig, tuple):
            y_orig = y_orig[0]

        y_mine = mine(
            hidden_state=x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kvcache=None,
        )

    # 5. 对比
    max_diff = (y_orig - y_mine).abs().max().item()
    print(f"max diff: {max_diff:.2e}")

    assert torch.allclose(y_orig, y_mine, atol=1e-5), (
        f"❌ allclose 失败，max_diff={max_diff:.2e}"
    )
    print(f"✅ allclose 通过 (atol=1e-5)")
