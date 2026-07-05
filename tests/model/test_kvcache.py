import torch
import pytest
from model.paligemma.gemma import GemmaAttention
from model.kvcache import KVCache

def test_kv_cache():
    torch.manual_seed(0)
    B, T, D = 2, 8, 512
    num_heads = 32
    num_kv_heads = 8
    head_dim = 16
    GemmaRoPE_theta = 10000

    gqa = GemmaAttention(D, num_heads, num_kv_heads, head_dim, GemmaRoPE_theta, 0)
    gqa.eval()

    x_full = torch.randn(B, T, D)
    pos_full = torch.arange(T).unsqueeze(0).expand(B, T)
    mask_full = torch.triu(torch.ones(T, T), diagonal=1) * (-1e9)

    # 方式一：一次性forward
    with torch.no_grad():
        y_full, _ = gqa(x_full, mask_full, pos_full)

    # 方式二：分两步用kv cache
    T1 = 5
    cache = KVCache()

    x1 = x_full[:, :T1]
    pos1 = pos_full[:, :T1]
    mask1 = mask_full[:T1, :T1]

    x2 = x_full[:, T1:]
    pos2 = pos_full[:, T1:]
    mask2 = mask_full[T1:, :] #注意这个mask！！应该为（3， 8）而不是(3, 3)

    with torch.no_grad():
        y1, _ = gqa(x1, mask1, pos1, cache)
        y2, _ = gqa(x2, mask2, pos2, cache)

    y_cached = torch.cat([y1, y2], dim=1)
    diff = (y_full - y_cached).abs().max().item()
    assert torch.allclose(y_full, y_cached, atol=1e-5), \
    f"KV Cache 输出不一致, max_diff={diff:.2e}"

from torch import nn
from model.vla.joint_model import JointModel, forward_mixture_attn
from model.vla.mixture import Mixture
# ========== 共享配置 ==========
class FakeConfig:
    hidden_size = 256
    num_hidden_layers = 2
    num_heads = 8
    num_kv_heads = 4
    head_dim = 32
    intermediate_size = 512
    rope_theta = 10000
    rms_norm_eps = 1e-6
    attention_bias = False
    attention_dropout = 0.0
    use_final_norm = True
    adaptive_mode = None
    time_hidden_size = 256
    cache = True


@pytest.fixture
def mixtures():
    cfg = FakeConfig()
    return nn.ModuleDict({
        "vlm": Mixture(cfg),
        "proprio": Mixture(cfg),
        "action": Mixture(cfg),
    })


@pytest.fixture
def dummy_inputs():
    """返回 vlm_emb, prop_emb, act_emb, pos, mask 等"""
    B = 1
    T_vlm, T_prop, T_act = 4, 1, 2
    T_total = T_vlm + T_prop + T_act
    H = 256
    return {
        "B": B,
        "T_vlm": T_vlm, "T_prop": T_prop, "T_act": T_act,
        "T_total": T_total, "H": H,
        "vlm_emb": torch.randn(B, T_vlm, H),
        "prop_emb": torch.randn(B, T_prop, H),
        "act_emb": torch.randn(B, T_act, H),
        "pos": {
            "vlm":     torch.arange(T_vlm)[None].expand(B, -1),
            "proprio": torch.arange(T_prop)[None].expand(B, -1),
            "action":  torch.arange(T_act)[None].expand(B, -1),
        },
        "mask_full": torch.zeros(B, 1, T_total, T_total),
    }

# TEST 1: no_append 模式不写 cache: 默认模式（训练、朴素推理）。
def test_no_append_does_not_write_cache(mixtures, dummy_inputs):
    """cache_mode='no_append' + kv_caches=None: 不读不写 cache, 和旧代码等价"""
    embeds_all = {
        "vlm": dummy_inputs["vlm_emb"],
        "proprio": dummy_inputs["prop_emb"],
        "action": dummy_inputs["act_emb"],
    }
    pos = dummy_inputs["pos"]
    mask = dummy_inputs["mask_full"]

    with torch.no_grad():
        out = forward_mixture_attn(
            mixtures, mask, pos, embeds_all, layer_idx=0,
            kv_caches=None, cache_mode="no_append",
        )

    # 1) 三个 mixture 都有输出
    assert set(out.keys()) == {"vlm", "proprio", "action"}
    # 2) 形状对
    assert out["vlm"].shape == (1, 4, 256)
    assert out["action"].shape == (1, 2, 256)
    # 3) 没有 NaN/Inf
    for name in out:
        assert torch.isfinite(out[name]).all(), f"{name} has NaN/Inf"


# TEST 2: append_non_active prefill 把 vlm/proprio 的 K/V 写进 cache

def test_prefill_writes_cache(mixtures, dummy_inputs):
    """prefill 阶段: active={vlm, proprio}, cache_mode='append_non_active'
    vlm/proprio 的 K/V 应该被写进 cache"""
    kv_caches = {"vlm": KVCache(), "proprio": KVCache()}
    embeds_prefill = {
        "vlm": dummy_inputs["vlm_emb"],
        "proprio": dummy_inputs["prop_emb"],
    }
    pos_prefill = {"vlm": dummy_inputs["pos"]["vlm"],
                    "proprio": dummy_inputs["pos"]["proprio"]}
    mask_prefill = torch.zeros(1, 1, 4 + 1, 4 + 1)  # T_vlm + T_prop

    with torch.no_grad():
        out = forward_mixture_attn(
            mixtures, mask_prefill, pos_prefill, embeds_prefill,
            layer_idx=0, kv_caches=kv_caches, cache_mode="append_non_active",
        )

    # 1) cache[vlm] 第 0 层有 K/V, 形状 (B, num_kv_heads, T_vlm, head_dim)
    assert kv_caches["vlm"].has_item(0)
    assert kv_caches["vlm"].key_cache[0].shape == (1, 4, 4, 32)
    assert kv_caches["vlm"].value_cache[0].shape == (1, 4, 4, 32)
    # 2) cache[proprio] 也有
    assert kv_caches["proprio"].has_item(0)
    assert kv_caches["proprio"].key_cache[0].shape == (1, 4, 1, 32)
    # 3) 输出只有 active (vlm, proprio), 没有 action
    assert set(out.keys()) == {"vlm", "proprio"}


#   TEST 3: 去噪阶段只读 cache, 不写

def test_denoise_reads_cache_does_not_write(mixtures, dummy_inputs):
    """去噪阶段: active={action}, vlm/proprio 从 cache 读, action 不写 cache"""
    # 先 prefill 填 cache
    kv_caches = {"vlm": KVCache(), "proprio": KVCache()}
    embeds_prefill = {
        "vlm": dummy_inputs["vlm_emb"],
        "proprio": dummy_inputs["prop_emb"],
    }
    pos_prefill = {"vlm": dummy_inputs["pos"]["vlm"],
                    "proprio": dummy_inputs["pos"]["proprio"]}
    mask_prefill = torch.zeros(1, 1, 4 + 1, 4 + 1)
    with torch.no_grad():
        forward_mixture_attn(
            mixtures, mask_prefill, pos_prefill, embeds_prefill,
            layer_idx=0, kv_caches=kv_caches, cache_mode="append_non_active",
        )
    # 记录 prefill 后 cache 的内容
    vlm_k_before = kv_caches["vlm"].key_cache[0].clone()

    # 去噪: 只有 action
    embeds_denoise = {"action": dummy_inputs["act_emb"]}
    pos_denoise = {"action": dummy_inputs["pos"]["action"]}
    mask_denoise = torch.zeros(1, 1, 2, 7)  # (B, 1, T_act, T_total)

    with torch.no_grad():
        out = forward_mixture_attn(
            mixtures, mask_denoise, pos_denoise, embeds_denoise,
            layer_idx=0, kv_caches=kv_caches, cache_mode="append_non_active",
        )

    # 1) 输出只有 action
    assert set(out.keys()) == {"action"}
    assert out["action"].shape == (1, 2, 256)
    assert torch.isfinite(out["action"]).all()
    # 2) action 不在 kv_caches 字典里 → 不应该有 "action" 这个 key
    assert "action" not in kv_caches
    # 3) vlm cache 内容不变 (去噪没改 cache)
    assert torch.equal(vlm_k_before, kv_caches["vlm"].key_cache[0])


# TEST 4: cache 版 vs 朴素版数值等价（最重要）

def test_cache_mode_matches_naive(mixtures, dummy_inputs):
    """同一份输入, 朴素 (全 active, no_append) vs cache (prefill + 去噪)
    结果应该几乎一样 (有 RoPE 顺序的小差异, atol 放宽到 1e-4)"""
    # 跑朴素版: 三个 mixture 全 active, no_append
    embeds_naive = {
        "vlm": dummy_inputs["vlm_emb"],
        "proprio": dummy_inputs["prop_emb"],
        "action": dummy_inputs["act_emb"],
    }
    pos_naive = dummy_inputs["pos"]
    mask_naive = dummy_inputs["mask_full"]
    with torch.no_grad():
        out_naive = forward_mixture_attn(
            mixtures, mask_naive, pos_naive, embeds_naive,
            layer_idx=0, kv_caches=None, cache_mode="no_append",
        )

    # 跑 cache 版: 先 prefill vlm+proprio, 再 denoise action
    kv_caches = {"vlm": KVCache(), "proprio": KVCache()}
    mask_prefill = torch.zeros(1, 1, 4 + 1, 4 + 1)
    pos_prefill = {"vlm": dummy_inputs["pos"]["vlm"],
                    "proprio": dummy_inputs["pos"]["proprio"]}
    with torch.no_grad():
        forward_mixture_attn(
            mixtures, mask_prefill, pos_prefill,
            {"vlm": dummy_inputs["vlm_emb"], "proprio": dummy_inputs["prop_emb"]},
            layer_idx=0, kv_caches=kv_caches, cache_mode="append_non_active",
        )
    mask_denoise = torch.zeros(1, 1, 2, 7)
    with torch.no_grad():
        out_cache = forward_mixture_attn(
            mixtures, mask_denoise, {"action": dummy_inputs["pos"]["action"]},
            {"action": dummy_inputs["act_emb"]},
            layer_idx=0, kv_caches=kv_caches, cache_mode="append_non_active",
        )

    # 对比 action 输出
    diff = (out_naive["action"] - out_cache["action"]).abs().max()
    assert diff < 1e-4, f"action diff too large: {diff}"

#   测什么：cache 版和朴素版输出数值等价——证明 cache 没改变计算结果, 只是省了重算。

#   为什么可能有小差异：repeat_kv 在 active 和非 active 分别做时, 浮点累加顺序可能略有不同, 所以 atol=1e-4 而不是 1e-6。

#   ---
#   TEST 5: 多层 cache 一致性（测 layer_idx 不同的层都能正确填）

def test_multi_layer_cache(mixtures, dummy_inputs):
    """跑 2 层, 验证每一层各自填各自的 cache (list 长度 = 2)"""
    kv_caches = {"vlm": KVCache(), "proprio": KVCache()}
    embeds_prefill = {
        "vlm": dummy_inputs["vlm_emb"],
        "proprio": dummy_inputs["prop_emb"],
    }
    pos_prefill = {"vlm": dummy_inputs["pos"]["vlm"],
                    "proprio": dummy_inputs["pos"]["proprio"]}
    mask_prefill = torch.zeros(1, 1, 4 + 1, 4 + 1)

    with torch.no_grad():
        # 第 0 层
        forward_mixture_attn(mixtures, mask_prefill, pos_prefill, embeds_prefill,
                            layer_idx=0, kv_caches=kv_caches, cache_mode="append_non_active")
        # 第 1 层
        forward_mixture_attn(mixtures, mask_prefill, pos_prefill, embeds_prefill,
                            layer_idx=1, kv_caches=kv_caches, cache_mode="append_non_active")

    # cache 应该有 2 层
    assert len(kv_caches["vlm"].key_cache) == 2
    assert len(kv_caches["vlm"].value_cache) == 2
    # 每层形状对
    for layer_idx in range(2):
        assert kv_caches["vlm"].key_cache[layer_idx].shape == (1, 4, 4, 32)

#   测什么：18 层真实场景的缩影——每一层 cache 各自独立, has_item(layer_idx) 按层判断。


# TEST 6: JointModel.forward 整体 cache 流程
class FakeConfigNoCache(FakeConfig):
    cache = False
    hidden_size = 256
    num_hidden_layers = 2
    num_heads = 8
    num_kv_heads = 4
    head_dim = 32
    intermediate_size = 512
    rope_theta = 10000
    rms_norm_eps = 1e-6
    attention_bias = False
    attention_dropout = 0.0
    use_final_norm = True
    adaptive_mode = None
    time_hidden_size = 256
    cache = False

class FakeJointConfig:
    num_hidden_layers = 2
    num_heads = 8
    num_kv_heads = 4
    head_dim = 32
    rms_norm_eps = 1e-6
    attention_bias = False
    attention_dropout = 0.0
    time_hidden_size = 256
    mixture = {
        "vlm":     FakeConfig(),
        "proprio": FakeConfig(),
        "action": FakeConfigNoCache(),
    }

def test_joint_model_full_cache_flow():
    """端到端: JointModel 跑 prefill + 去噪, 返回 cache, 验证整体能跑通"""
    model = JointModel(FakeJointConfig())
    model.eval()

    B = 1
    T_vlm, T_prop, T_act = 4, 1, 2
    T_total = T_vlm + T_prop + T_act
    H = 256
    vlm_emb = torch.randn(B, T_vlm, H)
    prop_emb = torch.randn(B, T_prop, H)
    act_emb = torch.randn(B, T_act, H)
    pos = {
        "vlm":     torch.arange(T_vlm)[None].expand(B, -1),
        "proprio": torch.arange(T_prop)[None].expand(B, -1),
        "action":  torch.arange(T_act)[None].expand(B, -1),
    }
    mask_prefill = torch.zeros(B, 1, T_vlm + T_prop, T_vlm + T_prop)
    mask_denoise = torch.zeros(B, 1, T_act, T_total)

    # 1) prefill
    kv_caches = model.build_mixture_caches()
    with torch.no_grad():
        out_prefill = model(
            mask_prefill,
            {"vlm": pos["vlm"], "proprio": pos["proprio"]},
            {"vlm": vlm_emb, "proprio": prop_emb},
            kv_caches=kv_caches, cache_mode="append_non_active",
            return_caches=True,
        )
    # return_caches=True 时返回 (hidden_states, kv_caches)
    assert isinstance(out_prefill, tuple)
    hidden_prefill, kv_caches = out_prefill
    # cache 填好了
    assert kv_caches["vlm"].has_item(model.num_hidden_layers - 1)  # 最后一层也填了

    # 2) denoise
    with torch.no_grad():
        out_denoise = model(
            mask_denoise,
            {"action": pos["action"]},
            {"action": act_emb},
            kv_caches=kv_caches, cache_mode="append_non_active",
        )
    assert "action" in out_denoise
    assert out_denoise["action"].shape == (B, T_act, H)
    assert torch.isfinite(out_denoise["action"]).all()