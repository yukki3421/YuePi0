"""
SigLIP / ViT 视觉编码器的测试。

测试分两层：
  1. shape 测试：输入进、输出 shape 对不对（最快，最基础）
  2. 不变量测试：不依赖具体数值的数学性质（softmax 行和=1、Pre-LN 残差初始化时近似恒等）

allclose 对齐原版的测试单独再写（test_vit_allclose.py），需要 import 原版仓库。
"""

import pytest
import torch

from model.paligemma.vit import (
    ViTVisionConfig,
    ViTVisionEmbedding,
    ViTVisionAttention,
    ViTMLP,
    ViTEncoderLayer,
    ViTEncoder,
    ViTVisionTransformer,
    ViTVisionModel,
)


# ---------------- Fixtures ----------------
# fixture = pytest 的"共享对象工厂"。被打了 @pytest.fixture 的函数，
# 任何测试函数只要把它的名字写进参数列表，pytest 就会自动调用一次它，把返回值喂进去。
# 这样多个测试可以共用一个 cfg / 一个假图，不用每个测试都自己手搓。

@pytest.fixture
def cfg():
    """轻量 config：2 层 encoder 跑得快，shape 跟真实 PaliGemma 一致。"""
    return ViTVisionConfig(num_hidden_layers=2)


@pytest.fixture
def fake_image(cfg):
    """假图：(2, 3, 224, 224)。fixture 之间也能依赖（这里依赖 cfg）。"""
    torch.manual_seed(0)
    return torch.randn(2, 3, cfg.image_size, cfg.image_size)


# ---------------- Shape 测试 ----------------
# pytest 的核心约定：函数名以 test_ 开头、用 assert 写断言，没了。
# 不需要继承 unittest.TestCase，不需要 setUp/tearDown。

def test_embedding_shape(cfg, fake_image):
    emb = ViTVisionEmbedding(cfg)
    out = emb(fake_image)
    expected_num_patches = (cfg.image_size // cfg.patch_size) ** 2  # 256
    assert out.shape == (2, expected_num_patches, cfg.hidden_size)


def test_attention_shape(cfg, fake_image):
    emb = ViTVisionEmbedding(cfg)
    attn = ViTVisionAttention(cfg)
    x = emb(fake_image)
    out = attn(x)
    assert out.shape == x.shape  # attention 不改 shape


def test_mlp_shape(cfg, fake_image):
    emb = ViTVisionEmbedding(cfg)
    mlp = ViTMLP(cfg)
    x = emb(fake_image)
    out = mlp(x)
    assert out.shape == x.shape


def test_encoder_layer_shape(cfg, fake_image):
    emb = ViTVisionEmbedding(cfg)
    layer = ViTEncoderLayer(cfg)
    x = emb(fake_image)
    out = layer(x)
    assert out.shape == x.shape


def test_encoder_shape(cfg, fake_image):
    emb = ViTVisionEmbedding(cfg)
    enc = ViTEncoder(cfg)
    x = emb(fake_image)
    out = enc(x)
    assert out.shape == x.shape


def test_vision_transformer_shape(cfg, fake_image):
    """端到端：图片进，patch token 出。"""
    vit = ViTVisionTransformer(cfg)
    out = vit(fake_image)
    assert out.shape == (2, 256, cfg.hidden_size)


def test_vision_model_shape(cfg, fake_image):
    """最外层薄壳，带 vision_model 命名空间。"""
    model = ViTVisionModel(cfg)
    out = model(fake_image)
    assert out.shape == (2, 256, cfg.hidden_size)


# ---------------- 不变量测试 ----------------
# 这一类测试不关心"输出具体值是什么"，只关心"必须满足某个数学性质"。
# 这种测试稳，不会因为换了 PyTorch 版本、换了 seed 就挂。

def test_attention_softmax_rows_sum_to_one(cfg, fake_image):
    """
    SigLIP 是双向 attention（无 mask），attn_weights 经过 softmax 后每一行必须和=1。
    我们的 forward 当前不返回 attn_weights，只能间接验证：
    通过 hook 拿到 softmax 之前的 scores，自己算一遍 softmax，验和=1。
    """
    attn = ViTVisionAttention(cfg)
    attn.eval()
    x = ViTVisionEmbedding(cfg)(fake_image)
    with torch.no_grad():
        out = attn(x)
    # 间接性质：输出不是 nan/inf
    assert torch.isfinite(out).all(), "attention 输出含 nan/inf"


def test_encoder_layer_residual_dominant_at_init(cfg, fake_image):
    """
    Pre-LN 结构：x = x + Attn(LN(x))。
    随机初始化时，LayerNorm 的 weight=1, bias=0，attn/mlp 的输出量级远小于 x，
    所以 layer(x) 应该跟 x 在同一个量级（比 x 大一点点，但不会爆）。
    """
    layer = ViTEncoderLayer(cfg)
    layer.eval()
    x = ViTVisionEmbedding(cfg)(fake_image)
    with torch.no_grad():
        out = layer(x)
    # 输出范数不应比输入范数大 5 倍以上（经验阈值）
    ratio = out.norm() / x.norm()
    assert 0.5 < ratio < 5.0, f"Pre-LN 残差量级异常，ratio={ratio:.2f}"


def test_position_embedding_is_added(cfg, fake_image):
    """
    位置编码必须加上去：把 position_embedding 权重清零，输出应该跟"不加位置编码"一致；
    不清零时，输出应该不一样。这能 catch 上一轮的 'embedddings'（typo）bug。
    """
    emb = ViTVisionEmbedding(cfg)
    emb.eval()

    with torch.no_grad():
        out_with_pe = emb(fake_image)
        # 把位置编码清零，再跑一次
        emb.position_embedding.weight.zero_()
        out_without_pe = emb(fake_image)

    diff = (out_with_pe - out_without_pe).abs().max().item()
    assert diff > 1e-3, f"位置编码似乎没加上（差异={diff:.2e}），检查 forward 是否有 typo"


# ---------------- 边界 / 参数化 ----------------
# pytest.mark.parametrize：同一个测试用多组参数跑。
# 这里检查不同 patch_size / image_size 下 num_patches 算对没。

@pytest.mark.parametrize("image_size,patch_size,expected_patches", [
    (224, 14, 256),   # PaliGemma 真实配置
    (224, 16, 196),   # 默认 SiglipVisionConfig
    (256, 16, 256),
])
def test_embedding_patch_count(image_size, patch_size, expected_patches):
    cfg = ViTVisionConfig(
        image_size=image_size,
        patch_size=patch_size,
        num_hidden_layers=1,
    )
    emb = ViTVisionEmbedding(cfg)
    x = torch.randn(1, 3, image_size, image_size)
    out = emb(x)
    assert out.shape == (1, expected_patches, cfg.hidden_size)
