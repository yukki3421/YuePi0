import torch
from model.paligemma.vit import *
import pytest

# ---------------- Fixtures ----------------
# fixture = pytest 的"共享对象工厂"。被打了 @pytest.fixture 的函数，
# 任何测试函数只要把它的名字写进参数列表，pytest 就会自动调用一次它，把返回值喂进去。
# 这样多个测试可以共用一个 cfg / 一个假图，不用每个测试都自己手搓。

@pytest.fixture
def cfg():
    return ViTVisionConfig(num_hidden_layers=5)

@ pytest.fixture
def cfg27():
    return ViTVisionConfig(num_hidden_layers=27)

@pytest.fixture
def fake_image(cfg):
    torch.manual_seed(0)
    return torch.randn(2, 3, cfg.image_size, cfg.image_size)

# -------------------形状测试-----------------------
def test_embedding_shape(cfg, fake_image):
    
    emb = ViTVisionEmbedding(cfg)
    out = emb(fake_image)
    expected_num_patches = (cfg.image_size // cfg.patch_size) ** 2
    assert out.shape == (2, expected_num_patches, cfg.hidden_size)

def test_attention_shape(cfg, fake_image):
    emb = ViTVisionEmbedding(cfg)
    attn = ViTVisionAttention(cfg)
    x = emb(fake_image)
    out = attn(x)
    assert out[0].shape == x.shape  # attention 不改 shape


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
    assert enc(x).shape == x.shape

def test_vision_transformer_full_config(cfg27, fake_image):
      model = ViTVisionTransformer(cfg27).eval() 
      with torch.no_grad():                                                          
          out = model(fake_image)
      assert out.shape == (2, 256, 1152)                                             
      assert torch.isfinite(out).all()   

def test_attention_softmax_rows_sum_to_one(cfg, fake_image):                       
    attn = ViTVisionAttention(cfg).eval()                                          
    x = ViTVisionEmbedding(cfg)(fake_image)
    with torch.no_grad():                                                          
        _, w = attn(x)   # [B, H, T, T]
    sums = w.sum(dim=-1)  # 每行和                                                 
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5) 

def test_projector_shape(cfg):
    proj = ImageProjector(cfg)
    x = torch.randn(2, 256, cfg.hidden_size)
    assert proj(x).shape == (2, 256, 2048)



