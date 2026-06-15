import torch
import pytest
from model.paligemma.gemma import GemmaForCausalLM

class FakeConfig:
    hidden_size = 2048
    num_hidden_layers = 5
    vocab_size = 257216
    intermediate_size = 4034
    num_heads = 16
    head_dim = 128
    num_kv_heads = 4
    rope_theta = 1000
    rms_norm_eps = 1e-6
    pad_token_id = 0

'''
@pytest.fixture 是 pytest 的依赖注入机制。
标了 @pytest.fixture 的函数，会自动提前运行，产出数据 / 对象
'''
@pytest.fixture
def model():
    return GemmaForCausalLM(FakeConfig())

@pytest.fixture
def inputs(model):
    B, T = 2, 16
    x = torch.randint(0, 257216, (B, T))
    attention_mask = torch.triu(torch.ones(T, T), diagonal=1)*(-1e9)
    position_ids = torch.arange(T).unsqueeze(0).expand(B, T)
    inputs_embedding = model.model.embed_tokens(x)
    return {
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "inputs_embedding": inputs_embedding
    }

def test_logits_shape(model, inputs):
    out = model(**inputs)
    assert out['logits'].shape == (2, 16, 257216)

def test_isfinite(model, inputs):
    out = model(**inputs)
    assert torch.isfinite(out["logits"]).all()

def test_gradient_flow(model, inputs):                                                       
    out = model(**inputs)                                 
    loss = out["logits"].sum()
    loss.backward()

    # 验证至少有一些参数有梯度
    has_grad = [p.grad is not None for p in model.parameters() if p.requires_grad]
    assert any(has_grad), "No parameter has gradient"