import torch
from omegaconf import OmegaConf
from model.vla.yuepi0 import PiZero
import pytest
def _load_cfg():
    cfg = OmegaConf.load('config/yuepi0.yaml')
    # 缩小 ViT 让 CPU 跑得动                                                                                                                                                                  
    cfg.vision_config.num_hidden_layers   = 3                                                                                                                                                 
    cfg.vision_config.hidden_size         = 128                                                                                                                                               
    cfg.vision_config.intermediate_size   = 256                                                                                                                                               
    cfg.vision_config.num_attention_heads = 4                                                                                                                                                 
    cfg.vision_config.projection_dim      = cfg.hidden_size                                                                                                                                   
    return cfg

def _make_batch(cfg, B=2):
    return {
        "input_ids": torch.randint(0, cfg.vocab_size, (B, cfg.max_image_text_tokens)),
        "pixel_values": torch.randn(B, 3, 224, 224),
        "attention_mask": torch.ones(B, cfg.max_image_text_tokens, dtype=torch.long),
        "proprio":        torch.randn(B, cfg.cond_steps, cfg.proprio_dim),
        "action":         torch.randn(B, cfg.horizon_steps, cfg.action_dim),
    }

def test_pizero_forward_loss_finite():
    cfg   = _load_cfg()
    model = PiZero(cfg)     
    batch = _make_batch(cfg)                                                                                                                                                                  
    loss  = model(batch)
    assert loss.dim() == 0                                                                                                                                                                    
    assert torch.isfinite(loss)

def test_pizero_backward_grads_flow():                                                                                                                                                        
    cfg   = _load_cfg()
    model = PiZero(cfg)                                                                                                                                                                       
    loss  = model(_make_batch(cfg))
    loss.backward()                                                                                                                                                                           
    # 关键链路上都得有梯度
    assert model.action_encoder.linear_1.weight.grad is not None                                                                                                                              
    assert model.action_decoder.proj.weight.grad     is not None                                                                                                                              
    assert torch.isfinite(model.action_decoder.proj.weight.grad).all() 

def test_pizero_infer_action_shape():
    cfg = _load_cfg()                                                                                                                                                                         
    model = PiZero(cfg).eval()                                                                                                                                                                
    B = 2                                                                                                                                                                                     
    batch = _make_batch(cfg, B)   # 跟训练用同一个 fixture,但 model 不会用 batch['action']                                                                                                    
                                                                                                                                                                                            
    actions = model.infer_action(batch, num_inference_steps=3)   # 测试用 3 步省时间                                                                                                          
                                                                                                                                                                                            
    assert actions.shape == (B, cfg.horizon_steps, cfg.action_dim)
    assert torch.isfinite(actions).all()


@pytest.mark.parametrize("mode", [None, "adaLN", "adaLN-Zero"])
def test_pizero_adaptive_modes(mode):
    cfg = _load_cfg()
    cfg.action_expert_adaptive_mode = mode
    model = PiZero(cfg)
    loss = model(_make_batch(cfg))
    assert torch.isfinite(loss)
    # 也测一下推理
    model.eval()
    actions = model.infer_action(_make_batch(cfg, B=1), num_inference_steps=2)
    assert actions.shape == (1, cfg.horizon_steps, cfg.action_dim)
    assert torch.isfinite(actions).all()


def test_infer_action_cache_matches_naive():
    cfg = _load_cfg()
    model = PiZero(cfg).eval()
    batch = _make_batch(cfg, B=1)
    # 固定 noise
    noise = torch.randn(1, cfg.horizon_steps, cfg.action_dim)
    a_cache = model.infer_action(batch, num_inference_steps=3, noise=noise)
    no_cache = model.infer_action_naive(batch, num_inference_steps=3, noise=noise)
    assert torch.allclose(a_cache, no_cache, atol=1e-4)