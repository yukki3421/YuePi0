import torch                                                                                                                                                                                  
from omegaconf import OmegaConf                                                                                                                                                               
from model.vla.yuepi0 import PiZero

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