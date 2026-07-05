import sys
sys.path.insert(0, '/home/cxy/projects/YuePi0/src')
import yaml
from omegaconf import OmegaConf
from model.paligemma.vit import ViTVisionModel, ImageProjector

with open('/home/cxy/projects/YuePi0/config/yuepi0.yaml') as f:
    cfg = OmegaConf.create(yaml.safe_load(f))
OmegaConf.resolve(cfg)

# 1. SigLIP ViT 参数量
vit = ViTVisionModel(cfg.vision_config)
projector = ImageProjector(cfg.vision_config)

print("=== SigLIP ViT ===")
# 看看 vit 里有什么
for name, module in vit.named_children():
    num = sum(p.numel() for p in module.parameters())
    print(f"  {name}: {num:,}")

# 逐层拆
if hasattr(vit, 'vision_model') or hasattr(vit, 'layers') or hasattr(vit, 'encoder'):
    pass

# 直接看第一层结构
print("\n=== ViT 第一层结构 ===")
for name, child in vit.named_children():
    if 'layer' in name.lower() or 'encoder' in name.lower() or 'block' in name.lower():
        layer0 = child[0] if hasattr(child, '__getitem__') else child
        print(f"  Layer module: {layer0}")
        for n, m in layer0.named_children():
            params = sum(p.numel() for p in m.parameters())
            print(f"    {n}: {params:,}")
        break

vit_total = sum(p.numel() for p in vit.parameters())
proj_total = sum(p.numel() for p in projector.parameters())
print(f"\nViT total: {vit_total:,} ({vit_total/1e6:.0f}M)")
print(f"Projector:  {proj_total:,} ({proj_total/1e6:.0f}M)")

# 逐参数列出
print("\n=== ViT 所有参数项 ===")
for name, p in vit.named_parameters():
    print(f"  {name}: {p.shape} = {p.numel():,}")

# 2. GemmaRMSNorm 参数确认
print("\n=== GemmaRMSNorm 参数 ===")
from model.vla.mixture import Mixture
joint_cfg = cfg.joint
shared = OmegaConf.create({
    'num_heads': joint_cfg.num_heads,
    'num_kv_heads': joint_cfg.num_kv_heads,
    'head_dim': joint_cfg.head_dim,
    'rms_norm_eps': joint_cfg.rms_norm_eps,
    'attention_bias': False,
    'attention_dropout': 0.0,
    'num_hidden_layers': joint_cfg.num_hidden_layers,
    'time_hidden_size': cfg.time_hidden_size,
})

for name in ['vlm', 'action']:
    m_cfg = OmegaConf.merge(shared, OmegaConf.create({
        'hidden_size': cfg.mixture[name].hidden_size,
        'intermediate_size': cfg.mixture[name].intermediate_size,
        'adaptive_mode': None,
        'use_final_norm': cfg.mixture[name].use_final_norm,
        'rope_theta': cfg.mixture[name].rope_theta,
    }))
    m = Mixture(m_cfg)
    layer0 = m.layers[0]
    
    print(f"\n--- {name} expert ---")
    print(f"  use_final_norm: {cfg.mixture[name].use_final_norm}")
    inorm = layer0.input_layernorm
    pnorm = layer0.post_attention_layernorm
    print(f"  input_layernorm.weight shape: {inorm.weight.shape}, numel: {inorm.weight.numel()}")
    print(f"  post_attn_layernorm.weight shape: {pnorm.weight.shape}, numel: {pnorm.weight.numel()}")
    print(f"  每层 norm 参数: {inorm.weight.numel() + pnorm.weight.numel()}")
    
    if hasattr(m, 'norm'):
        print(f"  final norm.weight shape: {m.norm.weight.shape}, numel: {m.norm.weight.numel()}")
    else:
        print(f"  没有 final norm")
