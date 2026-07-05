import sys
sys.path.insert(0, '/home/cxy/projects/YuePi0/src')
import yaml
from omegaconf import OmegaConf
from model.vla.mixture import Mixture

with open('/home/cxy/projects/YuePi0/config/yuepi0.yaml') as f:
    cfg = OmegaConf.create(yaml.safe_load(f))
OmegaConf.resolve(cfg)

joint_cfg = cfg.joint
shared = {
    'num_heads': joint_cfg.num_heads,
    'num_kv_heads': joint_cfg.num_kv_heads,
    'head_dim': joint_cfg.head_dim,
    'rms_norm_eps': joint_cfg.rms_norm_eps,
    'attention_bias': False,
    'attention_dropout': 0.0,
    'num_hidden_layers': joint_cfg.num_hidden_layers,
    'time_hidden_size': cfg.time_hidden_size,
}
shared_cfg = OmegaConf.create(shared)

for name in ['vlm', 'action']:
    m_cfg = OmegaConf.merge(shared_cfg, OmegaConf.create({
        'hidden_size': cfg.mixture[name].hidden_size,
        'intermediate_size': cfg.mixture[name].intermediate_size,
        'adaptive_mode': None,
        'use_final_norm': cfg.mixture[name].use_final_norm,
        'rope_theta': cfg.mixture[name].rope_theta,
    }))
    m = Mixture(m_cfg)
    layer0 = m.layers[0]
    attn = layer0.self_attn
    mlp = layer0.mlp

    h = m_cfg.hidden_size
    inter = m_cfg.intermediate_size
    nq = joint_cfg.num_heads      # 8
    nkv = joint_cfg.num_kv_heads   # 1
    hd = joint_cfg.head_dim        # 256

    q = attn.q_proj.weight.numel()
    k = attn.k_proj.weight.numel()
    v = attn.v_proj.weight.numel()
    o = attn.o_proj.weight.numel()
    attn_total = q + k + v + o

    gate = mlp.gate_proj.weight.numel()
    up = mlp.up_proj.weight.numel()
    down = mlp.down_proj.weight.numel()
    mlp_total = gate + up + down

    norm1 = sum(p.numel() for p in layer0.input_layernorm.parameters())
    norm2 = sum(p.numel() for p in layer0.post_attention_layernorm.parameters())
    norm_total = norm1 + norm2
    per_layer = attn_total + mlp_total + norm_total
    final_norm = sum(p.numel() for p in m.norm.parameters()) if hasattr(m, 'norm') else 0
    expert_total = per_layer * 18 + final_norm

    print(f"=== {name} expert ===")
    print(f"  hidden={h}, intermediate={inter}, heads={nq}, kv_heads={nkv}, head_dim={hd}")
    print(f"  Q proj: Linear({h}, {nq}*{hd}={nq*hd}) = {h}x{nq*hd} = {q:,}")
    print(f"  K proj: Linear({h}, {nkv}*{hd}={nkv*hd}) = {h}x{nkv*hd} = {k:,}")
    print(f"  V proj: Linear({h}, {nkv}*{hd}={nkv*hd}) = {h}x{nkv*hd} = {v:,}")
    print(f"  O proj: Linear({nq*hd}, {h}) = {nq*hd}x{h} = {o:,}")
    print(f"  attn/layer: {attn_total:,} ({attn_total/1e6:.1f}M)")
    print(f"  gate_proj: Linear({h}, {inter}) = {h}x{inter} = {gate:,}")
    print(f"  up_proj:   Linear({h}, {inter}) = {h}x{inter} = {up:,}")
    print(f"  down_proj: Linear({inter}, {h}) = {inter}x{h} = {down:,}")
    print(f"  mlp/layer: {mlp_total:,} ({mlp_total/1e6:.1f}M)")
    print(f"  norm/layer: {norm_total:,}")
    print(f"  per layer total: {per_layer:,} ({per_layer/1e6:.1f}M)")
    print(f"  18 layers: {per_layer*18:,} ({per_layer*18/1e9:.2f}B)")
    if final_norm:
        print(f"  final norm: {final_norm:,}")
    print(f"  EXPERT TOTAL: {expert_total:,} ({expert_total/1e9:.2f}B)")
    print()

# 额外参数
print("=== 额外参数（不属于 expert 层）===")
# embed_tokens
vocab = cfg.vocab_size
hidden_vlm = cfg.mixture.vlm.hidden_size
print(f"  embed_tokens: Embedding({vocab}, {hidden_vlm}) = {vocab*hidden_vlm:,} ({vocab*hidden_vlm/1e9:.2f}B)")
# vision tower
vis_h = cfg.vision_config.hidden_size
vis_inter = cfg.vision_config.intermediate_size
vis_layers = cfg.vision_config.num_hidden_layers
vis_heads = cfg.vision_config.num_attention_heads
vis_head_dim = vis_h // vis_heads
# SigLIP: standard MHA, q=k=v=Linear(h, h*heads*head_dim/h)... actually SigLIP uses standard attention
# q_proj: Linear(1152, 1152), k_proj: Linear(1152, 1152), v_proj: Linear(1152, 1152), o_proj: Linear(1152, 1152)
# mlp: Linear(1152, 4304) x2 (gate+down) or standard FFN
vis_per_layer = 4 * (vis_h * vis_h) + 3 * (vis_h * vis_inter)  # approx: 4 attn matrices + 3 mlp matrices
print(f"  ViT: hidden={vis_h}, inter={vis_inter}, layers={vis_layers}")
print(f"  ViT approx total: {vis_per_layer * vis_layers:,} ({vis_per_layer * vis_layers / 1e6:.0f}M)")
# projector
print(f"  projector: Linear({vis_h}, {hidden_vlm}) = {vis_h*hidden_vlm:,}")
# action encoder + decoder + proprio encoder
act_h = cfg.mixture.action.hidden_size
print(f"  action_encoder: ~3 Linear layers, approx {3*cfg.action_dim*act_h:,}")
print(f"  action_decoder: Linear({act_h}, {cfg.action_dim}) = {act_h*cfg.action_dim:,}")
print(f"  proprio_encoder: Linear({cfg.proprio_dim}, {act_h}) = {cfg.proprio_dim*act_h:,}")
