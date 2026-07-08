import torch
# 用于实现配对旋转, 对半分的配对旋转方法
def rotate_half(x):
    dim = x.shape[-1]
    x_2 = x[...,  dim//2:] # python的省略号切片, 就是前面所有维度全取
    x_1 = x[..., :dim//2]
    return torch.cat((-x_2, x_1), dim=-1)

# 输出旋转后的q, k
# 输入q, k 的形状: (B， H, T, D)
# cos，sin的形状：（B, T， D)
def apply_rotary_pos_emb(x, cos, sin):
    # 增加一个H维度
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    # 旋转公式
    x_rot = x*cos + rotate_half(x)*sin
    return x_rot
'''将(B, H_kv, T, D_h) 扩展成(B, H_Q, T, D_h), 其中H_Q = H_KV x G'''
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # expand只能扩展大小为1的维度，对于其他维度大小保持不变
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(
        batch,
        num_key_value_heads * n_rep,
        slen,
        head_dim,
    )

"""把 HF PaliGemma checkpoint 的 state_dict 映射到 YuePi0 PiZero 上。"""                                                                             
from pathlib import Path                                                                                                                             
from safetensors import safe_open
                                                                                                                                                    
                                                                                                                                                    
def hf_key_to_yuepi0_key(hf_key: str) -> str | None:
    """把 HF 的一个 key 转成 YuePi0 的 key。None 表示丢弃。"""                                                                                       
    # 规则 1: embed_tokens                                                                                                                           
    if hf_key == "language_model.model.embed_tokens.weight":                                                                                         
        return "embedder.embed_tokens.weight"                                                                                                        
                                                                                                                                                    
    # 规则 2: language_model.model.layers.[i].* → joint.mixtures.vlm.layers.[i].*                                                                    
    if hf_key.startswith("language_model.model.layers."):                                                                                            
        return hf_key.replace("language_model.model", "joint.mixtures.vlm")                                                                          
                                                                                                                                                    
    # 规则 3: 丢弃最后那个 norm
    if hf_key == "language_model.model.norm.weight":                                                                                                 
        return None

    # 规则 4: multi_modal_projector 前缀替换
    if hf_key.startswith("multi_modal_projector."):
        return "embedder." + hf_key                                                                                                                  

    # 规则 5: vision_tower 前缀替换                                                                                                                  
    if hf_key.startswith("vision_tower."):
        return "embedder." + hf_key                                                                                                                  
                
    # 不该出现的 key                                                                                                                                 
    raise ValueError(f"Unmapped HF key: {hf_key}")
                                                                                                                                                    
                                                                                                                                                    
def load_paligemma_weights(model, hf_path: Path):                                                                                                    
    """读 HF safetensors, 按映射加载到 model 上, 返回加载/跳过统计。"""                                                                              
    hf_state = {}                                                                                                                                    
    for shard in sorted(hf_path.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:                                                                                                  
            for k in f.keys():                                                                                                                       
                hf_state[k] = f.get_tensor(k)                                                                                                        
                                                                                                                                                    
    own_state = model.state_dict()
    loaded, skipped, shape_mismatch = [], [], []                                                                                                     
                                                                                                                                                    
    for hf_k, tensor in hf_state.items():                                                                                                            
        yp_k = hf_key_to_yuepi0_key(hf_k)                                                                                                            
        if yp_k is None:                                                                                                                             
            skipped.append(hf_k)                                                                                                                     
            continue
        if yp_k not in own_state:                                                                                                                    
            raise KeyError(f"Mapped key not found in PiZero: {yp_k}")                                                                                
        if own_state[yp_k].shape != tensor.shape:                                                                                                    
            shape_mismatch.append((yp_k, own_state[yp_k].shape, tensor.shape))                                                                       
            continue                                                                                                                                 
        own_state[yp_k].copy_(tensor)                                                                                                                
        loaded.append(yp_k)                                                                                                                          
                                                                                                                                                    
    return {"loaded": loaded, "skipped": skipped, "shape_mismatch": shape_mismatch}

def to_device_bf16(inputs: dict, device) -> dict:
    """把 dict 里所有 tensor 搬到 device, 浮点的额外转 bf16, 整数/布尔保持原 dtype。"""
    out = {}
    for k, v in inputs.items():
        v = v.to(device)
        if v.is_floating_point():
            v = v.to(torch.bfloat16)
        out[k] = v
    return out


if __name__ == "__main__":
    """把 HF PaliGemma 权重映射加载到 YuePi0 PiZero。"""
    from pathlib import Path                                                                                                                             
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))                                                                                 
                                                                                                                                                        
    import torch
    from omegaconf import OmegaConf                                                                                                                      
    from safetensors import safe_open
    from model.vla.yuepi0 import PiZero    

    HF_PATH = Path.home() / ".cache/huggingface/hub/paligemma-3b-pt-224"
                                                                                                                                                    
    config = OmegaConf.load("config/yuepi0.yaml")
    OmegaConf.resolve(config)                                                                                                                        
    model = PiZero(config)   # 注意:不要 meta 设备,我们要真填权重                                                                                    
                                                                                                                                                    
    stats = load_paligemma_weights(model, HF_PATH)                                                                                                   
    print(f"loaded:  {len(stats['loaded'])}")                                                                                                        
    print(f"skipped: {len(stats['skipped'])}  {stats['skipped']}")                                                                                   
    print(f"shape mismatches: {len(stats['shape_mismatch'])}")                                                                                       
    for k, s1, s2 in stats['shape_mismatch']:                                                                                                        
        print(f"  {k}: model={s1} hf={s2}")                                                                                                          
                                                                                                                                                    
    # 验证: PiZero 里**没**被加载的 key 有哪些 (应该是 proprio/action expert + encoders)                                                             
    loaded_set = set(stats['loaded'])                                                                                                                
    unloaded = [k for k in model.state_dict() if k not in loaded_set]                                                                                
    print(f"\nunloaded (随机初始化): {len(unloaded)} 个")
    for k in unloaded[:5]:                                                                                                                           
        print(f"  {k}")
    if len(unloaded) > 5:
        print(f"  ... 还有 {len(unloaded)-5} 个")