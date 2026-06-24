"""把 HF PaliGemma 权重映射加载到 YuePi0 PiZero。"""
from pathlib import Path                                                                                                                             
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))                                                                                 
                                                                                                                                                    
import torch
from omegaconf import OmegaConf                                                                                                                      
from safetensors import safe_open
from model.vla.yuepi0 import PiZero                                                                                                                  

                                                                                                                                                    
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
                                                                                                                                                    
                                                                                                                                                    
def load_paligemma_weights(model: PiZero, hf_path: Path):                                                                                            
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

                                                                                                                                                    
if __name__ == "__main__":
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