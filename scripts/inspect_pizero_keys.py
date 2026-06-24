from pathlib import Path                                                                                                                             
import sys                                                                                                                                           
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))                                                                                 
                                                                                                                                                    
import torch    
from omegaconf import OmegaConf                                                                                                                      
from model.vla.yuepi0 import PiZero                                                                                                                  
                                                                                                                                                    
# 用全尺寸 config (不是 fakedataTrain), 才能跟 HF 权重对得上                                                                                         
config = OmegaConf.load("config/yuepi0.yaml")                                                                                                        
OmegaConf.resolve(config)                                                                                                                            
                                                                                                                                                    
# meta device 上建模型, 不占显存 (我们只关心 key 结构)                                                                                               
with torch.device("meta"):                                                                                                                           
    model = PiZero(config)                                                                                                                           
                
state_dict = model.state_dict()                                                                                                                      
print(f"Total keys: {len(state_dict)}\n")
                                                                                                                                                    
# 按顶层前缀分组                                                                                                                                     
prefixes = {}
for k in state_dict:                                                                                                                                 
    top = k.split('.')[0]
    prefixes.setdefault(top, []).append(k)                                                                                                           
                                                                                                                                                    
for top, keys in prefixes.items():                                                                                                                   
    print(f"=== {top} ({len(keys)} keys) ===")                                                                                                       
    for k in keys[:8]:                                                                                                                               
        print(f"  {k:80s} {tuple(state_dict[k].shape)}")
    if len(keys) > 8:                                                                                                                                
        print(f"  ... 还有 {len(keys)-8} 个")
    print()                                            