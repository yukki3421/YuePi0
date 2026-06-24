from pathlib import Path
from safetensors import safe_open

HF_PATH = Path.home() / ".cache/huggingface/hub/paligemma-3b-pt-224"

# 1) 收集所有key + shape
all_keys = {}
for shared in sorted(HF_PATH.glob("*.safetensors")):
    with safe_open(shared, framework="pt") as f:
        for k in f.keys():
            all_keys[k] = tuple(f.get_tensor(k).shape)

print(f" Total keys: {len(all_keys)} \n")

# 2）按前缀分组                                                                                                                                   
prefixes = {}                                                                                                                                      
for k in all_keys:                                                                                                                                   
    top = k.split('.')[0]
    prefixes.setdefault(top, []).append(k)         

# 3) 每组打印前几个 key, 看结构                                                                                                                      
for top, keys in prefixes.items():                                                                                                                   
    print(f"=== {top} ({len(keys)} keys) ===")                                                                                                       
    for k in keys[:5]:                                                                                                                               
        print(f"  {k:80s} {all_keys[k]}")
    if len(keys) > 5:                                                                                                                                
        print(f"  ... 还有 {len(keys)-5} 个")                                                                                                        
    print()                                                                                             
                                              

