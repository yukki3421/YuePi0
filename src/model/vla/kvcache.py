import torch
from typing import List, Tuple  

class KVCache:
    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    
    # 返回已经保存了多少个token
    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # (B, h, T, D)中返回T这个维度 
            return self.key_cache[0].shape[-2]
    
    # 判断一个layer层是否 已经cache了
    '''
    len(key_cache) = 2  ← 已存了 layer 0、layer 1                                               
                                                                                              
    has_item(0) → True   (因为 2 > 0)                                                           
    has_item(1) → True   (因为 2 > 1)                                                           
    has_item(2) → False  (因为 2 > 2 = False，layer 2 还没存)  
    '''
    def has_item(self, layer_idx) -> bool:
        return len(self.key_cache) > layer_idx
    
    def get(self, layer_idx:int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def update(self, key_states, value_states, layer_idx):
        if len(self.key_cache) <= layer_idx:
            # 第一次添加
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]