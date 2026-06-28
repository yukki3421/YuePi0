import numpy as np

'''数据的归一化和反归一化'''

class BaseEnvAdapter:
    def __init__(self):
        pass
    # min-max归一化, bound 归一化(按最小/最大值缩放到 [-1,1])
    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1, 
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)
    
    #  denormalize_bound:反过来,把 [-1,1] 还原回 [min,max]
    def denormalize_bound(
        self, 
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1, 
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        clip_range = clip_max - clip_min
        rdata = (data - clip_min) / clip_range * (data_max - data_min) + data_min
        return rdata
    
    # gaussian 归一化 (按均值/)
    def normalize_gaussian(
        self,
        data: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        eps: float = 1e-8,
    ) -> np.ndarray:
        return (data - mean) / (std + eps)
    
    def denormalize_gaussian(
        self,
        data: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        eps: float = 1e-8,
    ) -> np.ndarray:
        return data * (std + eps) + mean