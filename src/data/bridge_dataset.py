import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# 关键: 不让 TF 占 GPU (和 inspect 脚本一样)                                                                                         
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # 关 TF 日志
                                                                                                                                       
import tensorflow as tf
import tensorflow_datasets as tfds                                                                                                   
                                                                                                                                    
tf.config.set_visible_devices([], "GPU")   # 只对 TF 隐藏 GPU, 不动 torch 

class BridgeDataset(Dataset):
    def __init__(self, cfg, data_dir, max_episodes=None):
        super().__init__()

        # ===== 1. 几何尺寸 (从 cfg 拿, 对齐 fake_dataset) =====                                                                     
        self.image_size = cfg.vision_config.image_size   # 224
        self.cond_steps = cfg.cond_steps                 # 1                                                                         
        self.horizon_steps = cfg.horizon_steps           # 4                                                                         
        self.action_dim = cfg.action_dim                 # 7                                                                         
        self.proprio_dim = cfg.proprio_dim               # 7

        data_dir = Path(data_dir)

        # ===== 2. 读归一化 stats =====                                                                                              
        # TODO: glob action_proprio_stats_*.json, 读 mean/std
        stats_files = list(data_dir.glob("action_proprio_stats_*.json"))
        assert len(stats_files) > 0, f"No stats file found in {data_dir}"
        stats_path = stats_files[0]
        
        with open(stats_path, "r") as f:
            stats = json.load(f)

        # 存成 self.action_mean / action_std / proprio_mean / proprio_std                                                      
        # (np.array, float32)    
        self.action_mean = np.array(stats["action"]["mean"], dtype=np.float32)
        self.action_std = np.array(stats["action"]["std"], dtype=np.float32)
        self.proprio_mean = np.array(stats["proprio"]["mean"], dtype=np.float32)
        self.proprio_std = np.array(stats["proprio"]["std"], dtype=np.float32)

        # ===== 3. 遍历 shard, 把所有 sample 预先算好 =====                                                                          
        builder = tfds.builder_from_directory(str(data_dir))
        ds = builder.as_dataset(split="train", shuffle_files=False) 

        self.samples = []
        for ep_idx, episode in enumerate(ds):
            if max_episodes is not None and ep_idx >= max_episodes:
                break

            steps = list(episode["steps"])
            T = len(steps)

            # TODO: 取 language, 如果是空字符串就 skip 这条 episode 
            language = steps[0]['language_instruction'].numpy().decode("utf-8", errors="ignore")
            if not language:
                continue

            # TODO: stack actions / states -> (T, 7)
            actions = np.stack([s['action'].numpy() for s in steps])
            states = np.stack([s['observation']['state'].numpy() for s in steps])

            # 将这条episode展开成T个样本
            for t in range(T):
                sample = self._make_sample(steps, actions, states, t, T, language)
                self.samples.append(sample) 
        print(f"[BridgeDataset] loaded {len(self.samples)} samples "                                                                 
                f"from {ep_idx} episodes")  

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        return self.samples[index]
    
    def _make_sample(self, steps, actions, states, t, T, language):
        # iamge: iamge_0[t] -> resize 224 -> 保持uint8 HWC
        image = steps[t]["observation"]["image_0"].numpy()
        image_224 = tf.image.resize(image, (224, 224))
        image_224 = tf.cast(image_224, tf.uint8).numpy()
        # 加上cond_steps维
        image_224 = image_224[None]

        # proprio
        state = states[t]
        state_norm = (state - self.proprio_mean) / self.proprio_std
        state_norm = state_norm[None]

        action_chunk = []
        # action
        for k in range(self.horizon_steps):
            idx = t + k
            if idx >= T:
                idx = T - 1
            action_chunk.append(actions[idx])
        action_chunk = np.stack(action_chunk) # (4, 7)
        action_chunk_norm = (action_chunk - self.action_mean ) / self.action_std

        return {
            "image": torch.from_numpy(image_224), # (1, 224, 224, 3) unit8
            "proprio": torch.from_numpy(state_norm).float(), # (1, 7)
            "action": torch.from_numpy(action_chunk_norm).float(), # (4, 7)
            "text": language # str
        }
    
if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "vision_config": {"image_size": 224},
        "cond_steps": 1,
        "horizon_steps": 4,
        "action_dim": 7,
        "proprio_dim": 7,
    })
    ds = BridgeDataset(cfg=cfg, data_dir=Path.home() / "datasets/bridge_dataset/1.0.0", max_episodes=5)

    s = ds[0]
    for k, v in s.items():
        if hasattr(v, "shape"):
            print(f"{k:8s} {tuple(v.shape)} {v.dtype}")
        else:
            print(f"{k:8s} {v!r}")