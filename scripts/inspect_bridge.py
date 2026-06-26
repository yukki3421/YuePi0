"""
看 Bridge V2 一条 trajectory 长什么样.

跑法:
    uv run scripts/inspect_bridge.py --data-dir $HOME/datasets/bridge_dataset/1.0.0

输出:
    1. 数据集元信息 (总 episode 数 / split 信息)
    2. 拿一条 trajectory 的 episode_metadata
    3. 这条 trajectory 的 step 数 / language_instruction
    4. 第 0 帧的所有字段 (shape/dtype/value)
    5. 整条轨迹上 action / state 的 min/max/mean
    6. 保存第 0 帧四个视角的 RGB 到 tmp/
"""

import argparse
import os
from pathlib import Path

# 关键: 不让 TF 占 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

tf.config.set_visible_devices([], "GPU")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path.home() / "datasets" / "bridge_dataset" / "1.0.0"),
    )
    parser.add_argument("--out-dir", type=str, default="tmp/bridge_inspect")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ============ 1) 加载 builder ============
    print("=" * 60)
    print("[1] Loading TFDS builder")
    print("=" * 60)
    builder = tfds.builder_from_directory(str(data_dir)) # 从一个已经存在于磁盘上的 TFDS 数据集目录，直接构造一个 builder对象，跳过下载和重新生成的步骤
    info = builder.info
    print(f"  name        : {info.name}")
    print(f"  version     : {info.version}")
    print(f"  splits      : {list(info.splits.keys())}")
    for split_name, split_info in info.splits.items():
        # 注意: 我们只下了 1 个 shard, 所以这里报的 episode 数是整体元数据,
        # 实际能读到的只是 1 shard 里的
        print(f"    {split_name:5s}: {split_info.num_examples} episodes (整个数据集)")
    print(f"  features    :")
    print(info.features)

    # ============2) 列出前N条episode摘要 ==========
    print()
    print("="*60)
    print("[episode summary]")
    print("="*60)

    # 用 read_config 限制读哪些文件
    ds = builder.as_dataset(split="train", shuffle_files=False)
    for ep_idx, episode in enumerate(ds):
        if ep_idx > 5:
            break

        # 1. steps是episode 内部的变长step 序列
        steps = list(episode['steps'])
        n_steps = len(steps)

        # 2. 每条episode 的语言指令, 每个step里都重复存一份
        lang = steps[0]['language_instruction'].numpy().decode("utf-8", errors="ignore")

        # 3. episode级别的metadata
        file_path = episode['episode_metadata']['file_path'].numpy().decode(
            "utf-8", errors="ignore"
        )

        print(f"[{ep_idx}] steps={n_steps}  lang={lang!r}")
        print(f"    file_path={file_path}")

        # 4. 把整条episode的action/state stack 起来
        actions = np.stack([s['action'].numpy() for s in steps])
        states = np.stack([s['observation']['state'].numpy() for s in steps])
        print(f"    action shape={actions.shape}, state shape={states.shape}")

        # 5. 检查第 0 帧四个相机是不是 dummy
        step0 = steps[0]
        for cam_name in ["image_0", "image_1", "image_2", "image_3"]:
            img = step0["observation"][cam_name].numpy()
            mean = img.mean()
            real = mean > 1.0
            print(f"    {cam_name} mean={mean:.1f} real={real}")

        print()


    # ============ 2) 拿一条 episode ============
    print()
    print("=" * 60)
    print("[2] Reading one episode from train shard 00000")
    print("=" * 60)
    # 用 read_config 限制读哪些文件
    ds = builder.as_dataset(split="train", shuffle_files=False)
    # 取第一条
    episode = next(iter(ds))

    # ---- episode_metadata
    print("  [episode_metadata]")
    for k, v in episode["episode_metadata"].items():
        val = v.numpy()
        if isinstance(val, bytes):
            val = val.decode("utf-8", errors="ignore")
        print(f"    {k:15s} = {val!r}")

    # ---- steps: 是一个 Dataset, 转成 list
    steps = list(episode["steps"])
    n_steps = len(steps)
    print(f"  [steps] count = {n_steps}")
    lang0 = steps[0]["language_instruction"].numpy().decode("utf-8", errors="ignore")
    print(f"  language_instruction = {lang0!r}")

    # ============ 3) 第 0 帧详细字段 ============
    print()
    print("=" * 60)
    print("[3] Step 0 — all fields")
    print("=" * 60)
    step0 = steps[0]

    def describe(name, tensor):
        arr = tensor.numpy()
        if isinstance(arr, bytes):
            s = arr.decode("utf-8", errors="ignore")
            print(f"  {name:40s} (text)            = {s!r}")
            return
        shape = arr.shape
        dtype = arr.dtype
        if arr.ndim == 0:
            print(f"  {name:40s} {str(shape):15s} {dtype}  = {arr.item()}")
        elif arr.ndim == 1 and arr.size <= 16:
            print(f"  {name:40s} {str(shape):15s} {dtype}  = {arr.tolist()}")
        else:
            print(
                f"  {name:40s} {str(shape):15s} {dtype}  "
                f"min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}"
            )

    # 顶层 step 字段
    for k, v in step0.items():
        if k == "observation":
            continue
        describe(k, v)
    # observation
    print("  [observation]")
    for k, v in step0["observation"].items():
        describe(f"observation.{k}", v)

    # ============ 4) 整条轨迹的 action / state 分布 ============
    print()
    print("=" * 60)
    print("[4] action / state stats over entire trajectory")
    print("=" * 60)
    actions = np.stack([s["action"].numpy() for s in steps])  # (T, 7)
    states = np.stack([s["observation"]["state"].numpy() for s in steps])  # (T, 7)

    print(f"  action  shape = {actions.shape}")
    for i in range(actions.shape[1]):
        col = actions[:, i]
        print(
            f"    dim {i}: min={col.min():+.4f}  max={col.max():+.4f}  "
            f"mean={col.mean():+.4f}  std={col.std():.4f}"
        )

    print(f"  state   shape = {states.shape}")
    for i in range(states.shape[1]):
        col = states[:, i]
        print(
            f"    dim {i}: min={col.min():+.4f}  max={col.max():+.4f}  "
            f"mean={col.mean():+.4f}  std={col.std():.4f}"
        )

    # ============ 5) 保存第 0 / 中间 / 最后 一帧图像 ============
    print()
    print("=" * 60)
    print("[5] Saving image_0 at first / middle / last step")
    print("=" * 60)
    indices = [0, n_steps // 2, n_steps - 1]
    for idx in indices:
        img = steps[idx]["observation"]["image_0"].numpy()
        Image.fromarray(img).save(out_dir / f"step{idx:04d}_image_0.png")
        print(f"  saved step {idx} -> {out_dir}/step{idx:04d}_image_0.png  shape={img.shape}")

    # 看看其他视角第 0 帧是不是 dummy (has_image_X)
    for cam_name in ["image_1", "image_2", "image_3"]:
        img = steps[0]["observation"][cam_name].numpy()
        Image.fromarray(img).save(out_dir / f"step0000_{cam_name}.png")
        has_key = f"has_{cam_name}"
        has_flag = bool(episode["episode_metadata"][has_key].numpy())
        print(f"  saved {cam_name}  has_flag={has_flag}")

    print()
    print("Done. 检查 tmp/bridge_inspect/ 下的 PNG.")


def get_action_chunk_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path.home() / "datasets" / "bridge_dataset" / "1.0.0"),
    )
    parser.add_argument("--out-dir", type=str, default="tmp/bridge_inspect")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ============ 1) 加载 builder ============
    print("=" * 60)
    print("[1] Loading TFDS builder")
    print("=" * 60)
    builder = tfds.builder_from_directory(str(data_dir)) # 从一个已经存在于磁盘上的 TFDS 数据集目录，直接构造一个 builder对象，跳过下载和重新生成的步骤
    # 用 read_config 限制读哪些文件
    ds = builder.as_dataset(split="train", shuffle_files=False)
    action_horizon = 4

    print(" [action chunk demo]")
    for ep_idx, episode in enumerate(ds):
        if ep_idx > 5:
            break

        # 1. steps是episode 内部的变长step 序列
        steps = list(episode['steps'])
        n_steps = len(steps)
        actions = np.stack([s['action'].numpy() for s in steps])
        T = actions.shape[0]

        for t in [0, T // 2, T -2]:
            indices = []
            chunk = []

            for k in range(action_horizon):
                idx = t + k
                if idx >= T:
                    idx = T - 1
                indices.append(idx)
                chunk.append(actions[idx])

            chunk = np.stack(chunk)
            print(f"      t={t}")
            print(f"        indices={indices}")
            print(f"        chunk shape={chunk.shape}")
            print(f"        first action={chunk[0].round(4).tolist()}")
            print(f"        last  action={chunk[-1].round(4).tolist()}")

def get_image_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path.home() / "datasets" / "bridge_dataset" / "1.0.0"),
    )
    parser.add_argument("--out-dir", type=str, default="tmp/bridge_inspect")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    builder = tfds.builder_from_directory(str(data_dir)) # 从一个已经存在于磁盘上的 TFDS 数据集目录，直接构造一个 builder对象，跳过下载和重新生成的步骤
    # 用 read_config 限制读哪些文件
    ds = builder.as_dataset(split="train", shuffle_files=False)
    action_horizon = 4

    print(" [action chunk demo]")
    for ep_idx, episode in enumerate(ds):

        # 只演示第一条 episode
        if ep_idx != 0:
            break
         # 1. steps是episode 内部的变长step 序列
        steps = list(episode['steps'])
        actions = np.stack([s['action'].numpy() for s in steps])
        T = actions.shape[0]
        t = T // 2

        # 1. 当前时刻图像
        image = steps[t]["observation"]["image_0"].numpy()

        # 2. 当前时刻 state
        state = steps[t]["observation"]["state"].numpy()

        # 3. episode 语言指令
        language = steps[0]["language_instruction"].numpy().decode("utf-8", errors="ignore")

        # 4. 从 t 开始切 action chunk
        indices = []
        chunk = []

        for k in range(action_horizon):
            idx = t + k
            if idx >= T:
                idx = T - 1

            indices.append(idx)
            chunk.append(actions[idx])

        action_chunk = np.stack(chunk)

        # 5. 打印
        print()
        print("=" * 60)
        print("[sample demo]")
        print("=" * 60)
        print(f"episode={ep_idx}, T={T}, t={t}")
        print(f"language={language!r}")
        print(f"image_0 shape={image.shape}, dtype={image.dtype}, mean={image.mean():.1f}")
        print(f"state shape={state.shape}, values={state.round(4).tolist()}")
        print(f"action_chunk indices={indices}")
        print(f"action_chunk shape={action_chunk.shape}")
        print(f"action_chunk first={action_chunk[0].round(4).tolist()}")
        print(f"action_chunk last ={action_chunk[-1].round(4).tolist()}")

def sample_demo():
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path.home() / "datasets" / "bridge_dataset" / "1.0.0"),
    )
    parser.add_argument("--out-dir", type=str, default="tmp/bridge_inspect")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print()                                                                                                                                              
    print("=" * 60)                                                                                                                                      
    print("[sample normalize demo]")                                                                                                                     
    print("=" * 60) 

    stats_files = list(data_dir.glob("action_proprio_stats_*.json"))
    assert len(stats_files) > 0, f"No stats file found in {data_dir}"
    stats_path = stats_files[0]
    
    with open(stats_path, "r") as f:
        stats = json.load(f)

    action_mean = np.array(stats["action"]["mean"], dtype=np.float32)
    action_std = np.array(stats["action"]["std"], dtype=np.float32)
    proprio_mean = np.array(stats["proprio"]["mean"], dtype=np.float32)
    proprio_std = np.array(stats["proprio"]["std"], dtype=np.float32)

    builder = tfds.builder_from_directory(str(data_dir))
    ds = builder.as_dataset(split="train", shuffle_files=False)

    episode = next(iter(ds))
    steps = list(episode['steps'])
    T = len(steps)

    actions = np.stack([s["action"].numpy() for s in steps])                                                                                             
    states = np.stack([s["observation"]["state"].numpy() for s in steps]) 

    t = T // 2

    image = steps[t]["observation"]["image_0"].numpy()
    state = states[t]
    language = steps[0]["language_instruction"].numpy().decode("utf-8", errors="ignore")

    # resize 256x256 -> 224x224 (PaliGemma / SigLIP 需要 224 输入)
    # tf.image.resize 返回 float32, 所以再 cast 回 uint8 保持和原图一致
    image_224 = tf.image.resize(image, (224, 224))
    image_224 = tf.cast(image_224, tf.uint8).numpy()

    print(f"stats_path = {stats_path.name}")
    print(f"T = {T}, t = {t}")
    print(f"language = {language!r}")
    print(f"image_0 raw shape = {image.shape}, dtype = {image.dtype}, mean = {image.mean():.1f}")
    print(f"image_0 224 shape = {image_224.shape}, dtype = {image_224.dtype}, mean = {image_224.mean():.1f}")

    action_horizon = 4
    indices = []
    chunk = []

    for k in range(action_horizon):
        idx = t + k
        if idx >= T:
            idx = T - 1
        indices.append(idx)
        chunk.append(actions[idx])
    action_chunk = np.stack(chunk)

    state_norm = (state - proprio_mean) / proprio_std
    action_chunk_norm = (action_chunk - action_mean) / action_std
    
    print()                                                                                                                                              
    print("[state]")                                                                                                                                     
    print(f"raw  shape = {state.shape}")                                                                                                                 
    print(f"raw  values = {state.round(4).tolist()}")                                                                                                    
    print(f"norm shape = {state_norm.shape}")                                                                                                            
    print(f"norm values = {state_norm.round(4).tolist()}")                                                                                               
                                                                                                                                                        
    print()                                                                                                                                              
    print("[action chunk]")                                                                                                                              
    print(f"indices = {indices}")                                                                                                                        
    print(f"raw  shape = {action_chunk.shape}")                                                                                                          
    print(f"norm shape = {action_chunk_norm.shape}")                                                                                                     
                                                                                                                                                        
    print(f"raw  first = {action_chunk[0].round(4).tolist()}")                                                                                           
    print(f"norm first = {action_chunk_norm[0].round(4).tolist()}")                                                                                      
                                                                                                                                                        
    print(f"raw  last  = {action_chunk[-1].round(4).tolist()}")                                                                                          
    print(f"norm last  = {action_chunk_norm[-1].round(4).tolist()}")                                                                                     
                                                                                                                                                        
    print()         
    print("[stats]")                                                                                                                                     
    print(f"action_mean  = {action_mean.round(4).tolist()}")                                                                                             
    print(f"action_std   = {action_std.round(4).tolist()}")                                                                                              
    print(f"proprio_mean = {proprio_mean.round(4).tolist()}")                                                                                            
    print(f"proprio_std  = {proprio_std.round(4).tolist()}") 


if __name__ == "__main__":
    # main()
    # get_action_chunk_demo()
    # get_image_demo()
    sample_demo()
