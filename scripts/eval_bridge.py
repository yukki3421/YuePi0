"""
推理验证: 加载训练好的 checkpoint, 喂真实图+指令, 让模型预测 action chunk,
和数据集里的真实 action 对比, 看模型到底学没学到 "看图 -> 出动作"。

跑法:
    uv run scripts/eval_bridge.py

注意:
    - 预测和真实 action 都在归一化空间, 直接比就行;
      额外再反归一化回物理量(米/弧度)给人看。
    - flow matching 推理是从随机噪声出发的, 每次结果会略有不同, 属正常。
"""

import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.bridge_dataset import BridgeDataset
from model.vla.processing import VLAPreProcessor
from model.vla.yuepi0 import PiZero
from model.utils import to_device_bf16


def preprocess_batch(raw_batch, processor):
    """和 train.py 里完全一致: raw -> 模型输入 (注意推理不需要 action)。"""
    images = raw_batch["image"].squeeze(1).permute(0, 3, 1, 2)
    proprio = raw_batch["proprio"]
    texts = raw_batch["text"]

    output = processor(prompts=texts, images=images, truncation=True)
    return {
        "input_ids": output["input_ids"],
        "attention_mask": output["attention_mask"],
        "pixel_values": output["pixel_values"],
        "proprio": proprio,
    }


def main():
    # 1. 配置 + 设备
    config = OmegaConf.load("config/realdataTrain.yaml")
    OmegaConf.resolve(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. dataset (顺便拿到归一化用的 mean/std)
    dataset = BridgeDataset(config, config.data_dir, config.max_episodes)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    action_mean = torch.tensor(dataset.action_mean)
    action_std = torch.tensor(dataset.action_std)

    # 3. processor
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_path, padding_side="right")
    processor = VLAPreProcessor(
        tokenizer=tokenizer,
        num_image_token=config.vision_config.num_image_tokens,
        max_seq_len=config.max_seq_len,
    )

    # 4. 模型 + 加载训练好的权重 (不再加载 PaliGemma 原始权重, 直接用 ckpt 覆盖全部)
    model = PiZero(config)
    ckpt_path = Path("checkpoints/yuepi0_bridge.pt")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(torch.bfloat16).to(device)
    model.eval()
    print(f"loaded checkpoint from {ckpt_path}")

    # 5. 取一个 batch, 推理
    raw_batch = next(iter(loader))
    inputs = preprocess_batch(raw_batch, processor)
    inputs = to_device_bf16(inputs, device)

    pred = model.infer_action(inputs, num_inference_steps=config.num_inference_steps)
    pred = pred.float().cpu()                       # (B, horizon, 7) 归一化空间
    gt = raw_batch["action"].float()                # (B, horizon, 7) 归一化空间

    # 6. 对比 (只看每个样本 chunk 的第 0 步动作, 最直观)
    print("\n" + "=" * 70)
    print("预测 vs 真实 (归一化空间, 只展示 action chunk 第 0 步)")
    print("=" * 70)
    for b in range(pred.shape[0]):
        text = raw_batch["text"][b]
        p0 = pred[b, 0].numpy().round(3)
        g0 = gt[b, 0].numpy().round(3)
        mse = ((pred[b] - gt[b]) ** 2).mean().item()
        print(f"\n[{b}] task = {text!r}")
        print(f"    pred = {p0.tolist()}")
        print(f"    gt   = {g0.tolist()}")
        print(f"    整个 chunk MSE = {mse:.4f}")

    # 7. 基线对照: 纯随机猜 (标准正态) 的 MSE, 看模型有没有比瞎猜强
    rand_mse = ((torch.randn_like(gt) - gt) ** 2).mean().item()
    model_mse = ((pred - gt) ** 2).mean().item()
    print("\n" + "=" * 70)
    print(f"模型预测 MSE   = {model_mse:.4f}")
    print(f"随机猜 MSE     = {rand_mse:.4f}   (标准正态, 越接近说明模型越没学到)")
    print(f"-> 模型{'有效' if model_mse < rand_mse else '没比瞎猜强'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
