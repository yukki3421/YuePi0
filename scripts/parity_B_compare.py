"""
对拍 进程 B：在 YuePi0 的环境里，读进程 A 存的同一份输入 + 同一份噪声，
喂给 YuePi0.infer_action，比对输出和原版「标准答案」的数值差异。

必须用 YuePi0 的 venv 运行（先跑完 parity_A_reference.py）：
    cd /home/cxy/projects/YuePi0 && PYTHONPATH=src .venv/bin/python scripts/parity_B_compare.py
"""
import torch
from omegaconf import OmegaConf

from model.load_pretrained import load_pretrained_pizero

CFG = "config/yuepi0.yaml"
CKPT = "/home/cxy/projects/open-pi-zero/checkpoints/bridge_beta_step19296_2024-12-26_22-30_42.pt"
IO = "/tmp/parity_io.pt"


def main():
    dtype = torch.float32
    device = torch.device("cpu")

    cfg = OmegaConf.load(CFG)
    model = load_pretrained_pizero(cfg, CKPT, strict=True)
    model.to(dtype).to(device).eval()

    io = torch.load(IO, map_location="cpu", weights_only=False)

    # YuePi0.infer_action 吃 batch 字典；key 名按 yuepi0.py forward 里的约定
    batch = {
        "input_ids": io["input_ids"].to(device),
        "pixel_values": io["pixel_values"].to(dtype).to(device),
        "attention_mask": io["attention_mask"].to(device),
        "proprio": io["proprios"].to(dtype).to(device),   # 注意：YuePi0 用单数 'proprio'
    }
    noise = io["noise"].to(dtype).to(device)
    action_ref = io["action_ref"].to(dtype).to(device)

    with torch.no_grad():
        action_yue = model.infer_action(
            batch,
            num_inference_steps=model.num_inference_steps,
            noise=noise,
        )

    diff = (action_yue - action_ref).abs()
    print(f"[B] action_yue shape={tuple(action_yue.shape)}")
    print(f"[B] action_ref =\n{action_ref}")
    print(f"[B] action_yue =\n{action_yue}")
    print(f"[B] max abs diff = {diff.max().item():.3e}")
    print(f"[B] mean abs diff = {diff.mean().item():.3e}")

    if diff.max().item() < 1e-3:
        print("[B] PASS ✅  两边前向数值等价")
    else:
        print("[B] FAIL ❌  数值差异过大，forward 逻辑有出入，需排查")


if __name__ == "__main__":
    main()
