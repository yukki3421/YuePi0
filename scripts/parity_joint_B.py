"""
joint 隔离对拍 进程 B：把 A 存的同一份 embeds 喂给 YuePi0.joint，比对 joint 输出。

运行（YuePi0 venv）：
    cd /home/cxy/projects/YuePi0 && PYTHONPATH=src .venv/bin/python scripts/parity_joint_B.py
"""
import torch
from omegaconf import OmegaConf

from model.load_pretrained import load_pretrained_pizero

CFG = "config/yuepi0.yaml"
CKPT = "/home/cxy/projects/open-pi-zero/checkpoints/bridge_beta_step19296_2024-12-26_22-30_42.pt"
IO = "/tmp/parity_joint_io.pt"


def main():
    dtype = torch.float32
    device = torch.device("cpu")
    cfg = OmegaConf.load(CFG)
    model = load_pretrained_pizero(cfg, CKPT, strict=True)
    model.to(dtype).to(device).eval()

    io = torch.load(IO, map_location="cpu", weights_only=False)
    vlm_emb = io["vlm_emb"].to(dtype)
    prop_emb = io["prop_emb"].to(dtype)
    act_emb = io["act_emb"].to(dtype)
    attention_mask = io["attention_mask"]
    ref = io["joint_action_out"].to(dtype)

    # 用 YuePi0 自己的 build_mask 拿 mask + position_ids（顺便验证两边 build 是否等价）
    causal_mask, vlm_pos, prop_pos, act_pos = \
        model.build_mask_and_position_ids(attention_mask, dtype)

    with torch.no_grad():
        out = model.joint(
            causal_mask,
            {"vlm": vlm_pos, "proprio": prop_pos, "action": act_pos},
            {"vlm": vlm_emb.clone(), "proprio": prop_emb.clone(), "action": act_emb.clone()},
            time_cond=None,
        )
    act_out = out["action"]

    diff = (act_out - ref).abs()
    print(f"[jointB] out shape={tuple(act_out.shape)} "
          f"mean={act_out.mean().item():.4f} std={act_out.std().item():.4f} "
          f"absmax={act_out.abs().max().item():.4f}")
    print(f"[jointB] ref  mean={ref.mean().item():.4f} std={ref.std().item():.4f} "
          f"absmax={ref.abs().max().item():.4f}")
    print(f"[jointB] max abs diff = {diff.max().item():.3e}")
    print(f"[jointB] mean abs diff = {diff.mean().item():.3e}")
    if diff.max().item() < 1e-3:
        print("[jointB] PASS ✅  joint 等价 -> bug 在 joint 外面（embedder/encoder/迭代）")
    else:
        print("[jointB] FAIL ❌  bug 在 joint 内部")


if __name__ == "__main__":
    main()
