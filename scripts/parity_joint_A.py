"""
joint 隔离对拍 进程 A：只测原版 joint_model 一次 forward。
随机造 joint 的输入 embeds（不经过 embedder/encoder），存盘 + 存原版 joint 输出。

运行（open-pi-zero venv）：
    cd /home/cxy/projects/open-pi-zero && \
    VLA_LOG_DIR=/tmp/vla_log VLA_DATA_DIR=/tmp/vla_data TRANSFORMERS_CACHE=$HOME/.cache/huggingface/hub \
    .venv/bin/python /home/cxy/projects/YuePi0/scripts/parity_joint_A.py
"""
import sys
import torch
from omegaconf import OmegaConf

OPZ = "/home/cxy/projects/open-pi-zero"
sys.path.insert(0, OPZ)
CKPT = f"{OPZ}/checkpoints/bridge_beta_step19296_2024-12-26_22-30_42.pt"
CFG = f"{OPZ}/config/eval/bridge.yaml"
OUT = "/tmp/parity_joint_io.pt"
OmegaConf.register_new_resolver("now", lambda fmt="": "na", replace=True)


def main():
    torch.manual_seed(0)
    dtype = torch.float32
    device = torch.device("cpu")

    cfg = OmegaConf.load(CFG)
    OmegaConf.resolve(cfg)
    from src.model.vla.pizero import PiZero
    model = PiZero(cfg, use_ddp=False)
    raw = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(raw["model"], strict=True)
    model.to(dtype).to(device).eval()

    B = 1
    n_vlm = model.max_image_text_tokens   # 276
    n_prop = model.num_proprio_tokens     # 1
    n_act = model.horizon_steps           # 4
    h_vlm = model.image_text_hidden_size  # 2048
    h_prop = model.proprio_hidden_size    # 1024
    h_act = model.action_hidden_size      # 1024

    # 随机但固定的 joint 输入 embeds
    vlm_emb = torch.randn(B, n_vlm, h_vlm, dtype=dtype)
    prop_emb = torch.randn(B, n_prop, h_prop, dtype=dtype)
    act_emb = torch.randn(B, n_act, h_act, dtype=dtype)

    # 用一个全有效的 attention_mask 走原版 build，拿到标准 block mask + position_ids
    attention_mask = torch.ones(B, n_vlm, dtype=torch.long)
    causal_mask, vlm_pos, prop_pos, act_pos = \
        model.build_causal_mask_and_position_ids(attention_mask, dtype=dtype)

    # time_cond：默认 adaptive_mode=None，所以 joint 内部不用 time_cond（仅 action_encoder 用）
    # 这里只测 joint，传 None
    kv_caches = model.joint_model.build_mixture_caches()
    with torch.no_grad():
        out = model.joint_model(
            attention_mask=causal_mask,
            position_ids_all={"vlm": vlm_pos, "proprio": prop_pos, "action": act_pos},
            embeds_all={
                "vlm": vlm_emb.clone(),
                "proprio": prop_emb.clone(),
                "action": act_emb.clone(),
            },
            time_cond=None,
            kv_caches=kv_caches,
            cache_mode="no_append",
        )
    act_out = out["action"]

    torch.save({
        "vlm_emb": vlm_emb, "prop_emb": prop_emb, "act_emb": act_emb,
        "attention_mask": attention_mask,
        "joint_action_out": act_out,
    }, OUT)
    print(f"[jointA] saved -> {OUT}")
    print(f"[jointA] joint action out shape={tuple(act_out.shape)} "
          f"mean={act_out.mean().item():.4f} std={act_out.std().item():.4f} "
          f"absmax={act_out.abs().max().item():.4f}")


if __name__ == "__main__":
    main()
