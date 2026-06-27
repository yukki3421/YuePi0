"""
对拍 进程 A：在 open-pi-zero 的环境里跑原版 PiZero.infer_action_naive，
生成「标准答案」存到磁盘，供进程 B（YuePi0）比对。

必须用 open-pi-zero 的 venv 运行：
    cd /home/cxy/projects/open-pi-zero && \
    VLA_LOG_DIR=/tmp/vla_log VLA_DATA_DIR=/tmp/vla_data TRANSFORMERS_CACHE=$HOME/.cache/huggingface/hub \
    .venv/bin/python /home/cxy/projects/YuePi0/scripts/parity_A_reference.py

存盘内容（/tmp/parity_io.pt）：
    input_ids, pixel_values, attention_mask, proprios  —— 原始输入（两边共用）
    noise                                              —— 初始噪声（两边共用，保证可比）
    action_ref                                         —— 原版输出（标准答案）
"""
import sys
import torch
from omegaconf import OmegaConf

OPZ = "/home/cxy/projects/open-pi-zero"
sys.path.insert(0, OPZ)

CKPT = f"{OPZ}/checkpoints/bridge_beta_step19296_2024-12-26_22-30_42.pt"
CFG = f"{OPZ}/config/eval/bridge.yaml"
OUT = "/tmp/parity_io.pt"

# hydra 注入的 ${now:} resolver，裸 OmegaConf 不认识，注册一个假的
OmegaConf.register_new_resolver("now", lambda fmt="": "na", replace=True)


def main():
    torch.manual_seed(0)
    dtype = torch.float32          # 对拍要精度，统一 float32
    device = torch.device("cpu")   # CPU 上跑，去掉 GPU 非确定性

    cfg = OmegaConf.load(CFG)
    OmegaConf.resolve(cfg)

    from src.model.vla.pizero import PiZero
    model = PiZero(cfg, use_ddp=False)

    # 加载原始权重（原版命名，strict）
    raw = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(raw["model"], strict=True)
    model.to(dtype).to(device).eval()

    # ---- 造一组随机但合法的输入 ----
    B = 1
    max_itt = model.max_image_text_tokens   # 276
    cond = model.num_proprio_tokens         # 1
    pdim = model.proprio_dim                # 7
    horizon = model.horizon_steps           # 4
    adim = model.action_dim                 # 7
    vocab = model.vocab_size

    # input_ids: 前 256 个是图像占位符，其余随机文本 token，最后留几个 padding
    image_token = model.image_token_index
    n_valid = max_itt - 10                  # 留 10 个 padding 位
    input_ids = torch.randint(0, vocab, (B, max_itt), dtype=torch.long)
    input_ids[:, :256] = image_token        # 前 256 个图像占位
    input_ids[:, n_valid:] = model.pad_token_id

    attention_mask = torch.ones(B, max_itt, dtype=torch.long)
    attention_mask[:, n_valid:] = 0          # padding 位 mask=0

    pixel_values = torch.randn(B, 3, 224, 224, dtype=dtype)
    proprios = torch.randn(B, cond, pdim, dtype=dtype)

    # 初始噪声（两边共用的核心）
    noise = torch.randn(B, horizon, adim, dtype=dtype)

    # ---- build mask + 跑原版 naive 推理 ----
    causal_mask, vlm_pos, proprio_pos, action_pos = \
        model.build_causal_mask_and_position_ids(attention_mask, dtype=dtype)

    with torch.no_grad():
        action_ref = model.infer_action_naive(
            input_ids=input_ids,
            pixel_values=pixel_values,
            causal_mask=causal_mask,
            vlm_position_ids=vlm_pos,
            proprio_position_ids=proprio_pos,
            action_position_ids=action_pos,
            proprios=proprios,
            noise=noise,
        )

    torch.save({
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "attention_mask": attention_mask,
        "proprios": proprios,
        "noise": noise,
        "action_ref": action_ref,
    }, OUT)

    print(f"[A] saved -> {OUT}")
    print(f"[A] action_ref shape={tuple(action_ref.shape)}")
    print(f"[A] action_ref=\n{action_ref}")


if __name__ == "__main__":
    main()
