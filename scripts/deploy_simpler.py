"""
YuePi0 在 SimplerEnv (bridge / widowx) 上的仿真部署脚本。

跑法 (YuePi0 venv, 已装好 simpler_env/sapien/maniskill):
    cd /home/cxy/projects/YuePi0 && \
    CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_CACHE=/home/cxy/.cache/huggingface/hub \
    PIZERO_CKPT=/path/to/bridge_xxx.pt \
    .venv/bin/python scripts/deploy_simpler.py --task widowx_carrot_on_plate

注意:
- 必须用 .venv/bin/python, 绝不用 uv run (uv run 会重装 setuptools 导致 sapien 报错)
- 从项目根目录跑, 否则 config/ 下的相对路径 (yuepi0.yaml / bridge_statistics.json) 找不到
- PIZERO_CKPT 指向 open-pi-zero 训练好的 bridge checkpoint

说明:
- env_adapter (BridgeSimplerAdapter) 已搬进本项目 src/agent/env_adapter, 负责 preprocess/postprocess/归一化
- 模型用 YuePi0 + 已对拍验证数值等价的 load_pretrained_pizero 加载原版权重
- 用 KV cache 两阶段推理: infer_action 内部 prefill vlm+proprio 一次, 再循环 denoise action
"""
import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
import imageio

import simpler_env
from model.load_pretrained import load_pretrained_pizero  # noqa: E402
from agent.env_adapter.simpler import BridgeSimplerAdapter

def main(args):
    device = torch.device("cuda:0")
    dtype = torch.float32
    # 配置
    yue_cfg = OmegaConf.load("config/yuepi0.yaml")
    deploy_cfg = OmegaConf.load("config/deploy_simpler.yaml")

    ckpt = deploy_cfg.checkpoint_path
    model = load_pretrained_pizero(yue_cfg, ckpt, strict=True)
    model.to(dtype).to(device).eval()
    print("[deploy] model ready")

    ada_cfg = deploy_cfg.env.adapter
    adapter = BridgeSimplerAdapter(
        dataset_statistics_path = ada_cfg.dataset_statistics_path, 
        pretrained_model_path= ada_cfg.pretrained_model_path,
        tokenizer_padding = ada_cfg.tokenizer_padding,
        num_image_tokens = ada_cfg.num_image_tokens,
        image_size = ada_cfg.image_size,
        max_seq_len = ada_cfg.max_seq_len,
    )
    adapter.reset()
    print("[deploy] adapter ready")

    env = simpler_env.make(args.task)
    successes = []
    for episode_id in range(args.n_eval_episode):
        obs, reset_info = env.reset(options={"obj_init_options": {"episode_id": episode_id}})
        instruction = env.get_language_instruction()
        print(f"[deploy] env={args.task} instruction={instruction!r}")

        recording = episode_id < args.n_video
        if recording:
            video_path = os.path.join(deploy_cfg.video_dir, f"{args.task}_ep{episode_id}.mp4")
            video_writer = imageio.get_writer(video_path)

        # preprocess → batch → 推理 → postprocess → 执行
        # 直到任务结束
        truncated = False
        success = False
        while not truncated:
            inputs = adapter.preprocess(env, obs, instruction)
            batch = {
                "input_ids": inputs["input_ids"].to(device),
                "pixel_values": inputs["pixel_values"].to(dtype).to(device),
                "attention_mask": inputs["attention_mask"].to(device),
                "proprio": inputs["proprio"].to(dtype).to(device)
            }
            with torch.no_grad():
                actions = model.infer_action(batch=batch, num_inference_steps=deploy_cfg.num_inference_steps)
            actions = actions[0].cpu().numpy()
            # 模型输出变成仿真器能执行的真实动作
            env_actions = adapter.postprocess(actions)
            for action in env_actions[: deploy_cfg.act_steps]:
                obs, reward, success, truncated, info = env.step(action)
                if recording:
                    # 从当前 obs 取一帧画面（[H,W,3] 的图像数组） 追加进视频
                    video_writer.append_data(adapter.get_video_frame(env, obs))
                if truncated:
                    break
        if recording:
            video_writer.close()
            print(f"[episode {episode_id}] [deploy] video saved to {video_path}")
        successes.append(success)
        print(f"[deploy] done. success={success}")
    success_rate = np.mean(successes)
    print(f"[deploy] success_rate = {success_rate} ({sum(successes)}/{len(successes)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # bridge（widowx 机器人）这套，你的权重能跑的有 4 个：
    # - widowx_carrot_on_plate（放萝卜，你现在用的）
    # - widowx_put_eggplant_in_basket（茄子放篮子）
    # - widowx_spoon_on_towel（勺子放毛巾上）
    # - widowx_stack_cube（叠方块）
    parser.add_argument("--task", type=str, default="widowx_carrot_on_plate")
    parser.add_argument("--n_eval_episode", type=int, default=240)
    parser.add_argument("--n_video", type=int, default=4)
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    main(args)


