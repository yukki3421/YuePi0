#!/usr/bin/env bash
# YuePi0 SimplerEnv 仿真部署运行脚本
#
# 用法:
#   bash scripts/run_deploy.sh                          # 默认任务 widowx_carrot_on_plate
#   bash scripts/run_deploy.sh widowx_stack_cube        # 指定任务
#
# 可选任务 (bridge 权重):
#   widowx_carrot_on_plate / widowx_put_eggplant_in_basket
#   widowx_spoon_on_towel  / widowx_stack_cube
#
# 注意:
#   - 用 .venv/bin/python, 绝不用 uv run (会重装 setuptools 导致 sapien 报错)
#   - GPU 0 (GPU 1 常被别人的 ollama 占满)

set -e  # 任一命令失败就退出

# 切到项目根目录 (相对路径 config/ 才找得到)
cd "$(dirname "$0")/.."

# 任务名: 取第一个参数, 没传则用默认
TASK="${1:-widowx_stack_cube}"

# 环境变量
export CUDA_VISIBLE_DEVICES=1
export TRANSFORMERS_CACHE=/home/cxy/.cache/huggingface/hub
export PIZERO_CKPT=/home/cxy/projects/open-pi-zero/checkpoints/bridge_beta_step19296_2024-12-26_22-30_42.pt

# 跑
.venv/bin/python scripts/deploy_simpler.py --task "$TASK"
