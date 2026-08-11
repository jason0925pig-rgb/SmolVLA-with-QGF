#!/usr/bin/env bash
set -euo pipefail

cd /home/zwwl_user3/World_Model/repos/guided-action-flow
source .venv-a800-py312/bin/activate

export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=egl
export WANDB_MODE=disabled
export LIBERO_CONFIG_PATH=/home/zwwl_user3/World_Model/repos/guided-action-flow/.libero_configs/vanilla
export PYTHONPATH=/home/zwwl_user3/World_Model/repos/guided-action-flow/src:/home/zwwl_user3/World_Model/repos/guided-action-flow/third_party/lerobot/src
export CUDA_VISIBLE_DEVICES=0

python scripts/counterfactual_probe.py \
  --policy-path /home/zwwl_user3/World_Model/repos/guided-action-flow/checkpoints/smolvla_libero \
  --output-dir runs/counterfactual_abc_small_20260705 \
  --task libero_spatial \
  --task-ids "[3]" \
  --seed 3000 \
  --n-episodes 2 \
  --branch-steps "[20,60,100]" \
  --prefix-steps 12 \
  --tail-max-steps 220 \
  --noise-std 0.45 \
  --ranking-epochs 80 \
  --critic-paths \
    /home/zwwl_user3/World_Model/repos/guided-action-flow/runs/paper_repro_20260704_critic_train200_best_seed0/critic.pt \
    /home/zwwl_user3/World_Model/repos/guided-action-flow/runs/paper_repro_20260704_critic_train200_best_seed1/critic.pt \
    /home/zwwl_user3/World_Model/repos/guided-action-flow/runs/paper_repro_20260704_critic_train200_best_seed2/critic.pt \
  --qgf-beta 3.0 \
  --qgf-grad-clip-norm 1.0
