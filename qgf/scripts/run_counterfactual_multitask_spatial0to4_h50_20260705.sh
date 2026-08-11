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
  --output-dir runs/counterfactual_multitask_spatial0to4_seed6100_ep4_h50_20260705 \
  --task libero_spatial \
  --task-ids "[0,1,2,3,4]" \
  --seed 6100 \
  --n-episodes 4 \
  --branch-steps "[20,60,100,140]" \
  --prefix-steps 50 \
  --tail-max-steps 260 \
  --noise-std 0.35 \
  --save-every-records 9 \
  --skip-ranking \
  --critic-paths \
    /home/zwwl_user3/World_Model/repos/guided-action-flow/runs/paper_repro_20260704_critic_train200_best_seed0/critic.pt \
    /home/zwwl_user3/World_Model/repos/guided-action-flow/runs/paper_repro_20260704_critic_train200_best_seed1/critic.pt \
    /home/zwwl_user3/World_Model/repos/guided-action-flow/runs/paper_repro_20260704_critic_train200_best_seed2/critic.pt \
  --qgf-beta 3.0 \
  --qgf-grad-clip-norm 1.0
