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

POLICY=/home/zwwl_user3/World_Model/repos/guided-action-flow/checkpoints/smolvla_libero
TASK=libero_spatial
TASK_IDS="[0,1,2,3,4]"
N_EPISODES=10
SEED=7200

ORIG0=/home/zwwl_user3/World_Model/repos/guided-action-flow/runs/paper_repro_20260704_critic_train200_best_seed0/critic.pt
ORIG1=/home/zwwl_user3/World_Model/repos/guided-action-flow/runs/paper_repro_20260704_critic_train200_best_seed1/critic.pt
ORIG2=/home/zwwl_user3/World_Model/repos/guided-action-flow/runs/paper_repro_20260704_critic_train200_best_seed2/critic.pt

CF0=/home/zwwl_user3/World_Model/repos/guided-action-flow/runs/counterfactual_mixed_spatial0to4_h50_rw0p2_seed0_20260705/critic.pt
CF1=/home/zwwl_user3/World_Model/repos/guided-action-flow/runs/counterfactual_mixed_spatial0to4_h50_rw0p2_seed1_20260705/critic.pt
CF2=/home/zwwl_user3/World_Model/repos/guided-action-flow/runs/counterfactual_mixed_spatial0to4_h50_rw0p2_seed2_20260705/critic.pt

python scripts/eval_policy.py \
  --policy-path "${POLICY}" \
  --output-dir runs/counterfactual_compare_spatial0to4_baseline_seed7200_20260705 \
  --task "${TASK}" \
  --task-ids "${TASK_IDS}" \
  --n-episodes "${N_EPISODES}" \
  --seed "${SEED}" \
  --device cuda \
  --max-videos 0

python scripts/eval_policy.py \
  --policy-path "${POLICY}" \
  --output-dir runs/counterfactual_compare_spatial0to4_original_qgf_seed7200_20260705 \
  --task "${TASK}" \
  --task-ids "${TASK_IDS}" \
  --n-episodes "${N_EPISODES}" \
  --seed "${SEED}" \
  --device cuda \
  --max-videos 0 \
  --critic-paths "${ORIG0}" "${ORIG1}" "${ORIG2}" \
  --qgf-beta 3.0 \
  --qgf-grad-clip-norm 1.0

python scripts/eval_policy.py \
  --policy-path "${POLICY}" \
  --output-dir runs/counterfactual_compare_spatial0to4_cf_qgf_seed7200_20260705 \
  --task "${TASK}" \
  --task-ids "${TASK_IDS}" \
  --n-episodes "${N_EPISODES}" \
  --seed "${SEED}" \
  --device cuda \
  --max-videos 0 \
  --critic-paths "${CF0}" "${CF1}" "${CF2}" \
  --qgf-beta 3.0 \
  --qgf-grad-clip-norm 1.0
