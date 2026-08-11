#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source "${ENV_PREFIX:-${ROOT}/.venv}/bin/activate"

export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled
export PYTHONPATH="${ROOT}/src:${ROOT}/third_party/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CF_FILE=runs/counterfactual_multitask_spatial0to4_seed6100_ep4_h50_20260705/counterfactual_samples.pt
ROLLOUT_DIR=runs/exp_multitask_spatial0to4_ep30_seed9000_raw

for SEED in 0 1 2; do
  python scripts/train_counterfactual_mixed_critic.py \
    --rollout-data-dirs "${ROLLOUT_DIR}" \
    --counterfactual-files "${CF_FILE}" \
    --output-dir "runs/counterfactual_mixed_spatial0to4_h50_rw0p2_seed${SEED}_20260705" \
    --action-horizon 50 \
    --stride 1 \
    --gamma 0.99 \
    --action-key action_policy \
    --cf-action-key action_chunk_policy \
    --hidden-dim 512 \
    --depth 3 \
    --batch-size 256 \
    --pair-batch-size 128 \
    --epochs 30 \
    --lr 1.0e-3 \
    --weight-decay 1.0e-4 \
    --val-fraction 0.2 \
    --mse-weight 1.0 \
    --ranking-weight 0.2 \
    --seed "${SEED}" \
    --device cuda
done
