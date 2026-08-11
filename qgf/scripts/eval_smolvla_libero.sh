#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ylhc/miniconda3/envs/gaf-libero/bin/python}"
POLICY_PATH="${POLICY_PATH:-$ROOT_DIR/checkpoints/smolvla_libero}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/runs/smolvla_libero_smoke}"
TASK="${TASK:-libero_spatial}"
TASK_IDS="${TASK_IDS:-[0]}"
N_EPISODES="${N_EPISODES:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ASYNC_ENVS="${ASYNC_ENVS:-false}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$ROOT_DIR/.libero_configs/vanilla}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-gaf-libero}"

"$PYTHON_BIN" -m lerobot.scripts.lerobot_eval \
  --output_dir="$OUTPUT_DIR" \
  --policy.path="$POLICY_PATH" \
  --env.type=libero \
  --env.task="$TASK" \
  --env.task_ids="$TASK_IDS" \
  --eval.batch_size="$BATCH_SIZE" \
  --eval.n_episodes="$N_EPISODES" \
  --eval.use_async_envs="$ASYNC_ENVS" \
  --env.max_parallel_tasks=1 \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
  "$@"
