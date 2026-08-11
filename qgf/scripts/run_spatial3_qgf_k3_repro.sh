#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_PREFIX="${ENV_PREFIX:-$ROOT/.venv-a800-py312}"
PY="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
TASK="${TASK:-libero_spatial}"
TASK_IDS="${TASK_IDS:-[3]}"
BASE_SEED="${BASE_SEED:-3000}"
TRAIN_SEED="${TRAIN_SEED:-5000}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-50}"
N_TRAIN_EPISODES="${N_TRAIN_EPISODES:-200}"
BETA="${BETA:-3}"
UNCERTAINTY_SCALE="${UNCERTAINTY_SCALE:-20}"
MIN_GATE="${MIN_GATE:-0.1}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"

RUN_TAG="${RUN_TAG:-spatial3_k3_gate20_seed${BASE_SEED}}"
BASELINE_DIR="runs/${RUN_TAG}_baseline_ep${N_EVAL_EPISODES}"
ROLLOUT_DIR="runs/${RUN_TAG}_train${N_TRAIN_EPISODES}_seed${TRAIN_SEED}"
QGF_DIR="runs/${RUN_TAG}_qgf_ep${N_EVAL_EPISODES}"
CRITIC0_DIR="runs/${RUN_TAG}_critic_train${N_TRAIN_EPISODES}_seed0"
CRITIC1_DIR="runs/${RUN_TAG}_critic_train${N_TRAIN_EPISODES}_seed1"
CRITIC2_DIR="runs/${RUN_TAG}_critic_train${N_TRAIN_EPISODES}_seed2"

mkdir -p runs/_logs
LOG="runs/_logs/run_${RUN_TAG}.log"
exec > >(tee -a "$LOG") 2>&1

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$ROOT/.libero_configs/vanilla}"
export PYTHONPATH="$ROOT/src:$ROOT/third_party/lerobot/src:$ROOT/third_party/LIBERO${PYTHONPATH:+:$PYTHONPATH}"

echo "[run] root: $ROOT"
echo "[run] python: $PY"
date
nvidia-smi || true

run_if_missing() {
  local marker="$1"
  shift
  if [ -e "$marker" ]; then
    echo "[skip] $marker already exists"
  else
    echo "[run] $*"
    "$@"
  fi
}

run_if_missing "$BASELINE_DIR/eval_info.json" \
  "$PY" scripts/eval_policy.py \
    --policy-path checkpoints/smolvla_libero \
    --output-dir "$BASELINE_DIR" \
    --env-type libero \
    --task "$TASK" \
    --task-ids "$TASK_IDS" \
    --n-episodes "$N_EVAL_EPISODES" \
    --seed "$BASE_SEED" \
    --device cuda \
    --max-videos "$MAX_VIDEOS"

run_if_missing "$ROLLOUT_DIR/summary.json" \
  "$PY" scripts/collect_rollouts.py \
    --policy-path checkpoints/smolvla_libero \
    --output-dir "$ROLLOUT_DIR" \
    --env-type libero \
    --task "$TASK" \
    --task-ids "$TASK_IDS" \
    --n-episodes "$N_TRAIN_EPISODES" \
    --seed "$TRAIN_SEED" \
    --device cuda

for seed in 0 1 2; do
  critic_dir_var="CRITIC${seed}_DIR"
  critic_dir="${!critic_dir_var}"
  run_if_missing "$critic_dir/critic.pt" \
    "$PY" scripts/train_critic.py \
      --data-dir "$ROLLOUT_DIR" \
      --output-dir "$critic_dir" \
      --action-horizon 50 \
      --hidden-dim 512 \
      --depth 3 \
      --epochs 30 \
      --seed "$seed" \
      --device cuda
done

run_if_missing "$QGF_DIR/eval_info.json" \
  "$PY" scripts/eval_policy.py \
    --policy-path checkpoints/smolvla_libero \
    --output-dir "$QGF_DIR" \
    --env-type libero \
    --task "$TASK" \
    --task-ids "$TASK_IDS" \
    --n-episodes "$N_EVAL_EPISODES" \
    --seed "$BASE_SEED" \
    --device cuda \
    --max-videos "$MAX_VIDEOS" \
    --critic-paths \
      "$CRITIC0_DIR/critic.pt" \
      "$CRITIC1_DIR/critic.pt" \
      "$CRITIC2_DIR/critic.pt" \
    --qgf-beta "$BETA" \
    --qgf-grad-clip-norm "$GRAD_CLIP" \
    --qgf-uncertainty-scale "$UNCERTAINTY_SCALE" \
    --qgf-min-gate "$MIN_GATE"

echo "[summary]"
"$PY" - <<PY
import json
from pathlib import Path

for name, path in [
    ("baseline", Path("$BASELINE_DIR") / "eval_info.json"),
    ("qgf_k3_gate", Path("$QGF_DIR") / "eval_info.json"),
]:
    data = json.loads(path.read_text())
    overall = data.get("overall", {})
    pc_success = overall.get("pc_success")
    episodes = overall.get("n_episodes")
    successes = None
    if pc_success is not None and episodes is not None:
        successes = round(float(pc_success) * int(episodes) / 100.0)
    print(
        name,
        "pc_success=", pc_success,
        "successes=", successes,
        "episodes=", episodes,
        "path=", path,
    )
PY

echo "[run] done"
