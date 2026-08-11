#!/usr/bin/env bash
set -Eeuo pipefail

# Foreground launcher for a LeRobot asynchronous policy server.  The physical
# GPU is deliberately explicit; inside Python it appears as logical cuda:0.
PHYSICAL_GPU="${SMOLVLA_PHYSICAL_GPU:-4}"
HOST="${SMOLVLA_SERVER_HOST:-0.0.0.0}"
PORT="${SMOLVLA_SERVER_PORT:-8080}"
FPS="${SMOLVLA_FPS:-30}"
INFERENCE_LATENCY="${SMOLVLA_INFERENCE_LATENCY:-0.0}"
OBS_QUEUE_TIMEOUT="${SMOLVLA_OBS_QUEUE_TIMEOUT:-1.0}"
VENV="${SMOLVLA_VENV:-/ssd/hanbo/TNNLS_2026/envs/lerobot-v0.4.4}"
CACHE_ROOT="${SMOLVLA_CACHE_ROOT:-/ssd/hanbo/TNNLS_2026/cache}"
TMP_ROOT="${SMOLVLA_TMP_ROOT:-/ssd/hanbo/TNNLS_2026/tmp}"

[[ "${PHYSICAL_GPU}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: SMOLVLA_PHYSICAL_GPU must be one integer." >&2
  exit 2
}
[[ -x "${VENV}/bin/python" ]] || {
  echo "ERROR: LeRobot Python is missing: ${VENV}/bin/python" >&2
  exit 2
}
mkdir -p "${CACHE_ROOT}" "${TMP_ROOT}"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export MODELSCOPE_CACHE="${CACHE_ROOT}/modelscope"
export TMPDIR="${TMP_ROOT}"

echo "SMOLVLA_POLICY_SERVER physical_gpu=${PHYSICAL_GPU} logical_device=cuda:0 host=${HOST}:${PORT} fps=${FPS}"
echo "The model path is supplied by the trusted robot client and must exist on this server."
exec "${VENV}/bin/python" -m lerobot.async_inference.policy_server \
  --host="${HOST}" \
  --port="${PORT}" \
  --fps="${FPS}" \
  --inference_latency="${INFERENCE_LATENCY}" \
  --obs_queue_timeout="${OBS_QUEUE_TIMEOUT}"
