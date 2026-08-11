#!/usr/bin/env bash
set -Eeuo pipefail

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="${1:?usage: test_smolvla_bundle.sh BUNDLE_ROOT DATASET_ROOT}"
DATASET_ROOT="${2:?usage: test_smolvla_bundle.sh BUNDLE_ROOT DATASET_ROOT}"
VENV="${SMOLVLA_VENV:-/ssd/hanbo/TNNLS_2026/envs/lerobot-v0.4.4}"
PHYSICAL_GPU="${SMOLVLA_PHYSICAL_GPU:-4}"
PORT="${SMOLVLA_SMOKE_PORT:-18080}"
ACTIONS_PER_CHUNK="${SMOLVLA_ACTIONS_PER_CHUNK:-10}"
SMOKE_TIMEOUT="${SMOLVLA_SMOKE_TIMEOUT:-120}"
SMOKE_REQUESTS="${SMOLVLA_SMOKE_REQUESTS:-1}"
LOG_FILE="${TMPDIR:-/tmp}/smolvla_policy_server_smoke_${USER:-user}.log"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -INT "${SERVER_PID}" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      kill -TERM "${SERVER_PID}" 2>/dev/null || true
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

[[ -x "${VENV}/bin/python" ]] || { echo "ERROR: missing ${VENV}/bin/python" >&2; exit 2; }
[[ -x "${BUNDLE_ROOT}/start_policy_server.sh" ]] || {
  echo "ERROR: bundle launcher is not executable" >&2
  exit 2
}
[[ -f "${BUNDLE_ROOT}/checkpoint/config.json" ]] || {
  echo "ERROR: bundle checkpoint is incomplete" >&2
  exit 2
}

cd "${BUNDLE_ROOT}"
SMOLVLA_PHYSICAL_GPU="${PHYSICAL_GPU}" \
SMOLVLA_SERVER_HOST=127.0.0.1 \
SMOLVLA_SERVER_PORT="${PORT}" \
SMOLVLA_VENV="${VENV}" \
  ./start_policy_server.sh >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 120); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ERROR: policy server exited during startup" >&2
    tail -n 80 "${LOG_FILE}" >&2 || true
    exit 3
  fi
  if "${VENV}/bin/python" - "${PORT}" 2>/dev/null <<'PY'
import socket
import sys
with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=0.2):
    pass
PY
  then
    break
  fi
  sleep 0.25
done

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" \
  "${VENV}/bin/python" "${TOOLS_DIR}/smoke_smolvla_policy_server.py" \
    --server "127.0.0.1:${PORT}" \
    --checkpoint checkpoint \
    --dataset-root "${DATASET_ROOT}" \
    --repo-id local/onearm_tele \
    --sample-index 0 \
    --actions-per-chunk "${ACTIONS_PER_CHUNK}" \
    --requests "${SMOKE_REQUESTS}" \
    --video-backend "${SMOLVLA_VIDEO_BACKEND:-torchcodec}" \
    --timeout "${SMOKE_TIMEOUT}"

cleanup
SERVER_PID=""
trap - EXIT INT TERM
echo "SMOLVLA_DEPLOYMENT_BUNDLE_TEST_OK physical_gpu=${PHYSICAL_GPU}"
