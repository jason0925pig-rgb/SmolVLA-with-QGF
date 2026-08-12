#!/usr/bin/env bash

# Source this profile on the company Jetson AGX Orin. It deliberately keeps
# the policy server on loopback, so inference traffic never leaves the robot.
SMOLVLA_ORIN_ROOT="${SMOLVLA_ORIN_ROOT:-/home/nvidia/work/telop}"
SMOLVLA_PROJECT_ROOT="${SMOLVLA_PROJECT_ROOT:-${SMOLVLA_ORIN_ROOT}/SmolVLA-with-QGF}"
TELEOP_PROJECT_ROOT="${TELEOP_PROJECT_ROOT:-${SMOLVLA_ORIN_ROOT}/One-Arm-Teleoperation}"
SMOLVLA_ORIN_VENV="${SMOLVLA_ORIN_VENV:-${SMOLVLA_ORIN_ROOT}/venvs/smolvla-orin}"
SMOLVLA_ORIN_BUNDLE="${SMOLVLA_ORIN_BUNDLE:-${SMOLVLA_ORIN_ROOT}/models/smolvla_onearm_20k_20260805}"
SMOLVLA_ORIN_CUSPARSELT="${SMOLVLA_ORIN_CUSPARSELT:-${SMOLVLA_ORIN_VENV}/opt/libcusparse_lt-linux-sbsa-0.5.2.1-archive/lib}"

export SMOLVLA_PHYSICAL_GPU=0
export SMOLVLA_PROJECT_ROOT
export TELEOP_PROJECT_ROOT
export SMOLVLA_SERVER_HOST=127.0.0.1
export SMOLVLA_SERVER_PORT="${SMOLVLA_SERVER_PORT:-8080}"
export SMOLVLA_SERVER_ADDRESS="127.0.0.1:${SMOLVLA_SERVER_PORT}"
export SMOLVLA_VENV="${SMOLVLA_ORIN_VENV}"
export SMOLVLA_CLIENT_VENV="${SMOLVLA_ORIN_VENV}"
export SMOLVLA_SERVER_MODEL_PATH="${SMOLVLA_ORIN_BUNDLE}/checkpoint"
export SMOLVLA_CACHE_ROOT="${SMOLVLA_ORIN_ROOT}/cache/smolvla"
export SMOLVLA_TMP_ROOT="${SMOLVLA_ORIN_ROOT}/tmp/smolvla"
# Policy action production/consumption runs at 30 Hz and uses the latest two
# 30 FPS RGB observations. The low-level JAKA servo loop remains at 125 Hz.
export SMOLVLA_FPS="${SMOLVLA_FPS:-30}"
# The deployed SmolVLA checkpoint remains stored in FP32, while inference uses
# CUDA AMP FP16 Tensor Core kernels.  Set this to fp32 for an exact fallback.
export SMOLVLA_INFERENCE_DTYPE="${SMOLVLA_INFERENCE_DTYPE:-fp16}"

# The checkpoint produces 50 actions. Orin warm inference measured
# In FP16 AMP, a warm 50-action chunk measured about 0.92 s. At 30 Hz, one
# chunk covers 1.67 s; the asynchronous client requests a replacement early.
export SMOLVLA_ACTIONS_PER_CHUNK="${SMOLVLA_ACTIONS_PER_CHUNK:-50}"
# Request a replacement immediately after the first of 50 actions executes.
# The subsequent response is timestep-aligned: already executed actions are
# dropped; future and overlapping actions replace the remaining queue.
export SMOLVLA_CHUNK_SIZE_THRESHOLD="${SMOLVLA_CHUNK_SIZE_THRESHOLD:-0.98}"
export SMOLVLA_VIDEO_BACKEND="${SMOLVLA_VIDEO_BACKEND:-pyav}"
export SMOLVLA_STATE_TIMEOUT_SECONDS="${SMOLVLA_STATE_TIMEOUT_SECONDS:-1.0}"
export SMOLVLA_CAMERA_TIMEOUT_SECONDS="${SMOLVLA_CAMERA_TIMEOUT_SECONDS:-1.0}"

export LD_LIBRARY_PATH="${SMOLVLA_ORIN_CUSPARSELT}:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HOME="${SMOLVLA_CACHE_ROOT}/huggingface"
export TORCH_HOME="${SMOLVLA_CACHE_ROOT}/torch"
export TMPDIR="${SMOLVLA_TMP_ROOT}"

mkdir -p "${SMOLVLA_CACHE_ROOT}" "${SMOLVLA_TMP_ROOT}"
