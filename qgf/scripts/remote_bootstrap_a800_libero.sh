#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_PREFIX="${ENV_PREFIX:-$ROOT/.venv-a800-py312}"
PYTHON312_ROOT="${PYTHON312_ROOT:-$ROOT/.python312}"
PYTHON312_TARBALL="${PYTHON312_TARBALL:-/tmp/py312.tar.gz}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
PYPI_EXTRA_INDEX_URL="${PYPI_EXTRA_INDEX_URL:-https://mirrors.cloud.tencent.com/pypi/simple/}"
PYPI_TRUSTED_HOST="${PYPI_TRUSTED_HOST:-mirrors.aliyun.com}"
TORCH_WHEEL_BASE_URL="${TORCH_WHEEL_BASE_URL:-https://mirrors.aliyun.com/pytorch-wheels/cu126}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1+cu126}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1+cu126}"
TORCH_WHEEL_PLATFORM="${TORCH_WHEEL_PLATFORM:-manylinux_2_28_x86_64}"
TRITON_VERSION="${TRITON_VERSION:-3.3.1}"
LEROBOT_COMMIT="${LEROBOT_COMMIT:-6a788fbdb02cabfae60f7408636945df0b1eafa0}"
LIBERO_COMMIT="${LIBERO_COMMIT:-8f1084e3132a39270c3a13ebe37270a43ece2a01}"

mkdir -p runs/_logs
exec > >(tee -a "runs/_logs/remote_bootstrap_a800_libero.log") 2>&1
export PIP_EXTRA_INDEX_URL="$PYPI_EXTRA_INDEX_URL"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "[bootstrap] root: $ROOT"
date

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "[bootstrap] nvidia-smi not found; continuing so Python setup can still run."
fi

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  PY312="$PYTHON312_ROOT/python/bin/python3.12"
  if [ ! -x "$PY312" ] && [ -r "$PYTHON312_TARBALL" ]; then
    echo "[bootstrap] extracting Python 3.12 from $PYTHON312_TARBALL"
    mkdir -p "$PYTHON312_ROOT"
    tar -xzf "$PYTHON312_TARBALL" -C "$PYTHON312_ROOT"
  fi

  if [ -x "$PY312" ]; then
    "$PY312" -m venv "$ENV_PREFIX"
  elif command -v micromamba >/dev/null 2>&1; then
    micromamba create -y -p "$ENV_PREFIX" python=3.12 pip
  elif command -v mamba >/dev/null 2>&1; then
    mamba create -y -p "$ENV_PREFIX" python=3.12 pip
  elif command -v conda >/dev/null 2>&1; then
    conda create -y -p "$ENV_PREFIX" python=3.12 pip
  elif command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv "$ENV_PREFIX"
  else
    echo "[bootstrap] Python 3.12 is required by LeRobot, but no Python 3.12 runtime was found." >&2
    exit 1
  fi
fi

PY="$ENV_PREFIX/bin/python"
"$PY" --version
"$PY" -m pip install \
  --default-timeout 120 \
  --retries 10 \
  -i "$PYPI_INDEX_URL" \
  --trusted-host "$PYPI_TRUSTED_HOST" \
  --upgrade pip setuptools wheel

PY_TAG="$("$PY" - <<'PY'
import sys
print(f"cp{sys.version_info.major}{sys.version_info.minor}")
PY
)"

echo "[bootstrap] installing PyTorch $TORCH_VERSION from $TORCH_WHEEL_BASE_URL"
"$PY" -m pip install \
  --default-timeout 120 \
  --retries 10 \
  -i "$PYPI_INDEX_URL" \
  --trusted-host "$PYPI_TRUSTED_HOST" \
  filelock fsspec jinja2 networkx numpy pillow sympy typing-extensions \
  "triton==$TRITON_VERSION"

"$PY" -m pip install \
  --default-timeout 120 \
  --retries 10 \
  -i "$PYPI_INDEX_URL" \
  --trusted-host "$PYPI_TRUSTED_HOST" \
  --no-deps \
  nvidia-cudnn-cu12==9.1.0.70 nvidia-nccl-cu12==2.21.5

mkdir -p .wheelhouse
torch_wheel="torch-${TORCH_VERSION}-${PY_TAG}-${PY_TAG}-${TORCH_WHEEL_PLATFORM}.whl"
vision_wheel="torchvision-${TORCHVISION_VERSION}-${PY_TAG}-${PY_TAG}-${TORCH_WHEEL_PLATFORM}.whl"
torch_url="$TORCH_WHEEL_BASE_URL/${torch_wheel/+/%2B}"
vision_url="$TORCH_WHEEL_BASE_URL/${vision_wheel/+/%2B}"

download_wheel() {
  local url="$1"
  local dest="$2"
  if [ -s "$dest" ]; then
    if "$PY" -m zipfile -t "$dest" >/dev/null 2>&1; then
      echo "[bootstrap] reusing existing wheel $dest"
      return
    fi
    echo "[bootstrap] removing incomplete wheel $dest"
    rm -f "$dest"
  fi
  echo "[bootstrap] downloading $url"
  curl -fL \
    --retry 10 \
    --retry-delay 5 \
    --connect-timeout 30 \
    --speed-limit 1024 \
    --speed-time 120 \
    -o "$dest" \
    "$url"
}

download_wheel "$torch_url" ".wheelhouse/$torch_wheel"
download_wheel "$vision_url" ".wheelhouse/$vision_wheel"

"$PY" -m pip install \
  --no-deps \
  ".wheelhouse/$torch_wheel" \
  ".wheelhouse/$vision_wheel"

echo "[bootstrap] installing guided-action-flow"
"$PY" -m pip install \
  --default-timeout 120 \
  --retries 10 \
  -i "$PYPI_INDEX_URL" \
  --trusted-host "$PYPI_TRUSTED_HOST" \
  -e ".[dev]"

echo "[bootstrap] cloning/pinning third-party repos"
LEROBOT_COMMIT="$LEROBOT_COMMIT" \
LIBERO_COMMIT="$LIBERO_COMMIT" \
bash scripts/bootstrap_third_party.sh
if [ -d third_party/lerobot/.git ]; then
  git -C third_party/lerobot checkout "$LEROBOT_COMMIT"
fi
if [ -d third_party/LIBERO/.git ]; then
  git -C third_party/LIBERO checkout "$LIBERO_COMMIT"
fi

echo "[bootstrap] installing LeRobot extras"
cat > .a800_lerobot_constraints.txt <<EOF
torch==${TORCH_VERSION}
torchvision==${TORCHVISION_VERSION}
torchcodec<0.11
numpy<2.3
setuptools<81
EOF
"$PY" -m pip install \
  --default-timeout 120 \
  --retries 10 \
  -i "$PYPI_INDEX_URL" \
  --trusted-host "$PYPI_TRUSTED_HOST" \
  --constraint .a800_lerobot_constraints.txt \
  -e "third_party/lerobot[smolvla,libero]"

echo "[bootstrap] installing LIBERO editable package if setup exists"
if [ -f third_party/LIBERO/setup.py ] || [ -f third_party/LIBERO/pyproject.toml ]; then
  "$PY" -m pip install \
    --default-timeout 120 \
    --retries 10 \
    -i "$PYPI_INDEX_URL" \
    --trusted-host "$PYPI_TRUSTED_HOST" \
    -e third_party/LIBERO
fi

echo "[bootstrap] linking local LIBERO assets into installed package when needed"
"$PY" - <<'PY'
from pathlib import Path
import importlib.util
import os

root = Path.cwd()
source_assets = root / "third_party" / "LIBERO" / "libero" / "libero" / "assets"
spec = importlib.util.find_spec("libero.libero")
if spec and spec.origin and source_assets.exists():
    package_dir = Path(spec.origin).parent
    target_assets = package_dir / "assets"
    if not target_assets.exists():
        os.symlink(source_assets, target_assets, target_is_directory=True)
        print(f"linked {target_assets} -> {source_assets}")
    else:
        print(f"assets already present at {target_assets}")
else:
    print("skipping assets link; source assets or libero package not found")
PY

echo "[bootstrap] writing vanilla LIBERO config"
mkdir -p .libero_configs/vanilla
cat > .libero_configs/vanilla/config.yaml <<EOF
assets: $ROOT/third_party/LIBERO/libero/libero/assets
bddl_files: $ROOT/third_party/LIBERO/libero/libero/bddl_files
benchmark_root: $ROOT/third_party/LIBERO/libero/libero
datasets: $ROOT/third_party/LIBERO/libero/datasets
init_states: $ROOT/third_party/LIBERO/libero/libero/init_files
EOF

echo "[bootstrap] downloading official SmolVLA LIBERO checkpoint"
"$PY" scripts/download_models.py \
  --repo-id lerobot/smolvla_libero \
  --local-dir checkpoints/smolvla_libero

echo "[bootstrap] smoke imports"
MUJOCO_GL=egl \
WANDB_MODE=disabled \
LIBERO_CONFIG_PATH="$ROOT/.libero_configs/vanilla" \
PYTHONPATH="$ROOT/src:$ROOT/third_party/lerobot/src:$ROOT/third_party/LIBERO${PYTHONPATH:+:$PYTHONPATH}" \
"$PY" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
import guided_action_flow
print("guided_action_flow import ok")
PY

echo "[bootstrap] done"
