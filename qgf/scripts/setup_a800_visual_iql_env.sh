#!/usr/bin/env bash
# Set up the isolated environment used for real-robot visual IQL on a CUDA GPU.
# Usage: bash qgf/scripts/setup_a800_visual_iql_env.sh [venv_path]
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
venv_path="${1:-${repo_root}/.venv}"
python_bin="${venv_path}/bin/python"

if [[ ! -x "${python_bin}" ]]; then
  python3 -m venv "${venv_path}"
fi

"${python_bin}" -m pip install --upgrade pip setuptools wheel
"${python_bin}" -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
"${python_bin}" -m pip install -e "${repo_root}/qgf[torch]" \
  "lerobot[smolvla]" av pyarrow

"${python_bin}" - <<'PY'
import av
import lerobot
import pyarrow
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

print(
    "ENV_READY",
    f"torch={torch.__version__}",
    f"cuda_available={torch.cuda.is_available()}",
    f"gpu0={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
)
PY
