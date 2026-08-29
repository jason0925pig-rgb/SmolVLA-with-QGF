#!/bin/bash
# Offline install of lerobot 0.4.4 + qgf into the pinned py310 env on the 4090.
#
# Why --no-deps for lerobot: lerobot 0.4.4 declares torchvision>=0.21, but the
# handoff (section 7) pins torchvision 0.20.1 to match driver 535 / cu121.  The
# feature extractor only touches av + torch + SmolVLAPolicy, so the declared
# torchvision floor is not exercised.  Installing every wheel in the offline
# wheelhouse with --no-deps gives the same closure without letting the resolver
# try to upgrade torch/torchvision (which are NOT in the wheelhouse on purpose).
set -eu
E=/opt/qgf_real_robot/envs/visual_iql_py310
WH=/opt/qgf_real_robot/wheelhouse
REPO=/opt/qgf_real_robot/repos/SmolVLA-with-QGF

echo "=== torch BEFORE (must not change) ==="
"$E/bin/python" -c "import torch,torchvision;print(torch.__version__, torchvision.__version__)"
BEFORE=$("$E/bin/python" -c "import torch,torchvision;print(torch.__version__,torchvision.__version__)")

echo
echo "=== installing every wheel in the wheelhouse, --no-deps ==="
mapfile -t WHEELS < <(ls "$WH"/*.whl 2>/dev/null | grep -vE '/(torch|torchvision|nvidia_|triton)[-_]')
echo "wheels to install: ${#WHEELS[@]}"
"$E/bin/pip" install -q --no-index --no-deps --disable-pip-version-check "${WHEELS[@]}"

echo "=== sdists that need a local build ==="
for s in "$WH"/*.tar.gz; do
  [ -e "$s" ] || continue
  echo "  building $(basename "$s")"
  "$E/bin/pip" install -q --no-index --no-deps --no-build-isolation --disable-pip-version-check "$s" \
    || echo "  WARN: $(basename "$s") failed to build (may be optional)"
done

echo
echo "=== installing the qgf package itself (editable, no deps) ==="
"$E/bin/pip" install -q --no-index --no-deps --no-build-isolation -e "$REPO/qgf"

echo
echo "=== torch AFTER (must be identical) ==="
AFTER=$("$E/bin/python" -c "import torch,torchvision;print(torch.__version__,torchvision.__version__)")
echo "$AFTER"
if [ "$BEFORE" != "$AFTER" ]; then
  echo "FATAL: torch/torchvision changed during install: '$BEFORE' -> '$AFTER'"
  exit 1
fi

echo
echo "=== import smoke test (this is the real gate) ==="
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 "$E/bin/python" - <<'PY'
import importlib.metadata as md
import torch

print("torch      ", torch.__version__, torch.version.cuda)
print("cuda avail ", torch.cuda.is_available(), "count", torch.cuda.device_count())
print("gpu        ", torch.cuda.get_device_name(0))
for p in ("lerobot", "transformers", "av", "pyarrow", "num2words", "accelerate", "safetensors"):
    try:
        print(f"{p:14s}", md.version(p))
    except Exception as exc:
        print(f"{p:14s} MISSING ({exc})")

import av  # noqa: F401
print("import av                                    OK")
import pyarrow.parquet  # noqa: F401
print("import pyarrow.parquet                       OK")
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: F401
print("import SmolVLAPolicy                         OK")
from guided_action_flow.critics.visual_transformer_critic import (  # noqa: F401
    VisualTransformerCriticConfig,
)
print("import VisualTransformerCriticConfig         OK")
from guided_action_flow.critics.checkpoint import load_action_chunk_critic  # noqa: F401
print("import load_action_chunk_critic              OK")
PY
echo
echo "INSTALL OK"
