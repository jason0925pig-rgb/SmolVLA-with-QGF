# 4070 LIBERO Agent Runbook

This document is written for an agent or researcher setting up the experiment machine from a clean checkout.

Assumptions:

- GPU: NVIDIA RTX 4070 with 8GB VRAM.
- OS: Linux is strongly preferred for LIBERO simulation.
- Python: start with Python 3.10 or 3.11 unless upstream docs for the pinned checkout require otherwise.
- First benchmark: LIBERO only.
- First model: `lerobot/smolvla_base`.

## 0. Clone This Project

```bash
git clone <THIS_REPO_URL> guided-action-flow
cd guided-action-flow
```

If the repo is already cloned:

```bash
git pull
```

## 1. Clone Clean Upstream Repos

```bash
bash scripts/bootstrap_third_party.sh
```

Expected result:

```text
third_party/lerobot
third_party/LIBERO
```

Record versions immediately:

```bash
git -C third_party/lerobot rev-parse HEAD
git -C third_party/LIBERO rev-parse HEAD
```

Paste the commit hashes into `docs/setup.md` once the first run works.

## 2. Create The Python Environment

Preferred route for the first run:

```bash
conda create -n gaf-libero python=3.10 -y
conda activate gaf-libero
python -m pip install --upgrade pip setuptools wheel
```

Install this project:

```bash
pip install -e ".[dev,torch]"
```

Install LeRobot from the clean upstream checkout. Inspect `third_party/lerobot/README.md` and official docs first. The expected shape is:

```bash
cd third_party/lerobot
pip install -e ".[smolvla,libero]"
cd ../..
```

If the `smolvla` or `libero` extras are not available in the pinned checkout, inspect:

```bash
python - <<'PY'
try:
    import tomllib
except ModuleNotFoundError:
    raise SystemExit("Use Python 3.11+, or run: pip install tomli")
from pathlib import Path
pyproject = tomllib.loads(Path("third_party/lerobot/pyproject.toml").read_text())
print(pyproject.get("project", {}).get("optional-dependencies", {}).keys())
PY
```

Then install the correct extras from that checkout.

## 3. Do Not Mix Incompatible LIBERO Environments Blindly

The original LIBERO repo is kept under `third_party/LIBERO` for clean reference and fallback adapter work.

Before installing original LIBERO directly, inspect its current install instructions:

```bash
sed -n '1,220p' third_party/LIBERO/README.md
```

If it requires an older Python/Torch/CUDA stack, create a separate environment. Do not force those packages into the SmolVLA/LeRobot environment.

## 4. Download SmolVLA

Use the Hugging Face cache or a local checkpoint directory. This command downloads the model snapshot without running inference:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="lerobot/smolvla_base",
    local_dir="checkpoints/smolvla_base",
    local_dir_use_symlinks=False,
)
print(path)
PY
```

If `huggingface_hub` is missing:

```bash
pip install huggingface_hub
```

Do not commit downloaded model files.

## 5. Verify Imports

Run:

```bash
python - <<'PY'
import guided_action_flow
print("guided_action_flow", guided_action_flow.__version__)

try:
    import lerobot
    print("lerobot import ok")
except Exception as exc:
    print("lerobot import failed:", repr(exc))

try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
except Exception as exc:
    print("torch import failed:", repr(exc))
PY
```

## 6. Inspect SmolVLA Action Sampling Before Implementing QGF

Find the SmolVLA policy and sampler implementation in the pinned LeRobot checkout:

```bash
find third_party/lerobot -iname '*smolvla*' -o -iname '*flow*'
```

The agent must identify:

- how observations are normalized;
- how action chunks are represented;
- action chunk shape: `[horizon, action_dim]` or `[batch, horizon, action_dim]`;
- flow sampling time convention;
- where velocity is predicted;
- whether sampling uses Euler steps or another solver;
- how actions are unnormalized before environment stepping.

Only after this inspection should `src/guided_action_flow/policies/smolvla.py` and `src/guided_action_flow/guidance/qgf.py` be wired to the real API.

## 7. Bring Up LIBERO

Use LeRobot's LIBERO docs first. The adapter target is:

```text
src/guided_action_flow/benchmarks/libero.py
```

The agent must inspect the pinned benchmark wrapper and record:

- suite names;
- task names;
- observation keys;
- camera/image keys;
- proprio keys;
- action dimension;
- horizon;
- success key in `info`;
- whether environment reward is sparse or dense.

Update:

```text
configs/benchmarks/libero.yaml
configs/policy/smolvla_base.yaml
```

with verified values only.

## 8. First Smoke Test

The first smoke test can use a dummy policy or one base SmolVLA action chunk and
only one LIBERO task to verify environment wiring. Reported QGF experiments
should use real SmolVLA rollouts, not dummy actions.

Required output:

```text
runs/smoke_libero/
├── config.yaml
├── metrics.jsonl
└── notes.md
```

Do not start critic training until this works.

## 9. First Real Experiment Ladder

Run in this order:

1. Base SmolVLA deterministic evaluation.
2. Base SmolVLA rollout collection.
3. Action-chunk critic training from collected rollouts.
4. In-loop vanilla QGF.
5. Critic ensemble and adaptive disagreement gate.
6. Multi-task or task-description conditioned QGF.

Abort QGF if the critic cannot rank held-out action chunks better than random.

## 10. Files The Agent Should Modify First

Start here:

```text
src/guided_action_flow/benchmarks/libero.py
src/guided_action_flow/policies/smolvla.py
scripts/eval_policy.py
configs/benchmarks/libero.yaml
configs/policy/smolvla_base.yaml
```

Do not begin by editing `QGuidanceConfig` or the critic architecture. First make base policy evaluation reproducible.
