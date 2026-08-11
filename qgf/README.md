# Guided Action Flow

Guided Action Flow is a research codebase for testing whether test-time
Q-guided flow sampling can improve a frozen flow-matching VLA policy on robot
manipulation benchmarks.

Current concrete target:

```text
base policy: official LeRobot SmolVLA LIBERO checkpoints
main benchmark: LIBERO
extended benchmarks: LIBERO-plus and LIBERO-PRO
method: in-loop QGF over SmolVLA action chunks
current stage: reproducible baseline + first QGF ablations, not a final claim
```

This repository is intentionally project glue. It does not vendor LeRobot,
SmolVLA, LIBERO, LIBERO-plus, or LIBERO-PRO source code into
`src/guided_action_flow`. External repositories live under `third_party/`, model
weights live under `checkpoints/`, rollout artifacts live under `runs/`, and all
large generated files are gitignored.

## Research Question

The working question is:

```text
Can a learned action-chunk critic guide SmolVLA's flow sampler toward
higher-return LIBERO actions without retraining the whole VLA?
```

The ablation path is deliberately narrow:

1. Official SmolVLA baseline.
2. Official SmolVLA plus in-loop vanilla QGF.
3. SmolVLA plus critic ensemble and adaptive disagreement gate.
4. Multi-task or task-description conditioned critic.
5. OOD-aware QGF and real robot deployment.

Action reranking is not part of the current project track. The implemented QGF
path modifies SmolVLA's denoising velocity inside the action sampler.

## Repository Layout

```text
configs/                         lightweight project configs
docs/                            architecture, setup, and experiment notes
scripts/                         experiment entry points
src/guided_action_flow/
  benchmarks/                    benchmark adapter interfaces
  critics/                       action-chunk critic and checkpoint loading
  guidance/                      QGF update rule
  policies/                      SmolVLA/LeRobot wrappers and QGF hook
  rewards/                       reward interfaces and LIBERO wrappers
  training/                      rollout dataset and task feature utilities
  evaluation/                    rollout and metrics utilities
tests/                           unit tests for local project glue
third_party/                     external source checkouts, gitignored
checkpoints/                     model and critic checkpoints, gitignored
runs/                            rollouts, metrics, videos, logs, gitignored
data/                            benchmark assets and generated data, gitignored
```

Important project notes:

- [docs/qgf_smolvla_design.md](docs/qgf_smolvla_design.md) records the verified
  SmolVLA flow convention, QGF sign, current ablations, and conclusions.
- [docs/architecture.md](docs/architecture.md) describes the intended module
  boundaries.
- [docs/setup.md](docs/setup.md) and
  [docs/agent_4070_libero_runbook.md](docs/agent_4070_libero_runbook.md)
  contain operational setup notes.

## Hardware Used So Far

The current runs were produced on a laptop RTX 4070 with 8188 MiB VRAM.

Observed GPU notes:

```text
official SmolVLA 0.45B LIBERO eval: fits comfortably on 8GB VRAM
QGF eval with K=3 critics: fits comfortably on 8GB VRAM
SmolVLA 2.25B smoke forward pass: peak around 5.5 GiB nvidia-smi memory
```

The 2.25B smoke was only a local feasibility check. Reported LIBERO results use
official `lerobot/smolvla_libero` or `lerobot/smolvla_libero_plus` 0.45B
checkpoints.

## Environment Setup

Linux is strongly preferred because LIBERO depends on MuJoCo/robosuite.

Create a Python environment. The current local environment uses Python 3.12, but
Python 3.10 to 3.12 is supported by this package metadata:

```bash
conda create -n gaf-libero python=3.12 -y
conda activate gaf-libero
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,torch]"
```

Clone upstream repositories:

```bash
bash scripts/bootstrap_third_party.sh
git clone https://github.com/sylvestf/LIBERO-plus.git third_party/LIBERO-plus
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git third_party/LIBERO-PRO
```

Pinned commits used by the current local runs:

```text
lerobot: 6a788fbdb02cabfae60f7408636945df0b1eafa0
LIBERO: 8f1084e3132a39270c3a13ebe37270a43ece2a01
LIBERO-plus: 4976dc30028e805ff8094b55501d532c48fec182
LIBERO-PRO: eafdb809426b13153aa1e4c42d6601844217dfec
```

Install LeRobot from the checkout after inspecting the pinned checkout's own
instructions:

```bash
cd third_party/lerobot
python -m pip install -e ".[smolvla,libero]"
cd ../..
```

If the selected LeRobot checkout changes its optional dependency names, inspect
`third_party/lerobot/pyproject.toml` and install the equivalent SmolVLA/LIBERO
extras from that checkout. Do not guess LeRobot APIs when modifying adapter
code; inspect the installed version or the local checkout first.

Use EGL for headless MuJoCo:

```bash
export MUJOCO_GL=egl
export WANDB_MODE=disabled
export PYTHONPATH="$PWD/src:$PWD/third_party/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
```

## Local LIBERO Configs

LeRobot LIBERO environments read `LIBERO_CONFIG_PATH`. This repo keeps local
config files under `.libero_configs/`, but those files contain machine-specific
absolute paths and are not committed.

Create these directories locally:

```bash
mkdir -p .libero_configs/vanilla .libero_configs/plus .libero_configs/pro
```

Example LIBERO-plus config:

```yaml
assets: /absolute/path/to/guided-action-flow/third_party/LIBERO-plus/libero/libero/assets
bddl_files: /absolute/path/to/guided-action-flow/third_party/LIBERO-plus/libero/libero/bddl_files
benchmark_root: /absolute/path/to/guided-action-flow/third_party/LIBERO-plus/libero/libero
datasets: /absolute/path/to/guided-action-flow/third_party/LIBERO-plus/libero/datasets
init_states: /absolute/path/to/guided-action-flow/third_party/LIBERO-plus/libero/libero/init_files
```

The vanilla and PRO configs use the same keys, pointed at the installed vanilla
LIBERO package or `third_party/LIBERO-PRO` respectively.

When running vanilla LIBERO:

```bash
export LIBERO_CONFIG_PATH="$PWD/.libero_configs/vanilla"
```

When running LIBERO-plus or LIBERO-PRO, switch `LIBERO_CONFIG_PATH` to the
matching local config and include the corresponding repository on `PYTHONPATH`.

## Download Official Checkpoints

Do not commit downloaded checkpoint files.

```bash
python scripts/download_models.py \
  --repo-id lerobot/smolvla_libero \
  --local-dir checkpoints/smolvla_libero

python scripts/download_models.py \
  --repo-id lerobot/smolvla_libero_plus \
  --local-dir checkpoints/smolvla_libero_plus

python scripts/download_models.py \
  --repo-id lerobot/smolvla_base \
  --local-dir checkpoints/smolvla_base
```

The current LIBERO QGF work uses SmolVLA checkpoints, not Pi0.5 checkpoints.

## Run Baselines

Vanilla LIBERO smoke:

```bash
PYTHON_BIN="$CONDA_PREFIX/bin/python" \
POLICY_PATH="$PWD/checkpoints/smolvla_libero" \
OUTPUT_DIR="$PWD/runs/smolvla_libero_smoke" \
TASK=libero_spatial \
TASK_IDS='[0]' \
N_EPISODES=1 \
bash scripts/eval_smolvla_libero.sh
```

LIBERO-plus smoke:

```bash
PYTHON_BIN="$CONDA_PREFIX/bin/python" \
POLICY_PATH="$PWD/checkpoints/smolvla_libero_plus" \
OUTPUT_DIR="$PWD/runs/smolvla_libero_plus_true_smoke" \
TASK=libero_spatial \
TASK_IDS='[0]' \
N_EPISODES=1 \
bash scripts/eval_smolvla_libero_plus.sh
```

LIBERO-PRO smoke with the vanilla LIBERO checkpoint:

```bash
PYTHON_BIN="$CONDA_PREFIX/bin/python" \
POLICY_PATH="$PWD/checkpoints/smolvla_libero" \
OUTPUT_DIR="$PWD/runs/smolvla_libero_pro_spatial_task_smoke" \
TASK=libero_spatial_task \
TASK_IDS='[0]' \
N_EPISODES=1 \
bash scripts/eval_smolvla_libero_pro.sh
```

The project-specific evaluator gives the same baseline path and also supports
QGF:

```bash
python scripts/eval_policy.py \
  --policy-path checkpoints/smolvla_libero \
  --output-dir runs/baseline_spatial3_ep50_seed3000 \
  --env-type libero \
  --task libero_spatial \
  --task-ids '[3]' \
  --n-episodes 50 \
  --seed 3000 \
  --device cuda \
  --max-videos 0
```

## Train A Single-Task Critic

Collect real SmolVLA rollouts. No dummy actions should be used for reported
runs:

```bash
python scripts/collect_rollouts.py \
  --policy-path checkpoints/smolvla_libero \
  --output-dir runs/qgf_single_task_spatial3_train50 \
  --env-type libero \
  --task libero_spatial \
  --task-ids '[3]' \
  --n-episodes 50 \
  --seed 2000 \
  --device cuda
```

Train the action-chunk critic:

```bash
python scripts/train_critic.py \
  --data-dir runs/qgf_single_task_spatial3_train50 \
  --output-dir runs/qgf_single_task_spatial3_critic_train50 \
  --action-horizon 50 \
  --hidden-dim 512 \
  --depth 3 \
  --epochs 20 \
  --seed 0 \
  --device cuda
```

Evaluate baseline and QGF:

```bash
python scripts/eval_policy.py \
  --policy-path checkpoints/smolvla_libero \
  --output-dir runs/qgf_single_task_spatial3_eval50_qgf_beta2_seed3000 \
  --env-type libero \
  --task libero_spatial \
  --task-ids '[3]' \
  --n-episodes 50 \
  --seed 3000 \
  --device cuda \
  --max-videos 0 \
  --critic-path runs/qgf_single_task_spatial3_critic_train50/critic.pt \
  --qgf-beta 2 \
  --qgf-grad-clip-norm 1.0
```

## Train A Task-Description Critic

The current transfer design uses frozen SmolVLA/VLM text hidden-state features
as the task description input. This is stronger than raw task id conditioning
because task id does not transfer across benchmark families.

Collect multiple tasks:

```bash
python scripts/collect_rollouts.py \
  --policy-path checkpoints/smolvla_libero \
  --output-dir runs/qgf_taskdesc_multitask_spatial0to4_train250 \
  --env-type libero \
  --task libero_spatial \
  --task-ids '[0, 1, 2, 3, 4]' \
  --n-episodes 50 \
  --seed 7200 \
  --device cuda
```

Precompute VLM text hidden features:

```bash
python scripts/precompute_task_features.py \
  --input-dir runs/qgf_taskdesc_multitask_spatial0to4_train250 \
  --output-dir runs/qgf_taskdesc_multitask_spatial0to4_train250_vlm_hidden \
  --policy-path checkpoints/smolvla_libero \
  --env-type libero \
  --task libero_spatial \
  --task-ids '[0, 1, 2, 3, 4]' \
  --feature-key task_vlm_hidden \
  --device cuda
```

Train a K=3 critic ensemble:

```bash
python scripts/train_critic.py \
  --data-dir runs/qgf_taskdesc_multitask_spatial0to4_train250_vlm_hidden \
  --output-dir runs/qgf_multitask_spatial0to4_critic_vlm_hidden_seed0 \
  --task-feature-source vlm_hidden \
  --task-feature-key task_vlm_hidden \
  --action-horizon 50 \
  --hidden-dim 512 \
  --depth 3 \
  --epochs 20 \
  --seed 0 \
  --device cuda
```

Repeat the same command with `--seed 1` and `--seed 2`, writing to separate
output directories.

Evaluate QGF with adaptive disagreement gate:

```bash
python scripts/eval_policy.py \
  --policy-path checkpoints/smolvla_libero \
  --output-dir runs/qgf_vlm_hidden_val_spatial_5_7_8_beta2_gate20_seed8000 \
  --env-type libero \
  --task libero_spatial \
  --task-ids '[5, 7, 8]' \
  --n-episodes 10 \
  --seed 8000 \
  --device cuda \
  --max-videos 0 \
  --critic-paths \
    runs/qgf_multitask_spatial0to4_critic_vlm_hidden_seed0/critic.pt \
    runs/qgf_multitask_spatial0to4_critic_vlm_hidden_seed1/critic.pt \
    runs/qgf_multitask_spatial0to4_critic_vlm_hidden_seed2/critic.pt \
  --qgf-beta 2 \
  --qgf-grad-clip-norm 1.0 \
  --qgf-uncertainty-scale 20 \
  --qgf-min-gate 0.1
```

## Current Experimental Progress

These are local results recorded under `runs/` and summarized in
[docs/qgf_smolvla_design.md](docs/qgf_smolvla_design.md). The artifacts are not
tracked by git; rerun the commands above or request the artifact bundle to
inspect exact `eval_info.json` files.

### Baseline Anchors

| Setting | Checkpoint | Budget | Success |
| --- | --- | --- | --- |
| LIBERO vanilla, 4 suites x 5 tasks x 5 episodes | `lerobot/smolvla_libero` | 100 episodes | 65/100 = 65.0% |
| LIBERO-plus spatial subset, 10 tasks x 5 episodes | `lerobot/smolvla_libero_plus` | 50 episodes | 39/50 = 78.0% |
| LIBERO-PRO zero-shot, 4 suites x 5 tasks x 5 episodes | `lerobot/smolvla_libero` | 100 episodes | 1/100 = 1.0% |

### Single-Task QGF Signal

| Setting | Task | Budget | Success |
| --- | --- | --- | --- |
| Baseline | `libero_spatial` task 3, seed 3000 | 50 episodes | 34/50 = 68.0% |
| K=3 critic ensemble + adaptive gate | same | 50 episodes | 41/50 = 82.0% |
| Baseline | `libero_spatial` task 3, seed 4000 | 50 episodes | 41/50 = 82.0% |
| K=3 critic ensemble + adaptive gate | same | 50 episodes | 43/50 = 86.0% |

This is the strongest positive signal so far, but it is still single-task.

### Task-Description Critic Transfer

The VLM-hidden task-description critic was trained on `libero_spatial` task ids
0 to 4 and evaluated on held-out validation tasks and `libero_object`.

Expanded validation:

| Setting | Budget | Success |
| --- | --- | --- |
| Baseline | 60 episodes | 32/60 = 53.3% |
| QGF beta=2, gate=20 | 60 episodes | 31/60 = 51.7% |
| QGF beta=3, gate=20 | 60 episodes | 29/60 = 48.3% |
| QGF beta=5, gate=20 | 60 episodes | 29/60 = 48.3% |

Held-out `libero_object` diagnostic:

| Setting | Budget | Success |
| --- | --- | --- |
| Baseline | 30 episodes | 18/30 = 60.0% |
| QGF beta=2, gate=20 | 30 episodes | 18/30 = 60.0% |

Current interpretation:

```text
QGF can change outcomes and produced a clear single-task gain, but the current
spatial-only VLM-hidden critic does not validate as a robust cross-family QGF
setting. The next clean step is multi-family critic data and validation, not
more tuning on held-out test tasks.
```

## Development Directions

Near-term:

1. Train a multi-family task-description critic using `libero_spatial`,
   `libero_object`, and later `libero_goal` rollouts.
2. Select QGF hyperparameters only on validation tasks, then report held-out test
   tasks once.
3. Extend the same baseline/QGF protocol to LIBERO-plus.
4. Build LIBERO-PRO rollouts and decide whether a PRO-specific checkpoint or
   critic is needed, because vanilla zero-shot performance is currently near
   zero.

Method directions:

1. OOD-aware QGF using critic ensemble disagreement, action-feature distance, or
   task-description distance.
2. Adaptive gate that reduces guidance when the critic is uncertain or the task
   is outside training support.
3. Better critic targets and data balancing for sparse success-to-go labels.
4. Real robot deployment with non-privileged critic inputs only.

Paper-level evidence will require:

- stronger validation/test splits across task families;
- full ablations for beta, gradient clipping, ensemble size, and gate design;
- LIBERO-plus and LIBERO-PRO results;
- OOD-aware QGF analysis;
- real robot deployment or a credible sim-to-real study.

## Verification

Run the unit tests for local project glue:

```bash
python3 -m pytest tests -q
```

Current local verification for this codebase:

```text
31 passed in 0.77s
```

## Git Hygiene

Commit source, tests, scripts, and docs. Do not commit:

```text
checkpoints/
runs/
data/
outputs/
third_party/
.libero_configs/
*.pt, *.pth, *.safetensors, *.npy, *.npz, videos
```

The `.gitignore` is configured for this workflow. If a result is important,
summarize it in docs and keep the full artifact in external storage.
