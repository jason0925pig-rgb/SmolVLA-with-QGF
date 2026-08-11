# LIBERO Experiment Plan

Start with one LIBERO task and keep the first loop intentionally small.

## Phase 0: Adapter Bring-Up

- Load one LIBERO task.
- Reset and step with a dummy action.
- Confirm image keys, proprio keys, action dimension, horizon, and success key.
- Record these facts in `configs/benchmarks/libero.yaml`.

## Phase 1: Base Policy

- Load `lerobot/smolvla_base`.
- Map LIBERO observations into the SmolVLA input format.
- Run deterministic base-policy evaluation.
- Save per-episode success, return, video path, and failure tags.

## Phase 2: Critic Dataset

- Collect base-policy rollouts and, if needed, noisy action-chunk rollouts.
- Compute reward with `LiberoSparseSuccessReward` first.
- Add task-specific dense shaping only after confirming the raw observation fields.

## Phase 3: Critic Validation

Before using gradients, validate that the critic ranks actions correctly:

- correlation between predicted `Q` and rollout return
- value separation between successful and failed rollout action chunks
- ensemble disagreement
- nearest-neighbor distance to collected action chunks

## Phase 4: QGF

Enable guidance in this order:

1. small-weight in-loop QGF
2. guidance-weight sweep
3. gradient clipping sweep
4. ensemble-disagreement or OOD gating
