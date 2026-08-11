# Agent Task Spec

Use this as the instruction file for a coding agent working on the experiment machine.

## Objective

Build the first runnable LIBERO evaluation path for:

```text
SmolVLA base policy -> LIBERO env -> rollout metrics
```

Then extend it to:

```text
SmolVLA base policy -> action-chunk critic -> Q-guided flow sampler -> LIBERO metrics
```

## Hard Rules

- Do not fabricate LeRobot, SmolVLA, or LIBERO APIs.
- Inspect the pinned `third_party/` checkout before writing adapter code.
- Keep upstream code clean; do not patch `third_party/` unless the patch is explicitly recorded.
- Keep project-specific code under `src/guided_action_flow`.
- Do not commit model checkpoints, rollout datasets, videos, or simulator caches.
- Record every exact upstream commit used for a result.

## Implementation Order

1. Confirm `third_party/lerobot` and `third_party/LIBERO` are clean clones.
2. Install the LeRobot stack with LIBERO support.
3. Download `lerobot/smolvla_base`.
4. Run import and GPU checks.
5. Implement `SmolVLAAdapter.from_pretrained`.
6. Implement `LiberoAdapter.build`.
7. Implement `scripts/eval_policy.py` for base-policy evaluation.
8. Save metrics under `runs/`.
9. Only then implement rollout dataset collection.
10. Only then train critic and add guidance.

## Minimum Evidence Before QGF

The agent must produce:

```text
one task name
one fixed seed
base-policy success rate
episode lengths
total rewards
action shape
observation keys
success info key
SmolVLA sampler time convention
```

Without these facts, QGF implementation is premature.

## QGF Integration Notes

The QGF paper form assumes a forward-time flow:

```text
t = 0 noise -> t = 1 clean action
a_hat_1 = a_t + (1 - t) * v_theta(s, a_t, t)
v_guided = v_theta + beta^-1 * grad_a Q(s, a_hat_1)
```

If SmolVLA uses reverse time or integrates with negative `dt`, the guidance sign must be derived from the actual sampler implementation instead of copied blindly.

## First PR Should Contain

- implemented LIBERO adapter;
- implemented SmolVLA adapter;
- one runnable base evaluation script;
- one smoke-test config;
- one short note documenting verified upstream APIs.

It should not contain critic training or QGF unless base evaluation already works.

