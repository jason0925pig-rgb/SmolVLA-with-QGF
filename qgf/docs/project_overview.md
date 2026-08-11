# Project Overview

## What This Project Does

This project studies whether test-time `Q` guidance can improve a flow-based VLA policy on robot manipulation tasks.

Initial concrete target:

```text
base policy: lerobot/smolvla_base
benchmark: LIBERO
method: Q-guided flow sampling over action chunks
first environment: simulation only
later direction: sim-to-real
```

The base policy should remain recognizable as a standard SmolVLA/LeRobot policy. The project-specific contribution is the additional critic and sampling-time guidance logic around its flow action generation.

## Research Question

The practical question is:

```text
Can a learned action-chunk critic guide SmolVLA's flow sampler toward higher-return LIBERO actions without retraining the whole VLA?
```

The first evidence should come from these comparisons:

1. Base SmolVLA.
2. Base SmolVLA plus in-loop vanilla QGF.
3. Base SmolVLA plus critic ensemble and adaptive disagreement gate.
4. Base SmolVLA plus task-description conditioned QGF.

Action reranking is intentionally out of scope for the current project track.

## What This Project Is Not

- Not a fork of SmolVLA.
- Not a rewrite of LeRobot.
- Not a new simulator.
- Not a full RL fine-tuning framework at the start.
- Not a claim that QGF works before critic ranking and rollout evidence are measured.

## Core Components

`guided_action_flow.policies`

Wraps the external SmolVLA/LeRobot policy behind a small `ChunkPolicy` interface.

`guided_action_flow.benchmarks`

Wraps LIBERO reset/step/task metadata behind a benchmark adapter.

`guided_action_flow.rewards`

Computes sparse or shaped rewards from normalized transitions. Simulation may use privileged state for reward computation, but the deployed QGF critic should not require privileged state as input.

`guided_action_flow.critics`

Trains action-chunk `Q(obs_features, proprio, action_chunk) -> scalar` models.

`guided_action_flow.guidance`

Applies `grad_a Q` during action sampling. The sign and time convention must be verified against the pinned SmolVLA sampler before enabling in-loop guidance.

## First Milestone Definition

The first useful milestone is not full QGF. It is:

```text
One LIBERO task runs with SmolVLA through this repo's adapter,
rollouts are saved,
success and return are logged,
and the exact upstream commits are recorded.
```

Only after that should we train the critic and enable guidance.
