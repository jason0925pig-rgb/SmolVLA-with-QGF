# Architecture

The project is organized around four boundaries.

## Policy

`SmolVLA` is treated as a frozen or lightly adapted base policy that produces an action chunk from image observations, proprioception, and a language instruction.

Project code should call it through `guided_action_flow.policies.base.ChunkPolicy`.

## Flow Guidance

`QGF` modifies the action sampling process at test time. The guidance module should not know about LIBERO internals or robot-specific observation keys. It receives tensors already prepared by the policy adapter:

```text
obs_features, action_t, time_t, flow_velocity -> guided_velocity
```

The sign and time convention must be verified against the installed SmolVLA sampler before enabling full in-loop guidance. Some flow policies use `t=0 noise -> t=1 action`; others implement the loop with a reversed time variable.
For the current pinned SmolVLA checkout, the verified reverse-time convention and QGF sign are recorded in `docs/qgf_smolvla_design.md`.

## Critic

The critic predicts a scalar value for an action chunk:

```text
Q(obs_features, proprio, action_chunk) -> scalar
```

For sim-to-real compatibility, the deployed QGF critic should avoid privileged simulator state as input. Privileged state can still be used to compute training rewards or diagnostics in simulation.

## Benchmark

LIBERO is wrapped behind `BenchmarkEnvAdapter`. The adapter owns:

- env construction
- reset/step lifecycle
- conversion from raw benchmark observations to project observations
- task language
- success extraction from `info`

Reward code should consume normalized transition data rather than raw LIBERO internals where possible.
