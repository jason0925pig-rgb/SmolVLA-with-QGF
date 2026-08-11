# Agent Notes

Preferred language for explanations is Chinese, while technical terms should remain in English.

Do not guess external APIs for `LeRobot`, `SmolVLA`, or `LIBERO`. If an API is needed, inspect the installed version or the pinned checkout under `third_party/` before changing adapter code.

Keep this repository focused on project-specific glue:

- `src/guided_action_flow/policies`: wrappers around external VLA policies
- `src/guided_action_flow/benchmarks`: wrappers around external benchmark envs
- `src/guided_action_flow/guidance`: Q-guided flow logic
- `src/guided_action_flow/critics`: trainable Q models
- `src/guided_action_flow/rewards`: reward construction and diagnostics

Large generated artifacts belong in `data/`, `runs/`, or `checkpoints/`, not in git.

