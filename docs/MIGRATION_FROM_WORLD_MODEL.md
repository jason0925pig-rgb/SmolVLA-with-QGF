# Repository migration

This repository is now the maintained home for work that starts at model
baseline evaluation and continues through SmolVLA deployment, critic training,
QGF guidance, and real-robot policy rollout.

## Sources consolidated here

- The active `guided-action-flow` working tree, including local critic and
  counterfactual-training changes.
- SmolVLA baseline and Armstrong policy-runtime files formerly stored in
  `One-Arm-Teleoperation`.
- LIBERO baseline/QGF clients and concise experiment reports formerly stored
  in the local `World_Model` research directory.
- QGF and real-robot data-format notes formerly scattered across both trees.

## Intentionally excluded

- SSH private keys and credentials.
- Model checkpoints, datasets, virtual environments, caches, and build output.
- Raw temporary rollout directories and the bulk collection of experiment
  videos. Keep those on artifact storage rather than Git.
- Downloaded paper PDFs and untouched third-party repository clones. Their
  upstream URLs should be documented instead of vendoring them here.

`One-Arm-Teleoperation` remains the hardware/data repository. This repository
depends on its installed ROS2 workspace through `TELEOP_PROJECT_ROOT`.
