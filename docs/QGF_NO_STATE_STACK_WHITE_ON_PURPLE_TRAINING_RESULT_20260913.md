# Stack white box on purple box: no-state Q critic result

## Scope

This run is the `no_state_Q_z_a` input ablation for **把白色盒子叠在紫色盒子上**.
The critic receives frozen two-camera SmolVLA visual tokens and the normalized
`50 x 8` action chunk. It receives neither current robot state nor next state.
The frozen SmolVLA policy is unchanged.

## Data and alignment

- Episodes: baseline `episode_000000` through `episode_000049`.
- Outcomes: 30 success / 20 failure.
- Video source: paired `chest.mp4` and `wrist_right.mp4`; raw MJPEG is not
  used by this pipeline.
- Alignment: exact normalized policy chunks, 15 Hz action timebase, 50-step
  horizon, and maximum start-transition mismatch of 0.1 s.
- Valid aligned samples: 3,798. The manifest excluded 481 chunks without a
  valid recorded policy observation/camera-frame pair.
- Fixed episode-level stratified split, seed `20260814`:
  - train: 45 episodes / 3,359 samples / 56 positive-reward samples;
  - validation: `9, 11, 12, 18, 25` (5 episodes / 439 samples / 6 positive
    reward samples; 3 success and 2 failure episodes).

## Training configuration

- Critic: `visual_action_transformer`, 128 visual tokens of dimension 960,
  50-step action horizon, hidden size 256, 3 Transformer layers, 4 heads,
  dropout 0.1.
- IQL: one critic member, 80 epochs, batch size 16, AdamW learning rate
  `3e-4`, weight decay `1e-4`, gamma `0.99`, expectile `0.7`, Polyak `0.005`,
  seed `20260814`.
- Runtime: local RTX 5070 Ti. The frozen visual encoder first extracted the
  feature cache; the critic then trained only on visual features and action
  chunks.
- Checkpoint selection: minimum validation TD loss, not final epoch.

## Result

- Selected epoch: **18**.
- Selected validation TD loss: **0.005336114214190145**.
- Final epoch-80 validation TD loss: `0.01626363233039488`; therefore the
  final epoch is not the deployment candidate.

## Artifact provenance

The binary weight is deliberately not committed because the repository ignores
model artifacts (`*.pt`) and datasets. The local artifact is:

`G:\zhuang_qgf\stack_white_on_purple\no_state_q_20260913\artifacts\critic_member_00.pt`

- Size: 20.90 MiB
- SHA-256: `637CA614F4B9C7205FBF9C884F83543D0D97AD2FDE4E876C0EEA73DFD8A91E8F`

The corresponding local run directory retains the raw-data junction manifest,
episode split, visual feature cache, stdout logs, training summary, and runner:

`G:\zhuang_qgf\stack_white_on_purple\no_state_q_20260913`

## Reproduction code

- Feature manifest: `qgf/scripts/build_real_robot_visual_iql_manifest.py`
- Frozen visual tokens: `qgf/scripts/extract_smolvla_visual_features.py`
- No-state IQL critic: `qgf/scripts/train_real_robot_visual_action_iql.py`
- Critic architecture: `qgf/src/guided_action_flow/critics/visual_action_transformer_critic.py`

The exact raw videos and feature cache are intentionally local-only and are not
required to inspect the training configuration or selected-result metadata.
