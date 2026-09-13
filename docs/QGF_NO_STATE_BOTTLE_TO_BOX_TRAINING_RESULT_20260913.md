# Bottle-to-box no-state Q critic: training result

## Scope

This is the real-robot Q input ablation `no_state_Q_z_a`.  The Q critic sees
only frozen two-camera SmolVLA visual tokens and the normalized 50 x 8 action
chunk.  It does not receive current robot state or next-state.  The frozen
SmolVLA policy itself is unchanged and still receives its normal observations.

## Reproducible training setup

- Source branch / commit: `qgf-no-state-input-ablation-20260913` / `f49fdea02681f1dcd98769f6a21e020bf2a14852`.
- Dataset: the existing Bottle-to-box 100-rollout visual feature cache.
- Split: identical fixed episode-level 90/10 split used by full visual Q; SHA-256 `7bb5141a96937965dcece3de63f3bd7b18db6c144a82eeb6126d59a8652166a5`.
- Train / validation samples: 7,970 / 947 aligned 15 Hz policy chunks.
- Train / validation positive-reward samples: 94 / 10.
- Critic: `visual_action_transformer`, 128 x 960 visual tokens plus a 50 x 8 action chunk; 256 hidden dimensions, three Transformer layers, four heads.
- IQL: one member, 80 epochs, batch size 16, AdamW learning rate `3e-4`, weight decay `1e-4`, gamma `0.99`, expectile `0.7`, Polyak `0.005`, seed `20260814`.

## Selected checkpoint

The checkpoint selection criterion is minimum validation TD loss, not the last
epoch.  The selected point is epoch 1 with validation TD loss `0.00658719`.
The last epoch's validation TD loss was `0.01643356`; therefore deploying the
last epoch would be inappropriate for this small-data ablation.

The A800 artifact is deliberately not committed to Git:

`/ssd/zhuang/runs/qgf_input_ablation/bottle_to_box_no_state_20260913/artifacts/critic_member_00.pt`

The same run directory retains `training_summary.json`,
`training_input_summary.json`, the copied split, code commit, and full training
log.  It must be copied to the Orin model directory before attended real-robot
rollouts; no deployment was performed in this training-only run.
