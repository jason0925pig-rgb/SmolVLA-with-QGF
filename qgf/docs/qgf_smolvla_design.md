# QGF for SmolVLA

This note records the verified QGF convention for the pinned LeRobot SmolVLA
checkout. It is intentionally limited to project-specific glue and does not
modify `third_party/`.

## Verified SmolVLA Flow Convention

The pinned implementation in
`third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py` trains:

```text
x_t = t * noise + (1 - t) * action
v_t = noise - action
```

Inference starts from noise at `t = 1` and integrates to `t = 0` with:

```text
dt = -1 / num_steps
x_t = x_t + dt * v_t
```

Therefore the clean action estimate at an intermediate SmolVLA timestep is:

```text
a_hat = x_t - t * v_t
```

This differs from the forward-time QGF expression used in the paper text for
`t = 0 noise -> t = 1 action`.

## Vanilla QGF Update

For a critic `Q(obs_features, proprio, action_chunk)`, the project implementation
queries the critic only at the estimated clean action:

```text
g = grad_a Q(obs_features, proprio, a_hat)
```

Because SmolVLA uses `a_hat = x_t - t * v_t`, increasing the clean action along
`+g` requires decreasing the velocity:

```text
v_guided = v_t - g / beta
```

The implementation detaches `x_t` and `v_t` before the critic query. This keeps
the method as test-time guidance and avoids backpropagation through the denoising
network.

Current implementation:

```text
src/guided_action_flow/guidance/qgf.py
src/guided_action_flow/policies/smolvla_qgf.py
src/guided_action_flow/training/critic_dataset.py
scripts/collect_rollouts.py
scripts/train_critic.py
scripts/eval_policy.py
tests/test_qgf.py
tests/test_critic_dataset.py
tests/test_smolvla_qgf.py
```

Verified behavior:

```text
python3 -m pytest tests/test_qgf.py -q
python3 -m pytest tests/test_qgf.py tests/test_critic_dataset.py tests/test_smolvla_qgf.py -q
```

## SmolVLA Integration Point

The in-loop guidance should be applied inside the action sampling loop after the
base denoiser returns `v_t` and before:

```text
x_t = x_t + dt * v_t
```

The intended local wrapper is:

```text
original_denoise_step(x_t) -> v_t
QGF(obs_features, proprio, x_t, v_t, t) -> v_guided
x_t = x_t + dt * v_guided
```

Do not implement action reranking for this project track. The ablation is:

```text
SmolVLA baseline
SmolVLA + in-loop vanilla QGF
```

LeRobot's stock `lerobot_eval.rollout()` calls `policy.select_action()` under
`torch.inference_mode()`. QGF needs `autograd.grad`, so the project wrapper must
use `torch.inference_mode(False)` and `torch.enable_grad()` around the critic
gradient path. A local smoke check confirmed that this can re-enable autograd
inside the outer inference-mode context.

The current implementation reuses SmolVLA's RTC denoise hook instead of copying
the full sampler. `install_smolvla_qgf()` installs a local processor on
`policy.model.rtc_processor`, enables the model-side RTC branch, and keeps the
policy-side `_rtc_enabled()` false so `select_action()` does not trip the RTC
assertion.

SmolVLA pads action chunks to `max_action_dim=32`, while LIBERO actions are
7-dimensional. QGF therefore passes only the critic action dimension to the
critic and leaves padded velocity dimensions unchanged.

## Critic Training Data

No dummy actions should be used for reported runs. The first critic should be
trained from real SmolVLA rollouts recorded in LIBERO:

```text
official SmolVLA 0.45B checkpoint -> LIBERO rollout recording -> action chunks + returns
```

Use sparse success-to-go first. Dense shaping and OOD-aware gating are later
ablations, not prerequisites for vanilla QGF.

For sim-to-real compatibility, critic inputs should prefer non-privileged
features available at deployment. Privileged simulator state may be used for
reward construction and diagnostics, but should be isolated from the deployed
critic input path unless explicitly marked as a privileged-state ablation.

Critic train/validation splits should be episode-level, not random chunk-level
splits. Randomly splitting action chunks from the same rollout leaks trajectory
context across train and validation and can make the critic look more stable
than it is.

## Current Baseline Anchors

Existing real baseline runs with official 0.45B checkpoints:

```text
runs/practical_baseline_vanilla_0p45b_4suites_5tasks_5eps/eval_info.json
runs/practical_baseline_plus_0p45b_spatial_10tasks_5eps/eval_info.json
runs/practical_baseline_pro_0p45b_4suites_5tasks_5eps/eval_info.json
```

Observed success rates:

```text
LIBERO vanilla: 65/100 = 65.0%
LIBERO-plus spatial subset: 39/50 = 78.0%
LIBERO-PRO zero-shot with vanilla checkpoint: 1/100 = 1.0%
```

These are the no-QGF anchors for the first ablation table.

## Smoke Evidence

The current code path has been smoke-tested on one real LIBERO rollout:

```text
runs/qgf_collect_smoke_libero_spatial0_1ep
runs/qgf_critic_smoke_libero_spatial0_1ep
runs/qgf_eval_script_baseline_smoke_libero_spatial0_1ep
runs/qgf_eval_script_qgf_smoke_libero_spatial0_1ep
```

Observed smoke results:

```text
collector: 1 episode, 80 steps, success=True
critic: 31 action-chunk samples, horizon=50
baseline eval script: 1/1 success
QGF eval script: 1/1 success
QGF guided denoise steps: 20
```

This smoke only verifies the data, training, and in-loop guidance path. It is
not evidence that QGF improves success rate; the actual ablation needs a larger
mixed success/failure critic dataset and the planned task budget.

## Single-Task Vanilla QGF Trial

First real single-task trial:

```text
suite/task: LIBERO vanilla, libero_spatial task_id=3
policy: official lerobot/smolvla_libero 0.45B checkpoint
train seeds: 2000-2049
eval seeds: 3000-3019
episodes: 50 train rollouts, 20 heldout eval episodes
critic input: policy-preprocessed observation.state + 7D action chunk
critic target: sparse success-to-go
critic horizon: 50
```

Training data and critic:

```text
runs/qgf_single_task_spatial3_train50
  50 real SmolVLA rollouts, 36 successes / 14 failures

runs/qgf_single_task_spatial3_critic_train50/critic.pt
  4602 action-chunk samples
  final train_loss=0.009576
  final val_loss=0.012906
```

Heldout eval results:

```text
baseline:
  runs/qgf_single_task_spatial3_eval20_baseline_seed3000
  11/20 success = 55.0%
  success seq: 10100101110101100110

QGF beta=50, grad_clip=0.1:
  runs/qgf_single_task_spatial3_eval20_qgf_beta50_clip0p1_seed3000
  11/20 success = 55.0%
  q_guidance_norm_mean=0.001704
  success seq unchanged

QGF beta=10, grad_clip=0.5:
  runs/qgf_single_task_spatial3_eval20_qgf_beta10_clip0p5_seed3000
  11/20 success = 55.0%
  q_guidance_norm_mean=0.015404
  success seq unchanged

QGF beta=2, grad_clip=1.0:
  runs/qgf_single_task_spatial3_eval20_qgf_beta2_clip1p0_seed3000
  14/20 success = 70.0%
  q_guidance_norm_mean=0.082130
  success seq: 10101111101111101010
```

Larger heldout beta sweep on the same task and same critic:

```text
suite/task: LIBERO vanilla, libero_spatial task_id=3
policy: official lerobot/smolvla_libero 0.45B checkpoint
critic: runs/qgf_single_task_spatial3_critic_train50/critic.pt
eval seeds: 3000-3049
episodes: 50 heldout eval episodes per setting
grad_clip: 1.0 for all QGF settings
```

Results:

```text
baseline:
  runs/qgf_single_task_spatial3_eval50_baseline_seed3000
  34/50 success = 68.0%
  success seq: 10100101110101100110110011011111110111010111111110

QGF beta=1, grad_clip=1.0:
  runs/qgf_single_task_spatial3_eval50_qgf_beta1_clip1p0_seed3000
  34/50 success = 68.0%
  q_guidance_norm_mean=0.157775
  gains vs baseline: [4, 12, 15, 19, 23, 26, 38, 40]
  regressions vs baseline: [8, 30, 31, 33, 37, 39, 44, 46]

QGF beta=2, grad_clip=1.0:
  runs/qgf_single_task_spatial3_eval50_qgf_beta2_clip1p0_seed3000
  34/50 success = 68.0%
  q_guidance_norm_mean=0.079396
  gains vs baseline: [4, 6, 10, 12, 16, 23, 26, 38, 40]
  regressions vs baseline: [9, 17, 30, 31, 32, 37, 39, 44, 46]

QGF beta=3, grad_clip=1.0:
  runs/qgf_single_task_spatial3_eval50_qgf_beta3_clip1p0_seed3000
  39/50 success = 78.0%
  q_guidance_norm_mean=0.053839
  gains vs baseline: [4, 6, 10, 12, 15, 16, 22, 26, 38, 40]
  regressions vs baseline: [13, 20, 24, 37, 46]
  success seq: 10101111111110111110011001111111110110111111110110

QGF beta=5, grad_clip=1.0:
  runs/qgf_single_task_spatial3_eval50_qgf_beta5_clip1p0_seed3000
  34/50 success = 68.0%
  q_guidance_norm_mean=0.031106
  gains vs baseline: [6, 49]
  regressions vs baseline: [8, 46]
```

Current interpretation:

```text
The 20-episode beta=2 gain did not hold over the larger 50-episode budget.
However, the same critic and task show a real +10 percentage-point improvement
at beta=3 over 50 heldout episodes. The sweep is not monotonic: beta=1 and
beta=2 are too aggressive for this critic/task pair, while beta=5 appears too
weak to improve aggregate success. Guidance scale is therefore a primary
ablation axis.
```

Independent heldout-seed stability check:

```text
suite/task: LIBERO vanilla, libero_spatial task_id=3
policy: official lerobot/smolvla_libero 0.45B checkpoint
critic: runs/qgf_single_task_spatial3_critic_train50/critic.pt
eval seeds: 4000-4049
episodes: 50 heldout eval episodes per setting
grad_clip: 1.0 for all QGF settings

baseline:
  runs/qgf_single_task_spatial3_eval50_baseline_seed4000
  41/50 success = 82.0%

QGF beta=2.5, grad_clip=1.0:
  runs/qgf_single_task_spatial3_eval50_qgf_beta2p5_clip1p0_seed4000
  40/50 success = 80.0%
  q_guidance_norm_mean=0.063833
  gains vs baseline: [7, 35, 36]
  regressions vs baseline: [5, 41, 44, 48]

QGF beta=3, grad_clip=1.0:
  runs/qgf_single_task_spatial3_eval50_qgf_beta3_clip1p0_seed4000
  40/50 success = 80.0%
  q_guidance_norm_mean=0.051095
  gains vs baseline: [1, 7, 9, 20, 23, 26, 35]
  regressions vs baseline: [8, 14, 19, 21, 34, 41, 44, 48]

QGF beta=3.5, grad_clip=1.0:
  runs/qgf_single_task_spatial3_eval50_qgf_beta3p5_clip1p0_seed4000
  37/50 success = 74.0%
  q_guidance_norm_mean=0.043740
  gains vs baseline: [1, 7, 9, 20, 23, 36]
  regressions vs baseline: [8, 14, 19, 21, 31, 37, 43, 47, 48, 49]
```

Updated interpretation after seed4000:

```text
The beta=3 result is seed-dependent on this single task. It improves seed3000
from 68.0% to 78.0%, but it does not replicate on the easier seed4000 window,
where baseline is already 82.0% and QGF beta=2.5/3/3.5 reduce success.

This means beta=3 should be treated as a useful positive anchor, not as a
validated global parameter. The next protocol should avoid per-test tuning:
choose beta on validation episodes, report test seeds once, and add adaptive
guidance gating or confidence-aware scaling before claiming robust improvement.
```

## Train200 Critic Protocol Check

To check whether the original instability was caused by too little critic data,
chunk-level validation leakage, or insufficient MLP capacity, a new critic
dataset was collected from held-out training seeds:

```text
suite/task: LIBERO vanilla, libero_spatial task_id=3
policy: official lerobot/smolvla_libero 0.45B checkpoint
train seeds: 5000-5199
episodes: 200 real SmolVLA rollouts
successes: 145/200 = 72.5%
rollout dir: runs/qgf_single_task_spatial3_train200_seed5000
```

The critic trainer now uses episode-level train/validation splitting. The split
was verified by unit test and by recorded metrics: 160 train episodes, 40
validation episodes, and zero episode overlap.

```text
test:
  python3 -m pytest tests/test_critic_dataset.py -q

current critic:
  runs/qgf_single_task_spatial3_critic_train200_current/critic.pt
  hidden_dim=512, depth=3
  samples=18341
  train samples=15343
  validation samples=2998
  best val_loss=0.020569 at epoch 3
  final train_loss=0.005088
  final val_loss=0.029658

big critic:
  runs/qgf_single_task_spatial3_critic_train200_big/critic.pt
  hidden_dim=1024, depth=4
  samples=18341
  train samples=15343
  validation samples=2998
  best val_loss=0.019776 at epoch 6
  final train_loss=0.004510
  final val_loss=0.033827
```

Both train200 critics overfit under the stricter episode-level validation
protocol. The larger critic reaches slightly better best validation loss, but
its final validation loss is worse, so capacity alone is not the stability fix.

Validation eval on the original seed3000 eval window:

```text
baseline:
  runs/qgf_single_task_spatial3_eval50_baseline_seed3000
  34/50 success = 68.0%

train50 critic, beta=3, grad_clip=1.0:
  runs/qgf_single_task_spatial3_eval50_qgf_beta3_clip1p0_seed3000
  39/50 success = 78.0%
  q_value_mean=0.098392
  q_grad_norm_raw_mean=0.161516
  q_guidance_norm_mean=0.053839
  gains vs baseline: [4, 6, 10, 12, 15, 16, 22, 26, 38, 40]
  regressions vs baseline: [13, 20, 24, 37, 46]

train200 current critic, beta=3, grad_clip=1.0:
  runs/qgf_single_task_spatial3_eval50_qgf_train200_current_beta3_clip1p0_seed3000
  34/50 success = 68.0%
  q_value_mean=0.114197
  q_grad_norm_raw_mean=0.128078
  q_guidance_norm_mean=0.042680
  gains vs baseline: [26, 40, 49]
  regressions vs baseline: [35, 39, 46]

train200 big critic, beta=3, grad_clip=1.0:
  runs/qgf_single_task_spatial3_eval50_qgf_train200_big_beta3_clip1p0_seed3000
  34/50 success = 68.0%
  q_value_mean=0.128200
  q_grad_norm_raw_mean=0.091883
  q_guidance_norm_mean=0.029626
  gains vs baseline: [19, 26, 38, 40]
  regressions vs baseline: [24, 30, 44, 46]
```

Interpretation:

```text
The train200 critics did not reproduce the earlier +10 percentage-point beta=3
gain on seed3000. They change individual episodes, but aggregate success returns
to baseline. The recorded gradient diagnostics also show weaker guidance norms
than the original train50 critic. The next implementation step should therefore
be best-validation checkpoint selection, critic ensembling, or adaptive gating,
not simply making the MLP larger or running more fixed-beta sweeps.
```

Limitations:

```text
This is still one task. The critic is state-only and single-task; it does not
yet use image features, language/task embeddings, OOD gating, or cross-task
data. The result is promising enough to justify adaptive QGF design, but not
enough for a paper claim yet.
```

## Best-Val Ensemble and Adaptive Gate

The next stability pass implemented three changes:

```text
1. Critic checkpoints now save the best validation model_state_dict when an
   episode-level validation split is available.
2. QGF accepts a critic ensemble and guides with the gradient of mean Q.
3. Adaptive gating scales the guidance by critic disagreement:
   gate = exp(-uncertainty_scale * std(Q_i)).clamp(min_gate, 1.0)
```

The action update is still in-loop vanilla QGF. No action reranking is used.

Verification:

```text
python3 -m pytest tests/test_qgf.py tests/test_critic_dataset.py tests/test_smolvla_qgf.py tests/test_train_critic.py tests/test_eval_policy.py -q
```

The K=3 critics were trained on the same 200 real rollout dataset:

```text
rollout dir: runs/qgf_single_task_spatial3_train200_seed5000
episodes: 200 real SmolVLA rollouts
successes: 145/200 = 72.5%
critic architecture: hidden_dim=512, depth=3
train/val split: episode-level, 160 train episodes / 40 validation episodes

critic seed 0:
  runs/qgf_single_task_spatial3_critic_train200_best_seed0/critic.pt
  selected_epoch=3
  selected_val_loss=0.020192

critic seed 1:
  runs/qgf_single_task_spatial3_critic_train200_best_seed1/critic.pt
  selected_epoch=23
  selected_val_loss=0.012065

critic seed 2:
  runs/qgf_single_task_spatial3_critic_train200_best_seed2/critic.pt
  selected_epoch=8
  selected_val_loss=0.026839
```

Seed3000 eval window:

```text
baseline:
  runs/qgf_single_task_spatial3_eval50_baseline_seed3000
  34/50 success = 68.0%

single best-val critic seed0, beta=3, grad_clip=1.0:
  runs/qgf_single_task_spatial3_eval50_qgf_train200_best_seed0_beta3_clip1p0_seed3000
  36/50 success = 72.0%
  q_guidance_norm_mean=0.035084
  gains vs baseline: [26, 38, 40]
  regressions vs baseline: [39]

K=3 ensemble, no adaptive gate, beta=3, grad_clip=1.0:
  runs/qgf_single_task_spatial3_eval50_qgf_train200_ensemble3_beta3_clip1p0_seed3000
  34/50 success = 68.0%
  q_value_std_mean=0.031989
  q_gate_mean=1.000000
  q_guidance_norm_mean=0.033873
  gains vs baseline: [4, 10, 12, 15, 16, 23, 38, 40]
  regressions vs baseline: [8, 20, 24, 32, 37, 39, 44, 46]

K=3 ensemble, adaptive gate, beta=3, grad_clip=1.0:
  runs/qgf_single_task_spatial3_eval50_qgf_train200_ensemble3_beta3_clip1p0_gate20_seed3000
  uncertainty_scale=20
  min_gate=0.1
  41/50 success = 82.0%
  q_value_std_mean=0.033089
  q_gate_mean=0.585183
  q_guidance_norm_mean=0.013663
  gains vs baseline: [4, 6, 10, 12, 15, 16, 22, 26, 34, 38, 40, 49]
  regressions vs baseline: [9, 20, 24, 31, 47]
```

Independent seed4000 eval window:

```text
baseline:
  runs/qgf_single_task_spatial3_eval50_baseline_seed4000
  41/50 success = 82.0%

K=3 ensemble, adaptive gate, beta=3, grad_clip=1.0:
  runs/qgf_single_task_spatial3_eval50_qgf_train200_ensemble3_beta3_clip1p0_gate20_seed4000
  uncertainty_scale=20
  min_gate=0.1
  43/50 success = 86.0%
  q_value_std_mean=0.035276
  q_gate_mean=0.556703
  q_guidance_norm_mean=0.015176
  gains vs baseline: [7, 9, 18, 26, 35, 36]
  regressions vs baseline: [13, 22, 45, 47]
```

Current interpretation:

```text
Best-validation checkpoint selection gives a small improvement over the final
train200 checkpoint on seed3000. A plain K=3 ensemble is not enough by itself:
without gating it changes many individual episodes but returns to the baseline
aggregate success rate.

The adaptive disagreement gate is the first train200 variant that improves both
tested eval windows on this task: +14 percentage points on seed3000 and +4
percentage points on seed4000. The gate reduces the average guidance norm from
0.033873 in the ungated ensemble to 0.013663 on seed3000 while preserving useful
episode-level gains.

This is still a single-task result. The next credible protocol is to lock
beta=3, grad_clip=1.0, uncertainty_scale=20, and min_gate=0.1 as the
single-task selected setting, then evaluate new LIBERO tasks/seeds without
per-test tuning.
```

## Fixed-Parameter Generalization Scout

This pass locks the single-task selected setting and does not tune on the
held-out/test tasks:

```text
beta=3
grad_clip_norm=1.0
uncertainty_scale=20
min_gate=0.1
critic ensemble: K=3 best-validation critics trained on vanilla libero_spatial task_id=3
eval seed: 6000
episodes: 5 per task
videos: disabled
```

Vanilla LIBERO held-out spatial tasks:

```text
task suite: libero_spatial
task ids: [0, 1, 2, 4]
baseline run:
  runs/qgf_generalize_vanilla_spatial_heldout4_ep5_baseline_seed6000
  14/20 success = 70.0%

fixed QGF run:
  runs/qgf_generalize_vanilla_spatial_heldout4_ep5_qgf_fixed_seed6000
  14/20 success = 70.0%
  q_value_std_mean=0.152425
  q_gate_mean=0.330027
  q_guidance_norm_mean=0.007429

per-task:
  task 0: baseline 4/5, QGF 4/5, gains=[], regressions=[]
  task 1: baseline 5/5, QGF 5/5, gains=[], regressions=[]
  task 2: baseline 4/5, QGF 4/5, gains=[], regressions=[]
  task 4: baseline 1/5, QGF 1/5, gains=[], regressions=[]
```

LIBERO-plus spatial variant tasks:

```text
task suite: libero_spatial
task ids: [0, 50, 100, 150]
baseline run:
  runs/qgf_generalize_plus_spatial_variants4_ep5_baseline_seed6000
  12/20 success = 60.0%

fixed QGF run:
  runs/qgf_generalize_plus_spatial_variants4_ep5_qgf_fixed_seed6000
  8/20 success = 40.0%
  q_value_std_mean=0.098042
  q_gate_mean=0.352658
  q_guidance_norm_mean=0.008629

per-task:
  task 0: baseline 3/5, QGF 3/5, gains=[], regressions=[]
  task 50: baseline 3/5, QGF 1/5, gains=[], regressions=[0, 1]
  task 100: baseline 4/5, QGF 1/5, gains=[], regressions=[2, 3, 4]
  task 150: baseline 2/5, QGF 3/5, gains=[0, 1], regressions=[4]
```

Current interpretation:

```text
The single-task task_id=3 critic does not show positive generalization under
the locked setting. On vanilla held-out spatial tasks it makes no episode-level
changes. On LIBERO-plus spatial variants it causes more regressions than gains
and reduces aggregate success by 20 percentage points.

The adaptive gate is active on these runs, but the average guidance norm is
very small: 0.007429 on vanilla held-out tasks and 0.008629 on LIBERO-plus.
This suggests the task_id=3 critic ensemble is uncertain or weakly actionable
outside its training task, and the remaining guidance can still be harmful on
some plus variants.

These results should not be used to retune beta or gate parameters. The next
technical step is to improve critic coverage, for example with a multi-task
and/or task-conditioned critic, then evaluate on a separate validation split
before touching held-out test tasks again.
```

## Multi-Task Task-Description Critic Scout

This pass tested whether a multi-task critic conditioned on the natural-language
task description is a better transfer path than a task-id conditioned critic.
Task ids are not portable across LIBERO, LIBERO-plus, and LIBERO-PRO, so the
implemented conditioning uses SmolVLA's own tokenized task description instead:

```text
task feature source: SmolVLA observation.language.tokens
feature construction: masked hashed bag-of-token vector
task_feature_dim: 128
extra text encoder: none
```

This is intentionally lightweight and dependency-free. It is task-description
conditioned, but it is not yet a semantic sentence embedding or frozen VLM text
hidden-state critic.

Verification for the implementation:

```text
python3 -m pytest \
  tests/test_task_features.py \
  tests/test_critic_dataset.py \
  tests/test_qgf.py \
  tests/test_smolvla_qgf.py \
  tests/test_train_critic.py \
  tests/test_eval_policy.py -q
```

Data collection:

```text
rollout dir: runs/qgf_taskdesc_multitask_spatial0to4_train250_seed7200
suite: LIBERO vanilla, libero_spatial
policy: official lerobot/smolvla_libero 0.45B checkpoint
task ids: [0, 1, 2, 3, 4]
episodes: 50 per task, 250 total real SmolVLA rollouts
overall success: 170/250 = 68.0%
collection wall time: 1096.3s

task 0: 44/50 = 88.0%
task 1: 31/50 = 62.0%
task 2: 38/50 = 76.0%
task 3: 36/50 = 72.0%
task 4: 21/50 = 42.0%
```

Critics:

```text
pooled critic:
  runs/qgf_multitask_spatial0to4_critic_pooled_seed0/critic.pt
  task_feature_source=none
  task_feature_dim=0
  samples=27160
  selected_epoch=2
  selected_val_loss=0.020344
  final train_loss=0.002491
  final val_loss=0.027917

task-token critic:
  runs/qgf_multitask_spatial0to4_critic_tasktokens_seed0/critic.pt
  task_feature_source=tokens
  task_feature_dim=128
  samples=27160
  selected_epoch=2
  selected_val_loss=0.020173
  final train_loss=0.002619
  final val_loss=0.027615
```

Both critics overfit after the early selected epoch. The task-token critic has
only a very small best-validation loss advantage over the pooled critic.

Fixed-parameter eval protocol:

```text
beta=3
grad_clip_norm=1.0
uncertainty_scale=20
min_gate=0.1
critic ensemble size: 1
eval seed: 7600
episodes: 5 per task
videos: disabled
```

Seen multi-task eval on training task ids [0, 1, 2, 3, 4]:

```text
baseline:
  runs/qgf_taskdesc_eval_vanilla_seen5_ep5_baseline_seed7600
  18/25 success = 72.0%

pooled QGF:
  runs/qgf_taskdesc_eval_vanilla_seen5_ep5_qgf_pooled_seed7600
  18/25 success = 72.0%
  q_grad_norm_raw_mean=0.057225
  q_guidance_norm_mean=0.019075
  q_gate_mean=1.000

task-token QGF:
  runs/qgf_taskdesc_eval_vanilla_seen5_ep5_qgf_tasktokens_seed7600
  18/25 success = 72.0%
  q_grad_norm_raw_mean=0.056770
  q_guidance_norm_mean=0.018923
  q_gate_mean=1.000

per-task success sequences were identical for all three settings:
  task 0: 4/5 10111
  task 1: 4/5 11011
  task 2: 4/5 10111
  task 3: 4/5 10111
  task 4: 2/5 00101
```

Held-out vanilla spatial eval on task ids [5, 6, 7, 8, 9]:

```text
baseline:
  runs/qgf_taskdesc_eval_vanilla_heldout5_ep5_baseline_seed7600
  15/25 success = 60.0%

pooled QGF:
  runs/qgf_taskdesc_eval_vanilla_heldout5_ep5_qgf_pooled_seed7600
  15/25 success = 60.0%
  q_grad_norm_raw_mean=0.060129
  q_guidance_norm_mean=0.020043
  q_gate_mean=1.000

task-token QGF:
  runs/qgf_taskdesc_eval_vanilla_heldout5_ep5_qgf_tasktokens_seed7600
  15/25 success = 60.0%
  q_grad_norm_raw_mean=0.059697
  q_guidance_norm_mean=0.019899
  q_gate_mean=1.000

per-task:
  task 5: baseline 0/5 00000, pooled 0/5 00000, task-token 0/5 00000
  task 6: baseline 5/5 11111, pooled 5/5 11111, task-token 5/5 11111
  task 7: baseline 3/5 01110, pooled 3/5 01011, task-token 3/5 01011
  task 8: baseline 2/5 10100, pooled 2/5 10100, task-token 2/5 10100
  task 9: baseline 5/5 11111, pooled 5/5 11111, task-token 5/5 11111
```

Interpretation:

```text
This task-token critic did not improve aggregate success on either seen or
held-out vanilla spatial tasks. It is not just a no-op: on held-out task 7 it
changes individual outcomes, but the gain and regression cancel out.

The hashed token feature is probably too weak to make task semantics actionable
for the critic. It also does not fix the more important issue: a single critic
without ensemble disagreement has q_gate_mean=1.0, so the adaptive gate is not
active. The guidance scale is modest, but still enough to change trajectories.

The current best path is not task-id conditioning and not this hashed-token
conditioning alone. The next credible variant should use:

1. a K=3 task-description critic ensemble, using the same adaptive disagreement
   gate that helped the single-task critic;
2. a stronger task-description feature, preferably a frozen SmolVLA/VLM text
   hidden-state feature rather than a hashed bag of token ids;
3. validation-task selection before held-out benchmark reporting, with no
   parameter tuning on test tasks.
```

## K=3 VLM-Hidden Task-Description Critic

This pass implements the stronger task-description critic requested after the
hashed-token scout:

```text
task feature source: frozen SmolVLA/VLM text hidden state
feature construction: language-only VLM text stack, masked mean pooling
feature key in episodes: task_vlm_hidden
feature dimension: 960
critic ensemble: K=3
QGF gate: adaptive critic-disagreement gate
```

The feature is computed from `observation.language.tokens` and
`observation.language.attention_mask` using the pinned SmolVLA
`vlm_with_expert` text path. It follows SmolVLA's own language embedding
scaling and uses `fill_kv_cache=True`, matching the prefix-cache branch needed
by the local SmolVLA implementation. No image/state context is included in the
task feature; this keeps the feature task-description-only and allows
precomputation from existing rollout files.

Implementation files:

```text
src/guided_action_flow/training/task_features.py
src/guided_action_flow/training/critic_dataset.py
src/guided_action_flow/policies/smolvla_qgf.py
scripts/precompute_task_features.py
scripts/train_critic.py
scripts/eval_policy.py
tests/test_task_features.py
tests/test_critic_dataset.py
tests/test_smolvla_qgf.py
tests/test_precompute_task_features.py
tests/test_train_critic.py
tests/test_eval_policy.py
```

Precompute:

```text
input rollouts:
  runs/qgf_taskdesc_multitask_spatial0to4_train250_seed7200

output rollouts with VLM hidden feature:
  runs/qgf_taskdesc_multitask_spatial0to4_train250_vlm_hidden

episodes: 250/250
feature shape: (960,)
example norm: 25.56
```

Critic training:

```text
data: runs/qgf_taskdesc_multitask_spatial0to4_train250_vlm_hidden
task ids in train rollouts: [0, 1, 2, 3, 4]
samples: 27160 action chunks
architecture: hidden_dim=512, depth=3
action_horizon=50
task_feature_source=vlm_hidden
task_feature_key=task_vlm_hidden

seed 0:
  runs/qgf_multitask_spatial0to4_critic_vlm_hidden_seed0/critic.pt
  selected_epoch=2
  selected_val_loss=0.019646
  final train_loss=0.005153
  final val_loss=0.028466

seed 1:
  runs/qgf_multitask_spatial0to4_critic_vlm_hidden_seed1/critic.pt
  selected_epoch=3
  selected_val_loss=0.025129
  final train_loss=0.005570
  final val_loss=0.035605

seed 2:
  runs/qgf_multitask_spatial0to4_critic_vlm_hidden_seed2/critic.pt
  selected_epoch=2
  selected_val_loss=0.019770
  final train_loss=0.005384
  final val_loss=0.026209
```

Validation protocol:

```text
validation task ids: [5, 7, 8]
validation seed: 7800
episodes: 5 per task, 15 total per setting
baseline run:
  runs/qgf_vlm_hidden_val_spatial_5_7_8_ep5_baseline_seed7800

candidate grid:
  beta in [2, 3, 5]
  uncertainty_scale in [10, 20]
  grad_clip_norm=1.0
  min_gate=0.1
```

Validation results:

```text
baseline:
  7/15 success = 46.7%
  task 5: 0/5 00000
  task 7: 5/5 11111
  task 8: 2/5 11000

beta=2, uncertainty_scale=10:
  runs/qgf_vlm_hidden_val_spatial_5_7_8_ep5_qgf_beta2_gate10_seed7800
  6/15 success = 40.0%
  q_gate_mean=0.733915
  q_guidance_norm_mean=0.014563
  gains=[(5, 1), (8, 2)]
  regressions=[(7, 0), (7, 3), (8, 1)]

beta=2, uncertainty_scale=20:
  runs/qgf_vlm_hidden_val_spatial_5_7_8_ep5_qgf_beta2_gate20_seed7800
  7/15 success = 46.7%
  q_gate_mean=0.542087
  q_guidance_norm_mean=0.010896
  gains=[]
  regressions=[]

beta=3, uncertainty_scale=10:
  runs/qgf_vlm_hidden_val_spatial_5_7_8_ep5_qgf_beta3_gate10_seed7800
  7/15 success = 46.7%
  q_gate_mean=0.719907
  q_guidance_norm_mean=0.010209
  gains=[]
  regressions=[]

beta=3, uncertainty_scale=20:
  runs/qgf_vlm_hidden_val_spatial_5_7_8_ep5_qgf_beta3_gate20_seed7800
  7/15 success = 46.7%
  q_gate_mean=0.553856
  q_guidance_norm_mean=0.007300
  gains=[]
  regressions=[]

beta=5, uncertainty_scale=10:
  runs/qgf_vlm_hidden_val_spatial_5_7_8_ep5_qgf_beta5_gate10_seed7800
  5/15 success = 33.3%
  q_gate_mean=0.725170
  q_guidance_norm_mean=0.006071
  gains=[]
  regressions=[(7, 2), (7, 3)]

beta=5, uncertainty_scale=20:
  runs/qgf_vlm_hidden_val_spatial_5_7_8_ep5_qgf_beta5_gate20_seed7800
  7/15 success = 46.7%
  q_gate_mean=0.536016
  q_guidance_norm_mean=0.004337
  gains=[]
  regressions=[]
```

Validation selection:

```text
No QGF candidate beat the baseline validation success rate. Among QGF
candidates, the selected locked setting is:

beta=5
grad_clip_norm=1.0
uncertainty_scale=20
min_gate=0.1

Reason: tied best validation SR, zero validation regressions relative to
baseline, and lowest average guidance norm among tied QGF candidates.
```

Locked test protocol:

```text
test task ids: [6, 9]
test seed: 7900
episodes: 10 per task, 20 total per setting
baseline run:
  runs/qgf_vlm_hidden_test_spatial_6_9_ep10_baseline_seed7900
locked QGF run:
  runs/qgf_vlm_hidden_test_spatial_6_9_ep10_qgf_beta5_gate20_seed7900
```

Locked test results:

```text
baseline:
  16/20 success = 80.0%
  task 6: 7/10 0111011011
  task 9: 9/10 1111101111

locked QGF:
  17/20 success = 85.0%
  task 6: 8/10 0111011111
  task 9: 9/10 0111111111
  q_value_std_mean=0.023368
  q_gate_mean=0.661483
  q_grad_norm_raw_mean=0.043700
  q_guidance_norm_mean=0.004514
  num_guided_denoise_steps=670
  gains=[(6, 7), (9, 5)]
  regressions=[(9, 0)]
```

Current interpretation:

```text
The K=3 VLM-hidden task-description critic is a cleaner transfer design than
task-id conditioning or hashed token features. Runtime integration works with
official SmolVLA checkpoints, and the locked test split shows a small +1/20
episode gain after validation-only selection.

The result is not yet a strong claim. Validation did not outperform baseline,
and the selected QGF setting was chosen only among tied QGF candidates. The test
gain is encouraging but small and needs larger validation/test budgets and more
task families before it can support a paper-level claim.

The most important next step is not more test tuning. It is to increase the
validation budget and add task families, then run the locked protocol on new
tasks once.
```

## Expanded Validation After VLM-Hidden Scout

The first VLM-hidden scout used only 5 episodes per validation task and only
`libero_spatial`. To reduce seed noise and check task-family transfer, the next
pass increased the validation budget and added `libero_object` without changing
the critic ensemble.

Protocol:

```text
critic ensemble:
  runs/qgf_multitask_spatial0to4_critic_vlm_hidden_seed0/critic.pt
  runs/qgf_multitask_spatial0to4_critic_vlm_hidden_seed1/critic.pt
  runs/qgf_multitask_spatial0to4_critic_vlm_hidden_seed2/critic.pt

validation suites:
  libero_spatial task ids: [5, 7, 8]
  libero_object task ids: [0, 1, 2]

seed: 8000
episodes: 10 per task, 60 total across both validation suites

fixed QGF settings:
  grad_clip_norm=1.0
  uncertainty_scale=20
  min_gate=0.1

candidate beta values: [2, 3, 5]
```

Expanded validation results:

```text
libero_spatial validation:
  baseline:
    runs/qgf_vlm_hidden_val2_spatial_ep10_baseline_seed8000
    12/30 success = 40.0%
    task 5: 2/10 0001001000
    task 7: 5/10 0101110100
    task 8: 5/10 1010001011

  beta=2:
    runs/qgf_vlm_hidden_val2_spatial_ep10_beta2_gate20_seed8000
    11/30 success = 36.7%
    q_gate_mean=0.549656
    q_guidance_norm_mean=0.011020

  beta=3:
    runs/qgf_vlm_hidden_val2_spatial_ep10_beta3_gate20_seed8000
    9/30 success = 30.0%
    q_gate_mean=0.548537
    q_guidance_norm_mean=0.006978

  beta=5:
    runs/qgf_vlm_hidden_val2_spatial_ep10_beta5_gate20_seed8000
    9/30 success = 30.0%
    q_gate_mean=0.546610
    q_guidance_norm_mean=0.004176

libero_object validation:
  baseline:
    runs/qgf_vlm_hidden_val2_object_ep10_baseline_seed8000
    20/30 success = 66.7%
    task 0: 6/10 1000110111
    task 1: 8/10 1110111101
    task 2: 6/10 1001101110

  beta=2:
    runs/qgf_vlm_hidden_val2_object_ep10_beta2_gate20_seed8000
    20/30 success = 66.7%
    q_gate_mean=0.695510
    q_guidance_norm_mean=0.009736

  beta=3:
    runs/qgf_vlm_hidden_val2_object_ep10_beta3_gate20_seed8000
    20/30 success = 66.7%
    q_gate_mean=0.692403
    q_guidance_norm_mean=0.006554

  beta=5:
    runs/qgf_vlm_hidden_val2_object_ep10_beta5_gate20_seed8000
    20/30 success = 66.7%
    q_gate_mean=0.691639
    q_guidance_norm_mean=0.003929

aggregate expanded validation:
  baseline: 32/60 success = 53.3%
  beta=2: 31/60 success = 51.7%
  beta=3: 29/60 success = 48.3%
  beta=5: 29/60 success = 48.3%
```

Selection conclusion:

```text
If baseline is included in validation selection, the selected policy is no-QGF.
No QGF candidate beat baseline on the expanded validation split.

The best QGF candidate is beta=2, uncertainty_scale=20, min_gate=0.1, but it is
still below baseline by 1/60 episodes. This makes beta=2/gate20 useful only as a
diagnostic held-out run, not as a deployable locked improvement claim.
```

Diagnostic held-out object run:

```text
suite: libero_object
task ids: [3, 4, 5]
seed: 8100
episodes: 10 per task, 30 total per setting

baseline:
  runs/qgf_vlm_hidden_test2_object_3_4_5_ep10_baseline_seed8100
  18/30 success = 60.0%
  task 3: 5/10 1101000101
  task 4: 8/10 1111100111
  task 5: 5/10 1100100101

beta=2, uncertainty_scale=20:
  runs/qgf_vlm_hidden_test2_object_3_4_5_ep10_qgf_beta2_gate20_seed8100
  18/30 success = 60.0%
  task 3: 5/10 1101000101
  task 4: 8/10 1111100111
  task 5: 5/10 1100010101
  q_value_std_mean=0.027730
  q_gate_mean=0.656346
  q_grad_norm_raw_mean=0.038576
  q_guidance_norm_mean=0.009477
  num_guided_denoise_steps=1270
  gains=[(5, 5)]
  regressions=[(5, 4)]
```

Updated interpretation:

```text
The spatial-only K=3 VLM-hidden critic does not validate as a robust
cross-family QGF setting. The earlier +1/20 spatial held-out gain remains a
useful signal that runtime QGF can affect outcomes, but the larger validation
split selects baseline, and the object held-out diagnostic only ties baseline.

The next scientifically clean step is to collect or reuse multi-family
successful/failed rollouts and train a multi-family task-description critic.
Further beta tuning on these held-out tasks would turn the test set into a
validation set and should be avoided.
```
