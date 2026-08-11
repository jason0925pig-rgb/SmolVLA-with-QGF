# QGF and Q Critic Algorithm Details

This note summarizes the current SmolVLA + QGF pipeline, including the vectors, critic architectures, training targets, IQL-related direction, uncertainty gate, progress score, task-language features, and counterfactual experiments we discussed or implemented.

## Big Picture

Baseline SmolVLA does this:

```text
image observations + robot state + language instruction
    -> SmolVLA flow/action model
    -> predicted future action chunk
    -> robot executes actions
```

QGF adds a critic during action generation:

```text
candidate action chunk
    -> Q critic gives score
    -> gradient of score with respect to action
    -> modify SmolVLA denoising velocity
    -> produce a slightly more promising action chunk
```

The base policy is not retrained during QGF inference. The critic is trained separately, then plugged into SmolVLA at test time.

## What SmolVLA Predicts

SmolVLA is a flow-based VLA policy. In this project, it predicts an action chunk rather than only one action.

For LIBERO, the actual robot action dimension used by the critic is usually:

```text
action_dim = 7
```

The SmolVLA internal padded action can have larger max dimension, but QGF slices the candidate action to the critic action dimension.

A typical action chunk is:

```text
action_chunk shape = [batch, action_horizon, action_dim]
```

Example:

```text
[1, 50, 7]
```

where 50 means the critic evaluates a future 50-step action sequence.

## SmolVLA Flow Convention

The pinned SmolVLA implementation uses:

```text
x_t = t * noise + (1 - t) * action
v_t = noise - action
```

Inference starts from noise at:

```text
t = 1
```

and integrates toward:

```text
t = 0
```

The clean action estimate at an intermediate step is:

```text
a_hat = x_t - t * v_t
```

This is important because the sign of QGF depends on this reverse-time convention.

## Vanilla QGF Math

The critic predicts:

```text
Q(obs_features, action_chunk, optional_task_features) -> scalar
```

At a SmolVLA denoising step:

```text
x_t = current noisy action chunk
v_t = SmolVLA predicted flow velocity
t = current denoising time
```

First estimate the clean action:

```text
a_hat = x_t - t * v_t
```

Then ask the critic:

```text
q = Q(obs_features, a_hat, task_features)
```

Then compute action-gradient:

```text
g = grad_a Q(obs_features, a_hat, task_features)
```

Because this SmolVLA convention has `a_hat = x_t - t * v_t`, increasing the clean action along `+g` means reducing the velocity by `g / beta`:

```text
v_guided = v_t - g / beta
```

Then SmolVLA continues its normal denoising update using `v_guided`.

## Meaning of Beta

`beta` controls guidance strength:

```text
v_guided = v_t - grad_Q / beta
```

So:

```text
smaller beta -> stronger Q guidance
larger beta  -> weaker Q guidance
```

`beta = 4` does not mean "Q takes 4x weight". It means the gradient is divided by 4.

The effective Q guidance size is approximately:

```text
||grad_Q|| / beta
```

So beta is a denominator, not a percentage.

## Does QGF Use Jacobian or BPTT?

Our implementation does not backpropagate through the full SmolVLA denoiser and does not compute the denoiser Jacobian.

Current implementation:

```text
1. detach x_t and v_t
2. estimate clean action a_hat
3. require grad only on a_hat
4. compute Q(a_hat)
5. compute grad_a Q
6. modify velocity once
```

This is why we call it test-time guidance.

It avoids:

- BPTT through the whole denoising process;
- backprop through SmolVLA weights;
- expensive Jacobian computation of the policy network.

It does not fully solve OOD by itself. It reduces some optimization complexity, but if the critic is trained on narrow data or does not see vision, guidance can still push actions in a bad direction.

## Runtime Inputs to the Current Critic

Current QGF hook uses:

```text
obs_features = latest batch["observation.state"]
action_chunk = estimated clean action chunk
task_features = optional language feature
```

Important:

```text
The current critic path does not directly pass raw RGB images.
```

SmolVLA uses vision internally to propose actions. But the Q critic mainly evaluates the proposed action using state/action/language features.

## Critic Architecture 1: MLP Critic

File:

```text
src/guided_action_flow/critics/action_chunk_critic.py
```

Input is concatenated:

```text
flat_obs_features
+ optional proprio
+ optional task_features
+ flat_action_chunk
```

Then an MLP predicts one scalar:

```text
Linear -> SiLU -> Linear -> SiLU -> ... -> Linear(1)
```

Default style:

```text
hidden_dim = 512
depth = 3
```

This is simple and fast, but it flattens the whole action chunk. It does not explicitly model action-step order except through position in the flattened vector.

## Critic Architecture 2: Transformer Action-Chunk Critic

File:

```text
src/guided_action_flow/critics/transformer_action_chunk_critic.py
```

This critic builds tokens:

```text
[CLS]
obs token
optional proprio token
optional task-language token
action token 0
action token 1
...
action token H-1
```

Each action step becomes one token:

```text
action_t -> Linear(action_dim, d_model)
```

Then a Transformer Encoder processes the sequence:

```text
TransformerEncoder -> CLS head -> scalar Q
```

Typical config:

```text
d_model = 256
num_layers = 3
num_heads = 4
dropout = 0.1
```

Why this is useful:

- it can model temporal relations inside the action chunk;
- it treats action steps as an ordered sequence;
- it is more expressive than a flat MLP.

Why it may not be enough:

- if the input still lacks image features, a Transformer critic can still ignore the visual scene;
- stronger architecture cannot fix missing information.

## Task-Language Features

We tried / implemented two language-conditioning routes.

### 1. Hashed token feature

File:

```text
src/guided_action_flow/training/task_features.py
```

The tokenizer IDs are hashed into a fixed-size vector:

```text
task_tokens -> hash buckets -> normalized bag-of-token vector
```

Example:

```text
task_feature_dim = 128
```

Pros:

- cheap;
- easy to save;
- no extra forward pass through VLM.

Cons:

- shallow semantic representation;
- similar sentences may not be close in feature space;
- word order and deeper language meaning are weak.

### 2. VLM hidden feature

This uses SmolVLA's language/VLM stack:

```text
task_tokens -> SmolVLA VLM text hidden states -> masked mean pooling
```

Pros:

- semantically stronger;
- aligned with SmolVLA's own language encoder.

Cons:

- more expensive;
- must ensure train-time and runtime feature dimensions match;
- still language-only, not image-language hidden unless we explicitly add visual features.

## Training Target 1: Success-To-Go

Basic sparse target:

```text
target_t = gamma^(steps until first future success)
```

If no future success exists:

```text
target_t = 0
```

After success:

```text
target_t = 1
```

This lets us train from episode-level success/fail labels.

Weakness:

Failed trajectories become mostly zero, even if some failed attempts were close to success.

## Training Target 2: Progress Blend

We discussed and implemented a blended target:

```text
target = (success_weight * success_to_go + progress_weight * progress_score)
         / (success_weight + progress_weight)
```

Typical setting:

```text
success_weight = 0.7
progress_weight = 0.3
```

All values are clamped to:

```text
[0, 1]
```

Meaning:

- `success_to_go` says whether this trajectory eventually succeeds;
- `progress_score` says whether this state looks partially completed.

This gives failed-but-near-success steps a non-zero target.

## Training Target 3: Retrieval Progress Blend

File:

```text
src/guided_action_flow/training/critic_dataset.py
```

For each state in an episode:

```text
1. find similar states from successful rollouts;
2. read their success-to-go values;
3. average nearest K values;
4. use that as progress estimate.
```

Then blend:

```text
target = 0.7 * success_to_go + 0.3 * retrieval_progress
```

This was useful because many failed rollouts had all-zero success-to-go. Retrieval progress provides a denser signal.

Limitation:

If retrieval uses only robot state, it may still retrieve by "arm pose" rather than "visual correctness".

## IQL / Offline Q-Learning Direction

The current `train_critic.py` is primarily supervised regression:

```text
MSE(Q(state, action_chunk), target)
```

This is Q-like, but not a full Bellman IQL implementation.

IQL usually involves:

```text
Q(s, a)
V(s)
advantage = Q(s, a) - V(s)
expectile regression for V
Bellman target for Q
advantage-weighted behavior cloning for policy, optional
```

In our project language, "Transformer-IQL critic" means the desired stronger direction is:

```text
offline data -> train a Transformer-style action critic with IQL-like value learning
```

But the local code path we inspected is mostly:

```text
success-to-go / progress target regression
optional counterfactual ranking
```

So if we present it honestly:

```text
Current implemented critic: supervised action-chunk value regression, QGF-compatible.
IQL direction: a stronger future replacement or upstream method to reproduce.
```

## Uncertainty Gate

The QGF config supports critic ensembles:

```text
critic = [critic_1, critic_2, critic_3]
```

At runtime:

```text
q_mean = mean(Q_i)
q_std = std(Q_i)
```

The gradient is gated by uncertainty:

```text
gate = exp(-uncertainty_scale * q_std)
gate = clamp(gate, min_gate, 1.0)
guided_grad = grad_Q * gate
```

Meaning:

- critics agree -> low std -> gate near 1 -> guidance active;
- critics disagree -> high std -> gate smaller -> guidance weakened.

If:

```text
uncertainty_scale = 0
```

then:

```text
gate = 1
```

and uncertainty gating is off.

## Gradient Clip

Before applying guidance, the gradient norm can be clipped:

```text
grad = grad * min(1, grad_clip_norm / ||grad||)
```

Purpose:

- prevent unstable action updates;
- avoid Q overconfidence causing huge action changes;
- make beta sweeps more meaningful.

## Counterfactual Action Critic

The counterfactual idea is:

```text
from the same state S, try multiple action chunks A/B/C
compare their outcomes
train Q to rank better actions higher
```

In simulation, we can save and restore MuJoCo/robosuite state:

```text
snapshot S
restore S -> execute action chunk A -> score_A
restore S -> execute action chunk B -> score_B
restore S -> execute action chunk C -> score_C
```

Then create ranking pairs:

```text
if score_A > score_B:
    enforce Q(S, A) > Q(S, B)
```

Implemented script:

```text
scripts/counterfactual_probe.py
scripts/train_counterfactual_mixed_critic.py
```

Mixed training objective:

```text
total_loss =
    mse_weight * MSE(Q(s, a), success_to_go)
  + ranking_weight * softplus(-(Q(s, good_a) - Q(s, bad_a)))
  + cf_regression_weight * MSE(Q(s, cf_a), cf_score)
```

This is closer to "action critic" than pure trajectory-progress scoring.

Limitation:

The current counterfactual critic script still uses `obs_feature`/state features. To make it visually grounded, the counterfactual samples should also save image features or RGB frames from the branch point.

## Public Data + Own Rollouts

We discussed mixing:

```text
public expert episodes
+ our SmolVLA baseline rollouts
+ counterfactual branch samples
```

The important caveat:

Public expert data does not have to be generated by SmolVLA. But if it comes from a different policy, its action distribution may differ. That can help coverage, but it can also make the critic score actions that SmolVLA cannot naturally produce.

Best practice:

```text
expert/public data -> teaches what success looks like
own SmolVLA rollouts -> teaches what SmolVLA actually tends to do
counterfactual branches -> teaches which alternative action is better at the same state
```

## Probe-Based Task Selection

Before final comparison, we used or discussed probe filtering:

```text
run baseline on candidate tasks
select tasks whose baseline success rate is around 30%-70%
```

Reason:

- 0%-10% tasks are too hard; QGF has little chance to rescue them;
- 90%-100% tasks are too easy; little room for improvement;
- 30%-70% tasks show whether QGF changes outcomes.

For perturbation suites, the desired protocol is:

```text
for each perturbation dimension:
    probe many candidate tasks
    keep 5 or more mid-difficulty tasks
    evaluate baseline and QGF on the same task/seed set
```

## What We Actually Changed or Tried

### 1. Vanilla QGF

Implemented:

```text
v_guided = v_t - grad_Q / beta
```

No uncertainty gate, no progress target, no language feature required.

Purpose:

Prove the critic can be inserted into SmolVLA's denoising loop.

### 2. Beta sweep

Tried different beta values.

Observation:

The best beta is not globally fixed. Lower beta gives stronger guidance but can damage performance when critic is wrong. Higher beta is safer but may be too weak.

### 3. Gradient clipping

Used to prevent overly large guidance.

### 4. Multi-task critic

Trained on multiple LIBERO spatial tasks instead of one task.

Purpose:

Improve generalization across task family.

Issue:

If the critic input lacks visual features, multi-task training can still learn progress/action priors rather than scene-aware correctness.

### 5. Task-language features

Added token hashing and VLM hidden feature options.

Purpose:

Let one critic distinguish different task instructions.

### 6. Retrieval progress target

Added dense target from nearest successful states.

Purpose:

Avoid failed trajectories being all zero.

### 7. Uncertainty gate / ensemble

Trained multiple critics and weakened guidance when they disagree.

Purpose:

Reduce bad guidance under uncertainty/OOD.

### 8. Transformer critic

Added Transformer architecture over state/task/action tokens.

Purpose:

Better model temporal structure inside action chunks.

### 9. Counterfactual branch data

Tested restoring simulator state and executing alternative branches.

Purpose:

Teach Q which action is better from the same state.

### 10. Mixed MSE + ranking training

Combined rollout success-to-go regression and counterfactual ranking loss.

Purpose:

Keep trajectory value learning while adding action preference learning.

## Current Weakest Link

The biggest current weakness is not only architecture or beta. It is input information.

Current critic often sees:

```text
robot/preprocessed state + action chunk + optional language
```

but not:

```text
raw image or strong visual embedding
object-relative geometry
detected target-object position
current gripper-object relation
```

Therefore, it can score whether an action resembles successful trajectories, but may fail to know whether the action is appropriate for this exact image.

## Best Next Algorithmic Upgrade

The cleanest next step is a vision-aware action critic:

```text
Q(image_embedding, robot_state, language_embedding, action_chunk) -> scalar
```

Suggested architecture:

```text
[CLS]
visual token from front camera
visual token from wrist camera
robot-state token
language token
action token 0
...
action token H-1
-> Transformer Encoder
-> scalar Q
```

Training:

```text
MSE success-to-go / progress target
+ counterfactual ranking loss
+ optional conservative/OOD regularization
```

Evaluation:

```text
baseline SmolVLA
state-only QGF
vision-aware QGF
counterfactual vision-aware QGF
```

This would directly address the "critic does not really look at the image" problem.

