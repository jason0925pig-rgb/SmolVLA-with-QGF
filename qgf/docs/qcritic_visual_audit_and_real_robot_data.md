# Q Critic Visual Audit and Real-Robot Data Checklist

This note answers two practical questions:

1. Does the current SmolVLA + QGF critic really use vision?
2. If we collect 30/50 real-robot episodes to train a Q critic, what should be saved?

## Short Answer

SmolVLA itself is a vision-language-action policy. It uses camera images, robot state, and language to predict an action chunk.

Our current Q critic path, however, does not directly consume raw RGB images. In the rollout collector, each training episode saves:

```text
state
action_policy
action_env
reward
success
done
task
task_group
task_id
seed
task_tokens / task_attention_mask, if available
```

The critic dataset then uses:

```text
obs_features = episode["state"]
action_chunk = episode["action_policy"] by default
target = success-to-go / progress-blended target
optional task_features = hashed language tokens or VLM text hidden feature
```

So the current critic is better described as:

```text
Q(robot/preprocessed state, optional language feature, action chunk) -> scalar score
```

not:

```text
Q(RGB image, robot state, language, action chunk) -> scalar score
```

This is why the concern from your classmate is meaningful: the critic can learn "where this trajectory is in the robot/action sequence", but it may not know whether the action is correct for the current visual scene.

## Evidence from Current Code

The rollout collector is `scripts/collect_rollouts.py`.

At each step it stores:

```python
state = policy_batch[OBS_STATE].detach().to("cpu")[0]
policy_action = policy.select_action(policy_batch)
action = postprocessor(policy_action)
```

Then the episode is saved as:

```python
episode = {
    "state": torch.stack(states),
    "action_policy": torch.stack(policy_actions),
    "action_env": torch.stack(env_actions),
    "reward": torch.tensor(rewards),
    "success": torch.tensor(successes),
    "done": torch.tensor(dones),
    "task": task_desc,
    ...
    "obs_feature_kind": "policy_preprocessed_observation.state",
}
```

The critic dataset is `src/guided_action_flow/training/critic_dataset.py`.

By default it reads:

```python
obs_key = "state"
action_key = "action_policy"
```

and creates samples:

```text
obs_features[start]
action_policy[start : start + action_horizon]
target[start]
```

The QGF runtime is `src/guided_action_flow/policies/smolvla_qgf.py`.

During inference it sets:

```python
self.obs_features = _latest_state(batch)
```

where `_latest_state(batch)` returns `batch["observation.state"]`.

Therefore, in the current deployed QGF hook, the critic is not called with image tensors. It is called with the latest state vector, optional task-language feature, and candidate action chunk.

## What This Means

### What the current critic can learn

It can learn correlations such as:

- "When the arm is around this joint/eef pose, this kind of action tends to be useful."
- "This action chunk resembles action chunks from successful rollouts."
- "This task instruction has a different action distribution from another task."
- "This point in the trajectory looks close to the successful part of previous state-action sequences."

This is enough to produce non-trivial ranking and sometimes improve QGF.

### What the current critic is weak at

It may be weak when the important information is visual:

- object moved to a different layout;
- camera viewpoint changes;
- lighting/background changes;
- target object is occluded;
- the same robot pose requires different actions because the object is elsewhere;
- the gripper is near the wrong object but the robot state looks plausible.

In these cases, a critic that mostly sees robot state and action can still rank "good-looking trajectories", but not necessarily "good action for this image".

## Does Our Current Q Critic Really Use Vision?

Current verdict:

```text
SmolVLA policy: yes, uses vision.
Current Q critic: mostly no, at least not directly.
```

More precise wording:

The current critic does not receive raw RGB frames or explicit visual embeddings in the QGF path. It receives `observation.state`, optional language features, and action chunks. If `observation.state` contains some environment-derived object information, the critic may indirectly see some scene state. But in our current collector and trainer, there is no explicit image feature branch. So for real-robot deployment, we should assume the critic is not visually grounded enough.

## Recommended Visual Ablation

To prove this instead of just arguing it, run these checks on a trained critic:

1. Normal input:

```text
Q(state_t, language, action_chunk)
```

2. Shuffle images only:

Not applicable to the current critic if images are not input. This already exposes the problem.

3. Shuffle language tokens:

```text
Q(state_t, shuffled_language, action_chunk)
```

If Q barely changes, language conditioning is weak.

4. Shuffle action chunks:

```text
Q(state_t, language, wrong_action_chunk)
```

If Q changes a lot here, it means Q mainly depends on action shape.

5. Shuffle robot state:

```text
Q(wrong_state, language, action_chunk)
```

If Q changes a lot here, it means Q mainly depends on robot state.

6. Future visual critic test:

After adding image features, compare:

```text
Q(image_t, state_t, language, action_chunk)
Q(shuffled_image, state_t, language, action_chunk)
Q(masked_object_image, state_t, language, action_chunk)
```

A real visual critic should be sensitive to these image changes.

## What to Save on the Real Robot

For 30/50 SmolVLA rollout episodes, save data at every decision step. Do not only save final videos.

### Required Per Episode

```text
episode_id
task_name
language_instruction
operator / autonomous / intervention mode
model_checkpoint
policy_config
qgf_config, if QGF is active
start_time
end_time
final_success: 0/1
failure_reason, optional but very useful
```

### Required Per Step

| Field | Why it matters |
| --- | --- |
| timestamp | Align image, state, command, and measured motion. |
| RGB image from every camera | Needed for a visual critic and debugging. |
| robot joint positions | Basic robot proprioception. |
| robot joint velocities | Helps know motion trend. |
| end-effector pose | Easier for critic to learn geometry than raw joints alone. |
| gripper state | Open/close and grasp timing are critical. |
| SmolVLA predicted action chunk | This is what Q scores. |
| executed action command | Real execution may differ after clipping/smoothing/safety. |
| measured next robot state | Needed to know what actually happened. |
| reward/success signal, if available | Training target source. |
| done/truncated flag | Episode boundary. |

### Strongly Recommended

| Field | Why it matters |
| --- | --- |
| camera intrinsics/extrinsics | Needed for repeatability and 3D reasoning. |
| hand-eye calibration | Needed if using object-relative features. |
| object poses / AprilTag / ArUco / detector boxes | Helps avoid a critic that ignores vision. |
| depth image or point cloud | Useful for real manipulation and object distance. |
| stage labels | Example: approach, grasped, lifted, above box, released. |
| intervention flag | Marks human correction or unsafe action. |
| safety clipping info | Tells whether action was modified before execution. |
| video.mp4 | Fast human inspection and presentation. |

### For Counterfactual Action Critic

If we want a real action critic instead of only a trajectory-progress critic, save branch data:

```text
same initial state S
candidate action chunk A
candidate action chunk B
candidate action chunk C
outcome after executing each branch
```

For real robots, exact simulator-style reset is hard. Practical versions are:

- use teleoperation to return the robot/object to the same approximate state;
- use repeated starts from a fixture;
- use a scripted reset routine;
- use simulation for exact counterfactuals and real robot for validation;
- use human ranking when exact reset is impossible.

### Suggested File Layout

```text
real_robot_dataset/
  task_put_bottle_in_box/
    episode_000001/
      metadata.json
      result.json
      video.mp4
      cameras/
        front/
          000000.jpg
          000001.jpg
        wrist/
          000000.jpg
          000001.jpg
      arrays/
        timestamps.npy
        joint_positions.npy
        joint_velocities.npy
        ee_pose.npy
        gripper.npy
        predicted_action_chunks.npy
        executed_actions.npy
        measured_next_states.npy
      annotations.json
```

### Suggested `metadata.json`

```json
{
  "episode_id": "task_put_bottle_in_box_episode_000001",
  "task_name": "put_bottle_in_box",
  "language_instruction": "put the water bottle into the cardboard box",
  "robot": "real_robot_name",
  "gripper": "parallel_jaw",
  "policy": "SmolVLA",
  "policy_checkpoint": "checkpoint/path/or/hf_id",
  "control_hz": 10,
  "action_horizon": 50,
  "cameras": ["front", "wrist"],
  "calibration_version": "2026-xx-xx",
  "collector": "autonomous_smolvla",
  "notes": ""
}
```

### Suggested `result.json`

```json
{
  "success": true,
  "failure_reason": "",
  "num_steps": 143,
  "grasped_object": true,
  "lifted_object": true,
  "placed_in_target": true,
  "human_intervention": false
}
```

## Do We Need Frame-by-Frame Success Labels?

No, not for the basic Q critic.

Minimum labeling:

```text
one final success/fail label per episode
```

Then training code can convert that to per-step `success_to_go` targets.

Better labeling:

```text
stage/progress labels per step or per segment
```

Example:

```text
0.0 = not started
0.2 = arm approaching object
0.4 = gripper aligned
0.6 = object grasped
0.8 = object above target
1.0 = object placed successfully
```

This helps because failed episodes are no longer all zero. The critic can learn "near success" versus "completely wrong".

## Success/Failure Balance

For critic training, 50 episodes should not be all success or all failure.

Best region:

```text
30% to 70% success rate
```

Why:

- if all success, Q has weak negative examples;
- if all failure, Q has weak positive examples;
- if mixed, Q can learn what makes an action chunk better or worse.

For real robot, this means you may need to tune task difficulty:

- make object placement easier/harder;
- vary object initial pose;
- use shorter tasks first;
- allow mild perturbations but not impossible starts.

## Immediate Recommendation

For the next real-robot dataset, do not repeat the old collector exactly. Save raw camera frames and action chunks. Then train two critics:

1. State-only critic:

```text
Q(state, language, action_chunk)
```

2. Vision-aware critic:

```text
Q(image_embedding, state, language, action_chunk)
```

If the vision-aware critic improves under object-layout changes while the state-only critic does not, that becomes a strong and explainable research result.

