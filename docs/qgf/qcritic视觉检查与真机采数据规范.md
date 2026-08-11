# Q Critic 视觉检查与真机采数据规范

这份文档回答两个问题：

1. 我们现在的 SmolVLA + QGF 里面，Q critic 到底有没有真正用好视觉？
2. 如果现在要上真机采 30 组 / 50 组数据训练 Q critic，每个 episode 应该保存什么？

## 结论先说

SmolVLA 本身是有视觉部分的。它作为 VLA 模型，正常输入包括：

```text
相机图像 + 机械臂状态 + 自然语言指令
```

然后输出：

```text
未来一段 action chunk
```

但是，我们当前接到 QGF 里的 Q critic，严格说并没有直接吃原始 RGB 图像。当前 critic 主要吃的是：

```text
observation.state
+ action_chunk
+ 可选 task language feature
```

所以它更像：

```text
Q(机械臂/预处理状态, 可选语言特征, 未来动作片段) -> 这个动作片段的分数
```

而不是：

```text
Q(当前画面, 机械臂状态, 语言指令, 未来动作片段) -> 这个动作片段的分数
```

因此你同学说的那个担心是合理的：当前 critic 可能能判断“这条轨迹像不像成功轨迹”“动作序列是不是走到了某个阶段”，但它未必真的知道“在当前这张画面里，这个动作是不是对的”。

## 代码证据

当前采 rollout 的脚本是：

```text
scripts/collect_rollouts.py
```

它在每一步保存的是：

```python
state = policy_batch[OBS_STATE].detach().to("cpu")[0]
policy_action = policy.select_action(policy_batch)
action = postprocessor(policy_action)
```

最后每个 episode 存成：

```python
episode = {
    "state": torch.stack(states),
    "action_policy": torch.stack(policy_actions),
    "action_env": torch.stack(env_actions),
    "reward": torch.tensor(rewards),
    "success": torch.tensor(successes),
    "done": torch.tensor(dones),
    "task": task_desc,
    "task_group": task_group,
    "task_id": int(task_id),
    "seed": int(seed),
    "policy_path": str(args.policy_path),
    "obs_feature_kind": "policy_preprocessed_observation.state",
}
```

也就是说，旧采集脚本没有把每一帧 RGB 图像作为 critic 训练样本保存进去。

训练 critic 的数据构造文件是：

```text
src/guided_action_flow/training/critic_dataset.py
```

默认读取：

```text
obs_key = "state"
action_key = "action_policy"
```

构造出来的训练样本是：

```text
obs_features[start]
action_policy[start : start + action_horizon]
target[start]
```

QGF 推理时的接入文件是：

```text
src/guided_action_flow/policies/smolvla_qgf.py
```

里面 `_latest_state(batch)` 返回的是：

```python
batch["observation.state"]
```

然后传给 critic。

所以当前 QGF runtime 里，critic 没有直接看到：

```text
observation.images.*
```

它看到的是：

```text
observation.state
```

## 那这是不是说明完全没视觉？

要分开说：

```text
SmolVLA policy 本身：有视觉。
Q critic 当前实现：没有直接视觉输入。
QGF 引导时：critic 主要靠 state/action/language 打分。
```

如果 `observation.state` 里面包含了一些环境状态或物体相关信息，那 critic 可能间接知道一点场景信息。但从我们当前代码来看，不能说它是一个真正视觉 grounding 的 critic。

更准确地说：

```text
当前 Q critic 是 state/action/language critic，不是 image/state/language/action critic。
```

## 当前 Critic 能学到什么

它能学到：

```text
机械臂在某些状态下，某类 action chunk 更容易成功；
某个任务指令对应某些动作模式；
某个动作序列看起来像成功轨迹的中后段；
动作是不是像专家/成功 rollout 里的动作；
当前状态离过去成功轨迹中的状态近不近。
```

所以它不是完全没用。它可以做“轨迹进度判断”和“动作先验判断”。

## 当前 Critic 不擅长什么

它不擅长：

```text
物体位置变了以后，判断动作是否还对；
相机视角变了以后，判断当前动作是否还对；
背景/光照/遮挡变化下判断目标物；
同样机械臂姿态下，因为物体位置不同，需要采取不同动作；
判断夹爪是不是对准了当前画面里的目标物。
```

这就解释了为什么在 LIBERO-Plus 这种扰动平台里，特别是物体布局、视角、背景这些变化时，Q critic 容易不稳定。

## 怎么验证 Critic 有没有看视觉

因为当前 critic 根本没有图像输入，所以最简单的视觉打乱实验其实会暴露问题：

```text
把图像换掉 -> 当前 Q critic 的输入不变 -> Q 分数基本不会变
```

这不是因为它视觉鲁棒，而是因为它没看图像。

更完整的检查方式：

| 检查 | 做法 | 如果 Q 变化很小说明什么 |
| --- | --- | --- |
| 换图像 | 当前 state/action/language 不变，只换 RGB | 当前 critic 没直接用视觉 |
| 换语言 | state/action 不变，换 task token | 语言条件很弱 |
| 换 action | state/language 不变，换 action chunk | critic 是否主要看 action |
| 换 state | action/language 不变，换 robot state | critic 是否主要看机械臂状态 |
| 遮挡目标物 | 对 vision critic 使用 mask 图像 | 真视觉 critic 应该明显变化 |

## 真机采数据时最重要的原则

不要只保存视频，也不要只保存最终 success/fail。

应该保存：

```text
每一帧看到什么
每一帧机器人在哪里
SmolVLA 当时预测了什么动作
真实执行了什么动作
执行后发生了什么
最后这个 episode 成功还是失败
```

Q critic 学的是：

```text
在当前 observation 下，这个 action chunk 值不值得做。
```

所以 observation、action chunk、outcome 三者都必须存。

## 每个 Episode 必须保存什么

### Episode 级元数据

```text
episode_id
task_name
language_instruction
robot_name
gripper_type
policy_name
policy_checkpoint
control_hz
action_horizon
camera_names
calibration_version
collection_mode: autonomous / teleop / mixed
start_time
end_time
final_success: 0/1
failure_reason
human_intervention: true/false
```

其中 `language_instruction` 很关键，比如：

```text
put the water bottle into the cardboard box
```

以后训练多任务 critic 时，语言就是区分任务的重要条件。

### 每一步必须保存

| 数据 | 必须程度 | 用处 |
| --- | --- | --- |
| timestamp | 必须 | 对齐图像、状态、动作 |
| front camera RGB | 必须 | 训练视觉 critic |
| wrist camera RGB | 强烈建议 | 判断夹爪和物体局部关系 |
| joint positions | 必须 | 机械臂 proprioception |
| joint velocities | 建议 | 判断运动趋势 |
| end-effector pose | 必须 | 比纯关节角更直观 |
| gripper state | 必须 | 判断抓取/释放 |
| SmolVLA predicted action chunk | 必须 | Q 要打分的对象 |
| executed action | 必须 | 真实执行可能被限幅/平滑 |
| measured next state | 必须 | 看到动作执行结果 |
| reward / success signal | 如果有就存 | 训练 target |
| done / truncated | 必须 | episode 边界 |
| intervention flag | 建议 | 标记人类介入 |

## 需要一帧一帧打 success/fail 标签吗

不需要。

最基础训练只需要：

```text
每个 episode 最后一个 success/fail 标签
```

比如：

```json
{
  "success": true
}
```

然后程序会自动把这条 episode 里的每一个时间步变成 success-to-go target。

但如果你能额外标阶段，会更好。

比如把“水瓶放进纸箱子”标成：

```text
0.0 还没接近
0.2 正在接近瓶子
0.4 夹爪对准瓶子
0.6 已经抓住瓶子
0.8 已经移动到箱子上方
1.0 成功放入箱子
```

这样失败轨迹也不是全 0，critic 更容易学到“差一点成功”和“完全失败”的区别。

## predicted action 和 executed action 都要存吗

要，两个都要。

原因是：

```text
SmolVLA predicted action chunk
```

是模型原始想做的动作。

```text
executed action
```

是真机实际执行的动作。

它们可能因为这些原因不一样：

```text
安全限幅
速度限制
控制器平滑
坐标系转换
夹爪命令后处理
人工急停/介入
```

训练 QGF 时，通常 Q 打分的是 policy action chunk；但分析真机失败时，必须知道真实执行了什么。

## 未来 50 Steps 的预测轨迹要不要存

要存，但叫法更准确应该是：

```text
predicted_action_chunk_t:t+H
```

其中 `H` 是 action horizon，比如：

```text
H = 50
```

每个决策时刻都保存：

```text
当前图像 image_t
当前机器人状态 state_t
语言 instruction
SmolVLA 预测的未来 H 步 action chunk
真实执行的动作
执行后的状态变化
```

这正是 action-chunk critic 需要的数据。

## 推荐数据目录结构

```text
real_robot_dataset/
  put_bottle_in_box/
    episode_000001/
      metadata.json
      result.json
      video.mp4
      cameras/
        front/
          000000.jpg
          000001.jpg
          ...
        wrist/
          000000.jpg
          000001.jpg
          ...
      arrays/
        timestamps.npy
        joint_positions.npy
        joint_velocities.npy
        ee_pose.npy
        gripper_state.npy
        predicted_action_chunks.npy
        executed_actions.npy
        measured_next_states.npy
      annotations.json
```

## 推荐 metadata.json

```json
{
  "episode_id": "put_bottle_in_box_episode_000001",
  "task_name": "put_bottle_in_box",
  "language_instruction": "put the water bottle into the cardboard box",
  "robot": "real_robot_name",
  "gripper": "parallel_jaw",
  "policy": "SmolVLA",
  "policy_checkpoint": "checkpoint/path/or/huggingface_id",
  "control_hz": 10,
  "action_horizon": 50,
  "cameras": ["front", "wrist"],
  "calibration_version": "2026-xx-xx",
  "collection_mode": "autonomous_smolvla",
  "notes": ""
}
```

## 推荐 result.json

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

## 30/50 组数据怎么采比较合理

不要让 50 组全成功，也不要全失败。

比较理想是：

```text
成功率 30% - 70%
```

原因：

```text
全成功 -> Q 不知道坏动作是什么；
全失败 -> Q 不知道好动作是什么；
有成功有失败 -> Q 才能学会区分。
```

如果水瓶放纸箱太简单，就增加扰动：

```text
换水瓶初始位置
换角度
换箱子位置
换背景
换视角
换瓶子大小
```

如果太难，就先简化：

```text
箱子放近一点
瓶子固定位置
减少遮挡
只做抓起而不是放入
```

## 对真机最推荐的下一版 Critic

建议做两个版本对照：

### 版本 A：State-only Critic

```text
Q(state, language, action_chunk)
```

这就是接近当前 LIBERO 代码的版本。

### 版本 B：Vision-aware Critic

```text
Q(image_embedding, state, language_embedding, action_chunk)
```

建议结构：

```text
[CLS]
front_image_token
wrist_image_token
robot_state_token
language_token
action_token_0
action_token_1
...
action_token_H
-> Transformer Encoder
-> scalar Q
```

这样才能真正回答你同学指出的问题：Q 到底是不是在看画面。

## 可以写进汇报里的结论

当前阶段可以这样对导师说：

```text
SmolVLA base policy 本身是视觉语言动作模型，会用图像预测动作；
但我们目前接入 QGF 的 critic 主要使用 observation.state、action chunk 和可选语言特征，
没有直接使用 raw RGB 或视觉 embedding。
因此当前 Q 更像轨迹进度/动作先验 critic，
在物体布局、视角、背景扰动下可能无法判断动作是否适合当前画面。
下一步真机采数据时，我们需要保存每帧图像、机器人状态、预测 action chunk、
真实执行 action 和最终成功标签，进一步训练 vision-aware action critic。
```

