# 真机 Q Critic 训练事实报告（截至 2026-08-19）

本文只说明已经由**实际训练产物、训练日志和现有代码**证实的内容。它不是对 Guided Action Flow 论文的泛化复述，也不把未保存的训练环境信息当作事实。

## 1. 结论先行

本次训练没有微调 SmolVLA，也没有训练一个新的动作策略（actor）。训练的是一个单独的、视觉条件化的 **Q critic**：

```text
输入：当前 8 维机器人状态 + 双路相机视觉 token + 一个 SmolVLA 提议的 50 步归一化动作块
输出：该动作块在当前观测下的估计 Q 值
```

训练方法是离线 IQL（Implicit Q-Learning）的 Q/V 形式。部署时，原始 SmolVLA 仍负责生成动作块；critic 只在 SmolVLA 的 flow-matching 去噪过程中对动作块求 `grad_a Q`，以 `1 / beta` 为系数改变去噪速度。当前部署的是 **1 个 critic**，没有 uncertainty/disagreement gate。

## 2. 已训练的具体模型

### 2.1 训练对象

| 项目 | 已证实内容 |
|---|---|
| 训练对象 | 视觉 Transformer Q critic 和配套 value model |
| 部署对象 | 只加载 critic 的 `model_state_dict`；value model 不参与在线推理 |
| SmolVLA | 视觉编码器被冻结；SmolVLA 本身未在这次 IQL 中更新参数 |
| critic 数量 | 1 个（`critic_member_00.pt`） |
| 不确定性门 | 禁用。单 critic 无成员间分歧；运行配置 `uncertainty_scale=0.0` |
| critic 输入状态 | 8 维：右臂 7 个关节弧度 + `gripper_closed`（0/1） |
| critic 动作 | `50 x 8` 的 SmolVLA **归一化**动作块：7 个关节通道 + 夹爪通道 |
| critic 视觉输入 | 头/胸部相机（代码名 `chest`）与右腕相机（`wrist_right`）各一帧，经冻结的 SmolVLA 视觉编码器后拼接为 `[128, 960]` token |

critic 网络为 3 层 Transformer Encoder，`d_model=256`、4 个 attention heads、dropout 0.1。它把状态、视觉 token 和 50 个动作 token 一起编码，再从 CLS token 输出一个标量 Q 值。

代码依据：

- [视觉 critic 网络](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF/blob/main/qgf/src/guided_action_flow/critics/visual_transformer_critic.py)
- [checkpoint 加载逻辑](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF/blob/main/qgf/src/guided_action_flow/critics/checkpoint.py)
- [真机 QGF policy server](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF/blob/main/lerobot_robot_armstrong_ros2/src/lerobot_robot_armstrong_ros2/policy_server_qgf.py)

## 3. 真机训练数据到底是什么

### 3.1 使用了哪些 episode

保存的 manifest 表明训练使用 `episode_000017` 到 `episode_000116`，共 100 个真实机器人 rollout：

| 项目 | 数值 |
|---|---:|
| 原始 episode 数 | 100 |
| 成功 episode | 47 |
| 失败 episode | 53 |
| 对齐后的动作块样本 | 8,917 |
| 训练 episode | 90 |
| 验证 episode | 10 |
| 训练样本 | 7,970 |
| 验证样本 | 947 |
| 训练集中正奖励动作块样本 | 94 |
| 验证集中正奖励动作块样本 | 10 |

划分在 **episode 层面**完成，而不是随机拆帧：同一 episode 的图像、状态和动作块不会同时出现在训练集和验证集。划分按记录的 `success/failure` outcome 分层，随机种子为 `20260814`。

验证 episode 为：`30, 38, 39, 40, 51, 60, 66, 78, 93, 112`。其余 90 个编号在训练集中，完整编号保存于训练产物的 `episode_split_90_10.json` 与 `training_summary.json`。

### 3.2 每个原始 episode 必须包含什么

数据构建脚本要求每个 episode 至少有：

```text
episode_metadata.json
transitions.parquet
normalized_policy_chunks.parquet
policy_observations.parquet
chest.mp4
wrist_right.mp4
```

原始 recorder 也保存 `chest.mjpeg`、`wrist_right.mjpeg`，但本次视觉特征提取实际解码的是两个 MP4，而不是 MJPEG archive。

代码依据：

- [episode 采集器](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF/blob/main/tools/qgf_episode_recorder.py)
- [episode finalize 与 success/reward 写入](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF/blob/main/tools/finalize_qgf_episode.py)
- [数据对齐 manifest 构建](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF/blob/main/qgf/scripts/build_real_robot_visual_iql_manifest.py)

### 3.3 三种“动作”中，Q 实际学习哪一个

原始 `transitions.parquet` 同时记录：

| 字段 | 含义 | 本次 critic 是否把它作为训练动作输入 |
|---|---|---|
| `action_policy` | 物理后处理前的策略目标记录 | 否 |
| `action_guarded` | 经过任务边界与夹爪时序处理后的目标 | 否 |
| `action_executed` | 底层已接受的手臂/夹爪命令 | 否 |
| `normalized_policy_chunks.action_chunk_normalized` | SmolVLA sampler 直接输出的 50 步归一化动作块 | **是** |

这是有意设计：QGF 在线对归一化 flow action 求导，因此训练也必须在同一归一化动作空间中进行。真实机器人实际执行后的状态、奖励和终止信息仍然用于构造监督目标，但 Q 的动作输入不是 `action_executed`。

### 3.4 时间和视觉对齐

1. 每条 normalized policy chunk 有一个 policy observation timestep。
2. 从相同 observation timestep 中选带有胸部和右腕有效帧索引、且与 chunk 时间最接近的观测，作为 `state` 和当前双路图像。
3. 50 步动作块按 `15 Hz` 对应 `50 / 15 = 3.333... s` 预测跨度。
4. 在 `transitions.parquet` 中寻找约 3.333 秒后的真实 `next_state`；起点匹配允许的最大时间差是 0.1 秒。
5. 如果这段跨度内到达 terminal transition，则提前截断到 terminal transition。

两路 MP4 均按 30 FPS 保存，但 critic 的一个样本只取与 policy observation 对齐的一对 RGB 帧；训练动作块频率仍是 15 Hz。因此不是把每一张 30 FPS 图像都当成一个独立动作样本。

本次 manifest 丢弃了 887 个没有对齐 policy observation 的 chunk，以及 4 个起始 transition 时间差超过 0.1 秒的 chunk，最终保留 8,917 个样本。

## 4. reward、success、done 是怎样构造的

episode 结束时由操作流程写入 `outcome=success` 或 `outcome=failure`。

- 成功 episode 的**最后一个** transition：`reward=1`、`success=true`、`terminated=true`、`done=true`。
- 失败 episode 的最后一个 transition：`reward=0`、`success=false`、`truncated=true`、`done=true`。
- 中间 transition：`reward=0`、`done=false`。

对一个 50 步动作块，manifest 在其时间跨度内聚合：

```text
reward = 该跨度内所有 transition reward 之和
success = 该跨度内是否出现 success
done / terminated / truncated = 该跨度内是否出现对应终止标记
```

训练时再次强制：`done = done OR terminated OR truncated OR success`，并令 `reward = max(reward, success)`。因此一个成功轨迹末端前、其 3.33 秒 horizon 覆盖到终点的若干 chunk 都可能成为正奖励样本；这解释了 47 个成功 episode 对应 104 个正奖励动作块样本，而不是只有 47 个。

## 5. 视觉特征如何得到

本次不是直接训练像素 CNN。流程为：

```text
chest.mp4 RGB frame + wrist_right.mp4 RGB frame
        ↓
冻结的、部署用 SmolVLA 图像编码器与 connector
        ↓
拼接的 BF16 visual tokens，形状 [128, 960]
        ↓
保存为每个 episode 一个 .pt feature cache
        ↓
视觉 Transformer critic
```

视觉编码器仅用于特征提取；本次 IQL 训练不会更新它。在线 QGF 运行时会用当前 policy observation 的同一双路图像重新提取 token，并检查 token 形状是否与 checkpoint 一致。

代码依据：

- [双路视觉特征提取](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF/blob/main/qgf/scripts/extract_smolvla_visual_features.py)
- [在线视觉 token 与 critic adapter](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF/blob/main/qgf/src/guided_action_flow/policies/smolvla_qgf.py)

## 6. IQL 的实际优化目标

设一个样本为 `(s, z, a, r, s', z', d)`：

```text
s, s'：8 维状态及其 next state
z, z'：双路视觉 token 及其 next token
a：50 x 8 的归一化 SmolVLA 动作块
r：稀疏终止奖励
d：终止标记
```

训练包含三个网络实例：在线 `Q(s,z,a)`、冻结更新的 target Q、以及 `V(s,z)`。

```text
Value loss:
  L_V = E[ w(Q_target(s,z,a) - V(s,z)) * (Q_target(s,z,a) - V(s,z))^2 ]
  w = 0.7，当 Q_target - V > 0；否则 w = 0.3

Q target:
  y = r + 0.99 * (1-d) * V(s',z')

Q loss:
  L_Q = MSE(Q(s,z,a), y)
```

每个优化 step 后 target Q 使用 Polyak 更新：

```text
target_Q ← (1 - 0.005) * target_Q + 0.005 * Q
```

优化器是两个 AdamW（一个 Q、一个 V），梯度范数裁剪到 10。GPU 上的前向/目标计算使用 BF16 autocast；损失计算转为 FP32。

这属于**离线强化学习 / offline RL**：只学习评价数据集中已经出现过的状态—动作块，不在训练过程中让机器人探索，也不训练一个新的 actor。

代码依据：[IQL 训练脚本](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF/blob/main/qgf/scripts/train_real_robot_visual_iql.py)。

## 7. 已执行训练的有效配置与产物

以下来自 A800 服务器训练目录的 `training_summary.json` 与 `train_single_qcritic_gpu3.log`：

```text
/ssd/hanbo/TNNLS_2026/work/armstrong_qgf_visual_iql_20260817/
```

| 参数 | 实际值 |
|---|---:|
| ensemble size | 1 |
| epochs | 80 |
| batch size | 16 |
| learning rate | 3e-4 |
| weight decay | 1e-4 |
| gamma | 0.99 / 每个 50-step chunk |
| expectile | 0.7 |
| Polyak | 0.005 |
| d_model / layers / heads / dropout | 256 / 3 / 4 / 0.1 |
| split seed | 20260814 |
| selected checkpoint epoch | **14**（不是最后的 epoch 80） |
| selected validation TD loss | **0.008718575622333446** |
| checkpoint | `outputs/real_17_116_single_qcritic/critic_member_00.pt` |
| checkpoint 文件大小 | 21,942,625 bytes |

epoch 14 的验证日志还记录：

```text
val_q_success_mean        = 0.9699218869
val_q_failure_mean        = 0.4785531461
val_q_success_failure_gap = 0.4913687408
val_positive_reward_samples = 10
val_samples = 947
```

这些值是 critic 的 TD/Q 诊断指标，不是成功率、分类准确率、AUC 或独立真机评测成绩，不能把它们解释为“critic 在验证集上有某百分比准确率”。

## 8. QGF 在线时实际做什么

在线运行时流程是：

```text
当前双路 RGB + 当前 8 维机器人状态
        ↓
SmolVLA 产生正在去噪的 normalized action chunk x_t 与速度 v_t
        ↓
估计 clean action：a_hat = x_t - t * v_t
        ↓
对 a_hat 求 grad_a Q(s, z, a_hat)
        ↓
guided_velocity = v_t - grad_a Q / beta
        ↓
SmolVLA 完成去噪，Orin 仍负责安全门、限位、速度/加速度与实际控制
```

当前真机 QGF server 要求加载 visual-transformer checkpoint，且设置：

```text
beta = 用户运行时给定值
实际 Q 引导系数 = 1 / beta
grad_clip_norm = 1.0
uncertainty_scale = 0.0
```

例如 `beta=2` 时，实际系数是 `0.5`。QGF server 本身不含机器人 SDK、上电、使能或 ROS 电机控制；这些仍由 Orin 上已有的 attended ROS client/safety stack 完成。

代码依据：

- [Q guidance 公式实现](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF/blob/main/qgf/src/guided_action_flow/guidance/qgf.py)
- [SmolVLA 去噪 hook](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF/blob/main/qgf/src/guided_action_flow/policies/smolvla_qgf.py)
- [QGF policy server](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF/blob/main/lerobot_robot_armstrong_ros2/src/lerobot_robot_armstrong_ros2/policy_server_qgf.py)
- [真实 rollout 启动流程](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF/blob/main/tools/run_qgf_collection_session.sh)

## 9. 严格限制与不能如实声称的内容

1. 训练目录是复制到 A800 的工作目录，那里没有 `.git` 元数据。因此仅凭 A800 训练产物**不能验证训练时源代码对应哪个 Git commit**。本报告给出的 GitHub 文件是当前仓库中的实现路径；其结构和日志中的文件路径相符，但不应声称已由 commit hash 完全复现。
2. `training_summary.json` 只记录 `device="cuda"`。日志名是 `train_single_qcritic_gpu3.log`，但训练产物没有保存 `nvidia-smi` 快照，因此不能只凭产物证明物理使用的是哪一张 A800。
3. 此训练没有留出独立的最终 test set。10 个验证 episode 用于选择 epoch 14；它们不是完全不参与模型选择的最终测试集。
4. QGF 后的 episode（从 117 开始）不在此次 critic 的 17–116 训练范围内；但 17–116 本身包含后续 medium baseline 对比中使用的 baseline episode。因此“medium baseline vs QGF”的结果是现实部署结果，不应被写成与 Q 训练集完全独立的无泄漏测试。
5. 该 critic 学的是数据中 SmolVLA 已提出过的 normalized action chunk 的价值估计。它并未通过本次训练学会从零生成动作，也不保证对远离训练分布的物体、相机位姿、机械臂状态或光照可靠。
6. 训练只用胸部和右腕两路 RGB 视觉特征；没有深度图、点云、力/扭矩传感器或夹爪真实力值作为 critic 输入。

## 10. 可追溯文件清单

| 用途 | GitHub 文件 |
|---|---|
| 采集双相机、状态、三种动作记录、policy chunk | [tools/qgf_episode_recorder.py](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF/blob/main/tools/qgf_episode_recorder.py) |
| 生成 `transitions.parquet`、奖励和终止标签 | [tools/finalize_qgf_episode.py](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF/blob/main/tools/finalize_qgf_episode.py) |
| 将真实 episode 对齐为训练样本并做 90/10 划分 | [qgf/scripts/build_real_robot_visual_iql_manifest.py](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF/blob/main/qgf/scripts/build_real_robot_visual_iql_manifest.py) |
| 从两路 MP4 提取冻结 SmolVLA visual token | [qgf/scripts/extract_smolvla_visual_features.py](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF/blob/main/qgf/scripts/extract_smolvla_visual_features.py) |
| IQL critic 训练、验证和 checkpoint 选择 | [qgf/scripts/train_real_robot_visual_iql.py](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF/blob/main/qgf/scripts/train_real_robot_visual_iql.py) |
| 视觉 Transformer critic 结构 | [qgf/src/guided_action_flow/critics/visual_transformer_critic.py](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF/blob/main/qgf/src/guided_action_flow/critics/visual_transformer_critic.py) |
| checkpoint 装载 | [qgf/src/guided_action_flow/critics/checkpoint.py](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF/blob/main/qgf/src/guided_action_flow/critics/checkpoint.py) |
| `grad_a Q / beta` guidance | [qgf/src/guided_action_flow/guidance/qgf.py](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF/blob/main/qgf/src/guided_action_flow/guidance/qgf.py) |
| 将 visual critic 接进 SmolVLA 去噪 | [qgf/src/guided_action_flow/policies/smolvla_qgf.py](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF/blob/main/qgf/src/guided_action_flow/policies/smolvla_qgf.py) |
| 真机 QGF policy server | [lerobot_robot_armstrong_ros2/.../policy_server_qgf.py](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF/blob/main/lerobot_robot_armstrong_ros2/src/lerobot_robot_armstrong_ros2/policy_server_qgf.py) |

## 11. 训练产物位置

训练输出不在 Git 仓库中，当前保存于 A800：

```text
/ssd/hanbo/TNNLS_2026/work/armstrong_qgf_visual_iql_20260817/
├── data/real_17_116_visual_iql/
│   ├── aligned_normalized_chunks.parquet
│   ├── episode_split_90_10.json
│   └── manifest_summary.json
├── features/                         # 每个 episode 的视觉 token .pt cache
├── outputs/real_17_116_single_qcritic/
│   ├── critic_member_00.pt           # 部署 Q checkpoint
│   ├── training_input_summary.json
│   └── training_summary.json
└── train_single_qcritic_gpu3.log
```

部署端通过环境变量 `SMOLVLA_QGF_CRITIC_PATH` 指向 `critic_member_00.pt`。这个路径必须是部署机器实际可读的副本；A800 上的训练路径不会自动出现在 Orin 上。
