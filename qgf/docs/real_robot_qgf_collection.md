# QGF 真机数据采集规范与一键采集

## 论文真正要求的数据

Q-GuidedFlow 的离线 Critic 数据基本单位是转移：

```text
(state_t, action_t, next_state_t, reward_t, done_t)
```

论文在 OGBench 中直接使用仿真 state，不要求视频。Armstrong 的瓶子、箱子和抓取阶段无法只靠七轴角度判断，因此本项目额外保留胸部和右腕 RGB；这是视觉真机 Critic 的工程需求，不是论文原文规定的视频格式。

每个保留 episode 生成：

```text
episode_000000/
├── transitions.parquet
├── normalized_policy_chunks.parquet
├── policy_observations.parquet
├── chest.mp4
├── wrist_right.mp4
├── chest.mjpeg
├── wrist_right.mjpeg
├── episode_metadata.json
├── capture_summary.json
├── launcher_result.json
├── recorder.log
└── samples.jsonl
```

`transitions.parquet` 包含：

- `state` / `next_state`：七轴弧度和夹爪闭合量；
- `action_policy`：SmolVLA 原始目标；
- `policy_action_timestep`：该动作在 LeRobot 异步动作队列中的真实执行序号；
- `action_guarded`：通过安全边界和夹爪状态机后的目标；
- `action_executed`：底层已接受的机械臂、夹爪命令；
- `reward/success/terminated/truncated/done`；
- 两个视频的帧号与时间戳；
- task prompt 和夹爪接触状态。

`normalized_policy_chunks.parquet` 直接保存 SmolVLA
`predict_action_chunk()` 在 postprocessor 之前的归一化输出，供 QGF 在模型动作空间训练与求梯度。`transitions.parquet` 中的动作则是发送到真机控制链的物理单位目标，两者不能混为一谈。

`policy_observations.parquet` 保存每次向策略服务器发请求时的七轴、夹爪、时间戳及两路相机帧号。它与 normalized chunk 都带有 `observation_timestep`，必须用这个字段连接，不能仅按文件行号或墙钟时间猜测对应关系。

异步执行可能只消费一个 50 步预测块的一部分，随后就被更新的块覆盖。因此 IQL 训练的行为动作序列应按 `policy_action_timestep` 和实际执行顺序重建，再用冻结 checkpoint 的 processor 转回归一化空间；不能假定每个预测的 50 步都完整执行过。

两份 `.mjpeg` 是长度前缀的原始 ROS JPEG 档案，记录格式为重复的
`<uint64 little-endian timestamp_ns><uint32 little-endian jpeg_size><jpeg bytes>`。
它们避免 MP4 二次压缩，后续可以用冻结的 SmolVLA/VLM 重新计算两路 RGB visual features。MP4 主要用于快速检查和人工复核。

论文 OGBench 实验使用较短 action chunk；学长的 `guided-action-flow` SmolVLA 实现则明确以 50 步完整 chunk 训练和引导。本采集器同时保存逐步 transition 和 postprocessor 前的完整 normalized chunk，因此两种 horizon 都能构造，当前 SmolVLA 真机实验默认按 50 步接口。

## 需要采多少条

论文没有给出真实机器人“必须 N 条”的数字。其 OGBench 单任务实验使用约 1 亿条 transition，压力实验达到 10 亿条；这些仿真规模不能照搬到真机。

本项目建议按阶段推进：

1. 先采 20–30 条，只验证同步、标签、视角和训练代码；不能据此宣称 Critic 已有稳定泛化。
2. 第一版可用 Critic 以 100–150 条为目标，成功率不要接近 0% 或 100%，最好让成功/失败各自至少约 30%。
3. 想做更可靠的视觉 QGF，建议 300–500 条，并主动变化瓶子/箱子位置、光照和失败类型；至少保留约 100 条成功和 100 条失败。
4. 现有 50 条适合作为 proof-of-concept 和数据管线验收，不足以复现论文规模，也不应预期强泛化。

帧数不能替代 episode 多样性。同一轨迹 30 FPS 相邻帧高度相关，不能把 3 万帧当作 3 万次独立试验。

## Windows 一键连续采集

先将本仓库部署到 Orin 并构建好现有 ROS 2 控制工程。现场急停必须触手可及。然后在 Windows PowerShell 中运行：

```powershell
cd E:\AAA__Github_Project\SmolVLA-with-QGF
.\tools\collect_qgf_rollouts.cmd -EpisodeCount 20
```

固定 task 已有默认值，也可显式指定：

```powershell
.\tools\collect_qgf_rollouts.cmd `
  -EpisodeCount 100 `
  -Task "把矿泉水放进纸箱里。"
```

整场只初始化一次：

1. 启动模型、两路相机、控制节点，只输入一次 `ARM` 和 `MOVE`；
2. 每轮自动完成回原位，或输入 `END` 正常结束并进入标签环节；
3. 输入 `S` 保存成功、`F` 保存失败、`D` 作废并物理删除；
4. 重置瓶子和箱子，按 Enter 开始下一轮，不重新加载模型、不重新上电；
5. 只有完成目标数量、主动退出整场或异常安全停止时，才统一退出伺服、解除使能并下电。

Ctrl+C、`ABORT`、实体急停、保护停、控制器错误、策略门异常关闭或 client 崩溃都会删除当前轮的 MP4、精确 JPEG 档案、Parquet 前的原始样本和 staging 目录，然后关闭整套系统。异常轮次不占编号。只有明确输入 `S` 或 `F` 才进入正式数据集。

## 标签限制

当前 reward 是 episode 级稀疏奖励：成功轨迹最后一步为 1，其余为 0；失败轨迹全为 0。这与学长 IQL 脚本的 sparse terminal-success 接口兼容，但样本效率不高。后续若加入“接近瓶子、稳定抓住、移动到箱上方、释放”等阶段标签或自动奖励，Critic 会更容易学习。

数据集还会保存 checkpoint 的 JSON/YAML preprocessing 配置、权重文件 SHA256 和当前项目 commit。以后做 visual feature 或动作归一化时必须使用同一个 checkpoint；不能只保留数据而删除模型权重。这里保存的是可长期复用的源数据，但 RGB critic 训练前仍需执行一次确定性的“JPEG 解码 → 与 SmolVLA 完全相同的 resize/normalize → 冻结视觉编码器 → visual feature/token”预处理；这不需要重新采真机数据。
