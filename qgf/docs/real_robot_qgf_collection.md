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
├── chest.mp4
├── wrist_right.mp4
├── episode_metadata.json
├── capture_summary.json
├── launcher_result.json
├── recorder.log
└── samples.jsonl
```

`transitions.parquet` 包含：

- `state` / `next_state`：七轴弧度和夹爪闭合量；
- `action_policy`：SmolVLA 原始目标；
- `action_guarded`：通过安全边界和夹爪状态机后的目标；
- `action_executed`：底层已接受的机械臂、夹爪命令；
- `reward/success/terminated/truncated/done`；
- 两个视频的帧号与时间戳；
- task prompt 和夹爪接触状态。

动作 chunk 可在训练时用连续转移重建。论文单任务实验的 Critic 使用 5 步 action chunk；SmolVLA 的 50 步推理队列不等于 Critic 必须使用 50 步输入。

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

每轮流程：

1. 启动被动记录器，此时不控制机器人；
2. 按现有安全启动器提示输入 `ARM`、`MOVE`；
3. 自动完成回原位，或输入 `STOP` / Ctrl+C 结束；
4. 输入 `S` 保存为成功、`F` 保存为失败、`D` 作废并物理删除；
5. 程序自动开始下一轮，保留数据的编号始终连续。

启动失败的轮次会自动删除且不占编号。只有明确输入 `S` 或 `F` 才进入正式数据集。

## 标签限制

当前 reward 是 episode 级稀疏奖励：成功轨迹最后一步为 1，其余为 0；失败轨迹全为 0。它足够建立第一版 IQL/QGF 接口，但样本效率不高。后续若加入“接近瓶子、稳定抓住、移动到箱上方、释放”等阶段标签或自动奖励，Critic 会更容易学习。
