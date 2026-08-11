# SmolVLA + QGF 单任务真机验证方案

方法与数据接口以学长仓库
[`chenchaosheng24-design/guided-action-flow`](https://github.com/chenchaosheng24-design/guided-action-flow)
中的 Transformer critic、IQL/supervised/hybrid 目标和 50 步 SmolVLA action chunk 为主要实现参照；原始 Q-GuidedFlow 论文用于核对算法定义。真机控制、安全门和双路视觉同步属于本项目的 Armstrong 适配层。

该仓库目前给出的单任务 LIBERO 结果包括一次从 34/50（68%）到 41/50（82%）的提升，但另一 seed 为 41/50 到 43/50。因此它只能作为“值得做真机检验”的依据，不能预先保证 Armstrong 上也提高 14 个百分点。

## 1. 实验目的

只验证一个结论：在相同的 SmolVLA checkpoint、相同任务、相同控制与安全参数下，加入由离线 baseline rollout 训练出的 Q critic 和 Q-GuidedFlow 后，是否提高“把矿泉水放进纸箱里”的真机成功率。

本实验不宣称位置泛化、物体泛化、跨任务泛化或跨机器人泛化。

## 2. 预注册并冻结的内容

正式采集前写进 `experiment_manifest.json`，此后不得根据结果临时改变：

- SmolVLA checkpoint SHA256；
- 固定中文 prompt；
- 瓶子、箱子、机器人底座、相机位置的地面/桌面标记；
- 两路相机分辨率、30 FPS、曝光与白平衡；
- 机器人动作频率、限位、加速度、夹爪状态机；
- 一轮最大时间；
- 成功判据、失败判据和异常重跑规则；
- 随机种子生成规则；
- QGF 候选参数范围和最终测试预算。

正式实验期间冻结 SmolVLA，不再微调 base policy。否则无法把提升归因于 QGF。

## 3. 成功与失败定义

成功必须同时满足：

1. 指定矿泉水瓶完全进入纸箱边界；
2. 夹爪已经释放瓶子；
3. 瓶子没有在释放后弹出或倾倒到箱外；
4. 机械臂回到预定义结束位姿并触发自动完成；
5. 全程没有实体急停、保护停或控制器异常。

其余正常结束均标记失败。实体急停、通信/相机/SDK 故障记为 `technical_abort`，不进入 Q 训练，但必须单独统计。只有满足预先规定的硬件故障条件才允许补跑，不能因为“这次结果不好”而补跑。

建议保存视频后由第二个人按盲法复核标签；复核者不看该轮是 baseline 还是 QGF。

## 4. 固定场景

用胶带或治具固定：

- 瓶子初始位置与朝向；
- 纸箱位置与朝向；
- 机器人初始关节位置；
- 胸部和腕部相机位姿；
- 灯的位置、亮度档位和色温。

“开灯”和“关灯”必须写成可重复条件，例如照度计读数范围，而不是仅凭肉眼描述。若没有照度计，至少拍摄灰卡并记录相机曝光参数。

## 5. 阶段 A：管线试运行，不计入结果

先做 10 轮 baseline：

- 5 轮开灯；
- 5 轮关灯；
- 检查 transition、normalized 50-step chunk、双路 RGB、时间戳、标签和自动删除；
- 检查成功与失败样本是否都能产生正确 terminal reward；
- 检查训练器能读取一整个 episode。

这 10 轮只用于修程序，全部放入 `pilot/`，不得混入正式训练或最终测试。

## 6. 阶段 B：Critic 训练数据

主实验只在开灯固定场景训练 Q：

- 推荐采集 120 轮 baseline；最低可接受为 100 轮；
- 按 episode 切分，禁止按帧随机切分；
- 推荐 90 轮 train、30 轮 validation；若只有 100 轮，则 80/20；
- 不使用最终测试轮训练或选择参数；
- 记录自然产生的成功/失败，不人工伪造标签。

如果成功率高于 90% 或低于 10%，Q 很难从单一标签学习有用排序。应在正式实验开始前调整任务难度，使成功和失败都至少约占 20%；调整后重新从头采集，不能把不同难度混在一起。

关灯数据不进入主训练集，使关灯结果成为预先声明的 secondary stress test。若以后决定把关灯也用于训练，必须作为另一项实验重新编号，不能与本实验混报。

## 7. 阶段 C：训练和参数选择

保持训练数据与 checkpoint 固定，先比较：

1. supervised success-to-go critic；
2. pure IQL critic；
3. IQL + supervised auxiliary loss；
4. state-only 与 state + 双路 RGB visual features。

主候选建议使用学长仓库的 Transformer critic + IQL 目标：

```text
V(s) <- expectile(Q_target(s,a))
Q(s,a) <- r_chunk + gamma^k * (1-done) * V(s')
```

仅在 validation 数据上比较 Q-return correlation、成功/失败 value separation、ranking accuracy 和过拟合差距。然后从小范围选择 QGF 参数，例如：

- beta：0.25、0.5、1.0、2.0；
- gradient clip norm：0.25、0.5、1.0；
- expectile：0.7、0.8、0.9；
- ensemble：至少 3 个不同 seed 的 critic；
- uncertainty gate：先关闭，再验证是否降低异常动作。

不要在最终 100/50 测试上反复调 beta。可为最有希望的 2–3 组配置各做 10 轮开灯 validation rollout，选定一个配置后冻结。

## 8. 阶段 D：最终 A/B 测试

最终测试使用全新的 rollout，并在每个小 block 内随机交错 baseline 与 QGF，避免上午/下午温度、光照、机械磨损和操作员熟练度造成偏差。

推荐预算：

| 条件 | Baseline | QGF | 总计 |
| --- | ---: | ---: | ---: |
| 开灯固定场景（primary） | 100 | 100 | 200 |
| 关灯固定场景（secondary） | 50 | 50 | 100 |

例如每两个共享同一复位条件的 trial 构成一对，随机决定顺序为 baseline→QGF 或 QGF→baseline。不要先连续跑完全部 baseline，再隔一天跑全部 QGF。

如果时间不足，最低 proof-of-concept 预算为开灯 50+50、关灯 25+25，但置信区间会明显更宽。

## 9. 必须报告的指标

Primary endpoint：开灯条件下 QGF 相对 baseline 的成功率绝对提升。

同时报告：

- 每组成功数/总数和成功率；
- 成功率差值及 95% confidence interval；
- 配对设计使用 McNemar test；若配对被破坏，则使用 Fisher exact test；
- 每轮完成时间的中位数和四分位区间；
- `technical_abort` 数量及原因；
- 实体急停、保护停、模型 safety guard、SDK failure 次数；
- 推理延迟、action queue 低水位次数；
- Q 值、gradient norm、QGF gate 开启比例和 critic ensemble disagreement；
- 开灯和关灯分别报告，不能只汇总成一个数字。

只有 primary endpoint 在冻结配置和独立测试集上提升，且安全事件没有增加，才能说“本实验支持 QGF 提高该固定任务的成功率”。单纯挑选最好的一组 beta 或展示成功视频不构成证据。

## 10. 推荐目录

```text
qgf_fixed_bottle_box/
├── experiment_manifest.json
├── pilot/
├── critic_data/
│   ├── train/
│   └── validation/
├── tuning_rollouts/
├── final_test/
│   ├── light_on/
│   │   ├── baseline/
│   │   └── qgf/
│   └── light_off/
│       ├── baseline/
│       └── qgf/
├── checkpoints/
├── metrics/
└── report.md
```

每轮必须关联唯一的 `episode_index`、`condition`、`method`、`pair_id`、`order_in_pair`、`seed`、人工标签和技术停止原因。最终分析脚本只能读取预注册字段，不能通过文件名猜实验条件。

## 11. 现场执行顺序

1. 用 `collect_qgf_rollouts.cmd -EpisodeCount 10` 完成 pilot；确认每轮只输入一次整场 `ARM/MOVE`，而 S/F 后仅复位场景。
2. 解码抽查两路 `.mjpeg`，并验证 `policy_observations.parquet` 能按 `observation_timestep` 与 `normalized_policy_chunks.parquet` 一一连接。
3. 清空或隔离 pilot，创建冻结的 `experiment_manifest.json`。
4. 开灯采 120 轮 baseline critic data；S 与 F 都保存，D 只用于操作员明确判定本轮不属于实验的情况。
5. 按 episode 固定 90/30 train/validation；从原始 JPEG 用同一 SmolVLA checkpoint 生成双路 visual tokens/features，并从 `policy_action_timestep` 重建实际执行动作序列。
6. 训练 state-only、双路 RGB supervised、pure IQL、IQL+supervised 四个候选；仅用 validation 和预先预算的少量 tuning rollout 选最终 critic/QGF 参数。
7. 冻结 Q、beta、clip、ensemble 和 gate 后，按随机配对顺序完成开灯 100+100、关灯 50+50 的 baseline/QGF 最终试验。
8. 盲法复核视频标签，运行固定统计脚本，生成成功率、95% CI、显著性检验、安全事件和延迟报告。

当前采集器负责第 1–4 步的原始资料与严谨 episode 边界。第 5 步是离线确定性预处理，不需要重新采集；第 6 步应接入学长仓库的 `train_critic_iql.py`，并在正式训练前先用一条 episode 做端到端 smoke test，禁止采满 120 条后才第一次测试数据加载器。
