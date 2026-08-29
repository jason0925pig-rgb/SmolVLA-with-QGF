# 红色包裹出箱:数据清洗后复训与 Orin 部署结果(2026-08-29)

按 `docs/SMOLVLA_NEW_50_A800_TRAINING_AND_ORIN_HANDOFF.md`(2026-08-28 修订版,含
第 2.1 节数据清洗、第 4.3 节 15 Hz 序列评测、附录 A 硬性防线)执行。

任务文本(冻结,逐字):

```text
把箱子里的红色包裹拿出来放到桌子上。
```

**全程未对机器人发出任何上电、使能、伺服、夹爪或运动命令。** 冒烟为 gRPC 回环推理。

## 1. 为什么要重训:J5 的 2π 分支混用

对上一版训练数据逐关节体检时发现,红色包裹任务的 J5 存在**两套等价坐标**:

```
J5 全集范围  [-2.329, +5.121]   跨度 7.45 rad ≈ 2.37π
  其中 19 条(原 ep50-68)录在 [+4.078, +5.121]
      31 条(原 ep69-99)录在 [-2.329, -1.643]
  正分支 -2π 后 = [-2.206, -1.162],与负分支重叠 -> 同一物理姿态
单条 episode 内部没有混用,是采集期间某个时点整体切换
```

作为对照,水杯与订书机任务的 J5 跨度分别只有 1.02 / 1.30 rad,不存在此问题。
这正是附录 A 第一条列出的失效模式:同一物理姿态被模型看成两个完全不同的输入。

## 2. 数据清洗(生成派生副本,原件只读)

```text
~/red_parcel_raw/lerobot_dataset/                     # 只读原始副本(444/555)
~/parcel_smolvla/20260828_red_parcel_clean_50/
├── clean_lerobot_v3/                                 # 本轮训练唯一输入
├── split_manifest.json
└── manifest/{cleaning_report.json, episode_decisions.csv}
```

原始副本经 a800 内网直传,150 个文件大小逐一比对一致后设为只读,全程未修改。

逐条流水线(第 2.1 节七步):

| 步骤 | 实现 |
| --- | --- |
| 15 Hz 时间轴 | 由 action 时间戳建立;视频保持 30 FPS,不重采样 |
| 结构检查 | state/action 均 8 维、时间戳严格递增、无 NaN/Inf |
| 夹爪事件 | 仅用示教 action 原值:`<=0.15` 开、`>=0.85` 闭、中间保持,连续 5 个 15 Hz 动作确认 |
| 截断 | `t_close` → 其后首个 `t_open` → `t_cut = t_open + 7.0 s`,按真实时间戳取最后一行 |
| 重建 | parquet + 双路 MP4(`-frames:v` 精确帧数,`-g 2`)+ episodes/tasks/info/stats 全量一致 |
| 记录 | 每条输出原/新时长、原/新动作数、双路视频帧数与 FPS、`t_close`/`t_open`/`t_cut`、翻转次数、SHA256 |

**J5 分支处理**:把 19 条正分支 episode 整体 −2π,统一到多数(负)分支。
这是**采集值的表示对齐,不是安全包络改写** —— 物理姿态不变,没有做任何 canonicalize
到守卫范围的操作,受影响 episode 逐条记在 `cleaning_report.json` 的 `j5_shifted` 字段。

### 清洗结果

| 项 | 值 |
| --- | --- |
| 保留 / 拒绝 | **50 / 0** |
| 帧数 | 48,178 |
| **J5 清洗后范围** | **[-2.329, -1.162],跨度 1.167 rad,全部单一分支** |
| 有限性 / 单调性 / 8 维 | 全部通过 |
| 抽查 5 组视频 | 帧数 = parquet 行数,30 fps 无倍速(如 ep0:1513 帧 @30fps = 50.43 s,t_cut = 50.40 s) |
| 划分 | seed 1000,train 45 / val [9, 24, 29, 40, 49];归一化统计只用 45 条训练集 |

## 3. 训练

| 项 | 值 |
| --- | --- |
| 机器 / 卡 | a800new(bm-220awn5),**GPU 1** |
| 环境 | Python 3.10.12,lerobot 0.4.4(与 Orin 推理端同版本),torch 2.10.0+cu128 |
| 初始化 | 官方 `lerobot/smolvla_base`(revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205`);未从任何本项目旧 checkpoint 继续训练 |
| 超参 | 20,000 steps,batch 64,seed 1000,save_freq 1000,num_workers 8 |
| 训练集限定 | `--dataset.episodes` 显式传 45 条,日志实测 `num_episodes=45`、`num_frames=43041` |
| 结果 | 最终训练 loss **0.004** |

## 4. 15 Hz 序列评测(第 4.3 节)

在 5 条 held-out 上按部署方式回放:**每 25 个 tick 重规划一次,其间执行 chunk 内的后续动作**
(而不是每拍都用真值观测重新推理)。夹爪过滤器与部署一致:`open<=0.15`、`close>=0.85`、
连续 5 拍确认;接触保持逻辑离线无接触信号,报告标注为 not simulatable offline。

五个候选 checkpoint(2k/5k/10k/15k/20k)**全部 `offline_pass=True`**:无 2π 跳变、
首 chunk 未越出训练数据范围。选中 **020000**(关节 MAE 最低)。

### 与上一版(未清洗)的同口径对照

同一套评测脚本、同样 replan=25、同样 5 条验证 episode:

| 指标 | 上一版(未清洗) | **本版(清洗后)** | 变化 |
| --- | ---: | ---: | --- |
| 7 关节 MAE | 0.0562 | **0.0451** | **↓ 20%** |
| 关节 max error | 0.7280 | 0.6783 | ↓ |
| **首 chunk 最大相邻差** | 0.0308 – 0.1111 | **0.0092 – 0.0223** | **平滑约 5 倍** |
| 首次闭合误差 | 0.63 s | 0.65 s | 持平 |
| 夹爪过滤后 F1 | 0.907 | 0.898 | 持平 |
| 首次张开误差 | 1.64 s | 3.05 s | **见下** |
| 漏张开率 | 0 | 0 | — |

**首 chunk 相邻差缩小 5 倍是本次清洗最直接的收益**,对应附录 A 里"首动作异常"的失效模式。

**关于首次张开误差变差,如实说明**:该均值被单条离群拉高。

```
本版逐条 open_err:  ep9 -0.87s  ep24 -1.20s  ep29 -11.67s  ep40 -1.47s  ep49 +0.07s
上一版同一条 ep29:  +1.00s(翻转 2 次);本版 ep29 翻转 6 次
```

剔除 ep29 后本版平均 0.90 s,优于上一版同口径的 1.94 s。也就是说 4/5 条改善、
**ep29 一条明显退步(提前约 11.7 秒松爪,且夹爪反复开合)**。该条属于早松手风险,
必须列为上机首跑的重点观察项,不能因为总体指标改善而忽略。

## 5. Orin 部署(不覆盖任何旧 bundle)

```text
/home/nvidia/work/telop/models/smolvla_20260828_red_parcel_clean/
├── checkpoint/            # 020000(选中)
├── checkpoint_last/       # 020000(末步,与 best 相同)
├── vlm/  deployment/
├── split_manifest.json / manifest/ / validation_reports/
├── start_policy_server.sh
└── SHA256SUMS
```

- a800new ↔ Orin 逐文件 SHA256 校验一致。
- 旧水瓶、上一版红包裹、水杯、订书机的 bundle 与 profile **均未改动**。
- **部署副本已本地化**:`config.json` 的 `vlm_model_name` 与 `policy_preprocessor.json` 内
  `tokenizer_processor` 的 `tokenizer_name`(位于 steps 列表内,容易漏改)均改为 bundle 内
  `vlm/` 绝对路径;运行时需 `HF_HUB_OFFLINE=1`。
- 按附录 A 要求,部署脚本在结束时打印 profile 名、绝对 bundle/checkpoint 路径、精确 prompt
  和 checkpoint SHA256,供 ARM 前人工核对。

## 6. 上机(需现场人员执行)

ARM / MOVE / 上电 / 使能必须由能触达实体急停的现场人员完成。

```bash
cd ~/work/telop/SmolVLA-with-QGF
export SMOLVLA_ORIN_BUNDLE=/home/nvidia/work/telop/models/smolvla_20260828_red_parcel_clean
export SMOLVLA_SERVER_MODEL_PATH=$SMOLVLA_ORIN_BUNDLE/checkpoint
export SMOLVLA_EXPECTED_CHECKPOINT=$SMOLVLA_ORIN_BUNDLE/checkpoint
export QGF_RUN_MODE=baseline QGF_EPISODE_COUNT=5
export QGF_DATASET_ROOT=/home/nvidia/work/telop/red_parcel_clean_real_rollouts
export SMOLVLA_TASK_B64=$(echo -n "把箱子里的红色包裹拿出来放到桌子上。" | base64 -w0)
export SMOLVLA_CANONICALIZE_POLICY_OBSERVATION=false
./tools/run_qgf_collection_session.sh
```

**首跑重点观察**:

1. **早松手** —— 第 4 节 ep29 显示该模型在部分情形下会提前约 11.7 秒张开夹爪。
   若在搬运途中松爪,立即 END 并记录。
2. 起始位姿 envelope 与完成位姿仍是旧任务参数;本轮未从清洗数据重新标定新姿态范围,
   报错属预期,须现场处理,不得为此放宽全局阈值。
3. `SMOLVLA_CANONICALIZE_POLICY_OBSERVATION=false` —— 训练数据已统一到负分支,
   运行时必须保留原始坐标喂给模型,不要开启 canonicalize。

## 7. 产物

| 产物 | 位置 |
| --- | --- |
| 只读原始副本 | a800new `~/red_parcel_raw/lerobot_dataset` |
| 清洗数据集 | a800new `~/parcel_smolvla/20260828_red_parcel_clean_50/clean_lerobot_v3` |
| 清洗报告 / 逐条决策 | 同上 `manifest/`(本仓库 `docs/task_reports/red_clean_*`) |
| 训练输出 20 个 checkpoint | 同上 `outputs/train_red_clean/` |
| 序列评测报告 | 同上 `validation_reports/seq_*.json` |
| Orin bundle | `/home/nvidia/work/telop/models/smolvla_20260828_red_parcel_clean/` |
| 清洗 / 评测 / 部署脚本 | 本仓库 `tools/a800_task_training/` |
