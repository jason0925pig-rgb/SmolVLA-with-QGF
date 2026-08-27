# 红色包裹出箱任务:SmolVLA 训练与 Orin 部署结果(2026-08-28)

按 `docs/SMOLVLA_NEW_50_A800_TRAINING_AND_ORIN_HANDOFF.md`(2026-08-27)执行,包括该文件
第 2 节的单阶段截断、第 3 节的固定 45/5 划分与第 5 节的 15 Hz 夹爪时序验证。

任务文本(冻结,逐字):

```text
把箱子里的红色包裹拿出来放到桌子上。
```

**全程未对机器人发出任何上电、使能、伺服、夹爪或运动命令。** 冒烟测试为 gRPC 回环推理。

## 1. 数据重建:首次"闭合 → 张开"后保留 7 秒

从双包裹分拣的 50 条原始演示(源 `episode_index` 50–99)重建单阶段数据集。

| 项 | 值 |
| --- | --- |
| 截断规则 | 首次确认闭合(`gripper>=0.85`,连续 5 个 15 Hz tick)→ 其后首次确认张开(`<=0.40`,连续 5 tick)→ `t_end = t_open + 7.0 s` |
| 检测结果 | **50/50 全部有效**,无隔离条目 |
| 帧数 | 99,573 → **48,178**(裁掉 51.6%,即绿色包裹阶段) |
| 帧级一致 | 视频以 `ffmpeg -frames:v` 精确裁切,每条解码到最后一帧核验 = parquet 行数 |
| 人工复核 | 50 张三联复核图(close / open / end 三个时刻)已由人工逐条确认 |
| 任务文本 | 全部 50 条重写为新文本;原双包裹文本仅保留在 source manifest 中溯源 |
| 划分 | seed 1000,validation = 新编号 **[9, 24, 29, 40, 49]**,train 45 条 |
| 归一化统计 | 由 **45 条训练集**聚合;验证集不参与任何统计 |

原始数据未修改;源文件 SHA-256 记录在 `source_readonly_manifest/source_files_sha256.json`。

### 一个必须记录的编码问题

首次裁切使用 x264 默认 GOP(约 250 帧一个关键帧),导致训练时随机取帧需要从最近关键帧
解码上百帧,dataloader 被拖垮:**8.84 s/step**,20k 步需要 47 小时。原始录制视频是隔帧
关键帧编码(随机访问友好)。改用 `-g 2` 从**源视频**重新裁切(仍是单次有损)后恢复到
**1.13 s/step**。重建脚本 `tools/a800_task_training/recut_videos_g2.py` 已固定该参数。
此后所有裁切类任务都应保留 `-g 2`。

## 2. 训练

| 项 | 值 |
| --- | --- |
| 机器 / 卡 | a800new(bm-220awn5),**GPU 2** |
| 环境 | Python 3.10.12,lerobot **0.4.4**(与 Orin 推理端同版本),torch 2.10.0+cu128 |
| 初始化 | 官方 `lerobot/smolvla_base`(revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205`),features 改写为 chest/wrist_right/state[8]/action[8];**未从任何本项目旧 checkpoint 继续训练** |
| 超参 | 20,000 steps,batch 64,seed 1000,**save_freq 1000**(20 个 checkpoint),num_workers 8 |
| 训练集限定 | `--dataset.episodes` 显式传 45 条训练编号,日志实测 `num_episodes=45`、`num_frames=43041` |
| 结果 | 最终训练 loss **0.004** |

## 3. 离线验证(仅 5 条 validation)

20 个 checkpoint 做轻量验证;7 个候选做 15 Hz 回放,使用与部署端相同的夹爪过滤器
(`open<=0.40`、`close>=0.85`、连续 5 tick 确认)。接触保持逻辑离线无接触信号,
报告标注为 **not simulatable offline**。

选中 checkpoint **020000**:

| 指标 | 值 |
| --- | --- |
| 首次闭合时刻误差 | **0.00 s** |
| 首次张开时刻误差 | **0.00 s** |
| 漏张开率 / 提前张开率 | **0 / 0** |
| 过滤后闭合 F1 | **1.000** |
| 7 关节 MAE | **0.04452 rad**(≈2.6°) |
| validation flow loss | 0.1010 |

完整排序(词典序规则,见 `tools/a800_task_training/select_ckpt.py`):

```
020000  missed+early=0.00  F1=1.000  evt_err=0.00s  jmae=0.04452
015000  missed+early=0.00  F1=1.000  evt_err=0.00s  jmae=0.04709
002000  missed+early=0.00  F1=1.000  evt_err=0.00s  jmae=0.05340
010000  missed+early=0.00  F1=0.998  evt_err=0.05s  jmae=0.04262
003000  missed+early=0.00  F1=0.979  evt_err=0.48s  jmae=0.05639
001000  missed+early=0.00  F1=0.974  evt_err=0.59s  jmae=0.06226
005000  missed+early=0.00  F1=0.920  evt_err=1.72s  jmae=0.04832
```

005000 的关节 MAE 并不差,但夹爪事件误差 1.72 s、F1 仅 0.920,被判据正确排到末位 ——
这正是交接文档要求"不能只看 loss 或逐帧夹爪准确率"的原因。

交接文档第 5 节第 4 项把 "validation flow loss 和 7 关节 MAE" 并列、未规定先后;
本批统一**以 7 关节 MAE 为准**:flow matching 的 validation loss 每次评估重采噪声,
跨 checkpoint 不可比;关节 MAE 是确定性指标,直接反映动作精度。

## 4. Orin 部署

```text
/home/nvidia/work/telop/models/smolvla_20260827_red_parcel_out_table/
├── checkpoint/            # 020000(选中)
├── checkpoint_last/       # 020000(末步,与 best 相同)
├── vlm/  deployment/
├── split_manifest.json / preprocessing_report.json / validation_reports/
├── source_readonly_manifest/
├── start_policy_server.sh
└── SHA256SUMS
```

- a800new ↔ Orin 校验一致(摘要 `bab7fde4bee6939b`)。
- 旧任务的 bundle 与 profile **未做任何改动**。
- 新 profile:`tools/smolvla_red_parcel_env.sh`,仅预设 `SMOLVLA_ORIN_BUNDLE`;
  15 Hz、50-action chunk 沿用公共模板默认值。
- **部署副本已本地化**:`config.json` 的 `vlm_model_name` 与 `policy_preprocessor.json` 内
  `tokenizer_processor` 的 `tokenizer_name`(位于 steps 列表内,易漏)均改为 bundle 内
  `vlm/` 绝对路径;运行时需 `HF_HUB_OFFLINE=1`。

## 5. 离线冒烟(不接触硬件)

```text
SMOLVLA_POLICY_SERVER_SMOKE_OK  requests=3  actions=(50, 8)
latency_s = [1.78(首次), 0.87, 0.84]
```

chunk 形状 (50, 8),数值有限;热推理 0.84 s,15 Hz 下 50 步覆盖 3.33 s,异步余量约 4 倍;
全程离线,未启动任何 ROS 硬件节点。

## 6. 上机(需现场人员执行)

```bash
cd ~/work/telop/SmolVLA-with-QGF
export SMOLVLA_ORIN_BUNDLE=/home/nvidia/work/telop/models/smolvla_20260827_red_parcel_out_table
export QGF_RUN_MODE=baseline QGF_EPISODE_COUNT=5
export SMOLVLA_TASK_B64=$(echo -n "把箱子里的红色包裹拿出来放到桌子上。" | base64 -w0)
export QGF_NOTES_B64=$(echo -n "red_parcel_out_table; ckpt=020000" | base64 -w0)
./tools/run_qgf_collection_session.sh
```

ARM / MOVE / 上电 / 使能必须由能触达实体急停的现场人员操作。首跑注意:任务完成位姿与
初始位姿 envelope 仍是旧任务参数,新值需按本任务实测标定。
