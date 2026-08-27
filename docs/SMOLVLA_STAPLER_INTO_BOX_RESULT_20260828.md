# 订书机入盒任务:SmolVLA 训练与 Orin 部署结果(2026-08-28)

按 `docs/SMOLVLA_NEW_50_A800_TRAINING_AND_ORIN_HANDOFF.md` 的通用规则执行(45/5 划分、
统计只由训练集产生、离线 15 Hz 夹爪时序验证、词典序选点)。本任务为单阶段演示,
不需要该文件第 2 节的截断处理。

任务文本(冻结,逐字):

```text
把订书机放进快递纸盒
```

**全程未对机器人发出任何上电、使能、伺服、夹爪或运动命令。** 冒烟测试为 gRPC 回环推理。

## 1. 数据

| 项 | 值 |
| --- | --- |
| 源 | Orin `onearm_Tele/lerobot_dataset` 的 `episode_index` 150–199(连续 50 条) |
| 传输 | Orin → Windows 笔记本(纯中转)→ a800new,tar 流式;100 个 MP4 + 50 个 parquet 齐全 |
| 重建数据集 | `stapler_into_box_lerobot_v3`,重新编号 0–49,**59,933 帧**,无截断 |
| 划分 | seed 1000,validation = 新编号 **[9, 24, 29, 40, 49]**,train 45 条 |
| 归一化统计 | 由 **45 条训练集**经 lerobot `aggregate_stats` 聚合;验证集不参与 |
| 加载核验 | lerobot 0.4.4 实测:50 集 / 59,933 帧 / 30 fps,8 维 state/action,双相机,无 NaN |

原始数据未修改;源文件 SHA-256 记录在 `source_readonly_manifest/source_files_sha256.json`。

## 2. 训练

| 项 | 值 |
| --- | --- |
| 机器 / 卡 | a800new(bm-220awn5),**GPU 3** |
| 环境 | Python 3.10.12,lerobot **0.4.4**(与 Orin 推理端同版本),torch 2.10.0+cu128 |
| 初始化 | 官方 `lerobot/smolvla_base`(revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205`),features 改写为 chest/wrist_right/state[8]/action[8];**未从任何本项目旧 checkpoint 继续训练** |
| 超参 | 20,000 steps,batch 64,seed 1000,**save_freq 1000**(20 个 checkpoint),num_workers 8 |
| 训练集限定 | `--dataset.episodes` 显式传 45 条训练编号 |
| 用时 / 结果 | 4 小时 34 分,最终训练 loss **0.005** |

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
| 7 关节 MAE | **0.01522 rad**(≈0.87°) |
| validation flow loss | 0.0485 |

完整排序:

```
020000  missed+early=0.00  F1=1.000  evt_err=0.00s  jmae=0.01522
015000  missed+early=0.00  F1=1.000  evt_err=0.00s  jmae=0.01562
006000  missed+early=0.00  F1=1.000  evt_err=0.00s  jmae=0.01861
002000  missed+early=0.00  F1=1.000  evt_err=0.00s  jmae=0.02169
010000  missed+early=0.00  F1=0.999  evt_err=0.03s  jmae=0.01636
005000  missed+early=0.00  F1=0.998  evt_err=0.05s  jmae=0.01926
001000  missed+early=0.00  F1=0.993  evt_err=0.15s  jmae=0.02604
```

交接文档第 5 节第 4 项把 "validation flow loss 和 7 关节 MAE" 并列、未规定先后;
本批统一**以 7 关节 MAE 为准**:flow matching 的 validation loss 每次评估重采噪声,
跨 checkpoint 不可比;关节 MAE 是确定性指标,直接反映动作精度。

## 4. Orin 部署

```text
/home/nvidia/work/telop/models/smolvla_20260827_stapler_into_box/
├── checkpoint/            # 020000(选中)
├── checkpoint_last/       # 020000(末步,与 best 相同)
├── vlm/  deployment/
├── split_manifest.json / preprocessing_report.json / validation_reports/
├── source_readonly_manifest/
├── start_policy_server.sh
└── SHA256SUMS
```

- a800new ↔ Orin 校验一致(摘要 `d997683a97f6cd97`)。
- 旧任务(水瓶、双包裹)与本批另两个任务的 bundle、profile **均未改动**。
- 新 profile:`tools/smolvla_stapler_env.sh`,仅预设 `SMOLVLA_ORIN_BUNDLE`;
  15 Hz、50-action chunk 沿用公共模板默认值。
- **部署副本已本地化**:`config.json` 的 `vlm_model_name` 与 `policy_preprocessor.json` 内
  `tokenizer_processor` 的 `tokenizer_name`(位于 steps 列表内,易漏)均改为 bundle 内
  `vlm/` 绝对路径;运行时需 `HF_HUB_OFFLINE=1`。

## 5. 离线冒烟(不接触硬件)

```text
SMOLVLA_POLICY_SERVER_SMOKE_OK  requests=3  actions=(50, 8)
latency_s = [1.75(首次), 0.89, 0.88]
```

chunk 形状 (50, 8),数值有限;热推理 0.88 s,15 Hz 下 50 步覆盖 3.33 s,异步余量约 3.8 倍;
全程离线,未启动任何 ROS 硬件节点。

## 6. 上机(需现场人员执行)

```bash
cd ~/work/telop/SmolVLA-with-QGF
export SMOLVLA_ORIN_BUNDLE=/home/nvidia/work/telop/models/smolvla_20260827_stapler_into_box
export QGF_RUN_MODE=baseline QGF_EPISODE_COUNT=5
export SMOLVLA_TASK_B64=$(echo -n "把订书机放进快递纸盒" | base64 -w0)
export QGF_NOTES_B64=$(echo -n "stapler_into_box; ckpt=020000" | base64 -w0)
./tools/run_qgf_collection_session.sh
```

ARM / MOVE / 上电 / 使能必须由能触达实体急停的现场人员操作。首跑注意:任务完成位姿与
初始位姿 envelope 仍是旧任务参数,新值需按本任务实测标定。

## 7. 本批三个任务的横向对照

| 任务 | 帧数 | 关节 MAE (rad) | 夹爪事件误差 | F1 | Orin 热推理 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 红色包裹出箱 | 48,178 | 0.04452 | 0.00 s | 1.000 | 0.84 s |
| 水杯放紫盒 | 61,990 | 0.01517 | 0.00 s | 1.000 | 0.84 s |
| **订书机入盒** | 59,933 | **0.01522** | 0.00 s | 1.000 | 0.88 s |

三个 checkpoint 相互独立,均从官方 base 全新训练,互不继承。红色包裹任务的关节 MAE
明显更高,与其数据经过截断重建、且原演示为双物体长程任务有关。
