# 水杯放紫盒任务:SmolVLA 训练与 Orin 部署结果(2026-08-27)

按 `docs/SMOLVLA_NEW_50_A800_TRAINING_AND_ORIN_HANDOFF.md` 的通用规则执行(45/5 划分、
统计只由训练集产生、离线 15 Hz 夹爪时序验证、词典序选点)。本任务为单阶段演示,
不需要该文件第 2 节的截断处理。

任务文本(冻结,逐字):

```text
把水杯放到紫色的箱子上
```

**全程未对机器人发出任何上电、使能、伺服、夹爪或运动命令。** 冒烟测试为 gRPC 回环推理。

## 1. 数据

| 项 | 值 |
| --- | --- |
| 源 | Orin `onearm_Tele/lerobot_dataset` 的 `episode_index` 100–149(连续 50 条) |
| 传输 | Orin → Windows 笔记本(纯中转)→ a800new,tar 流式;100 个 MP4 + 50 个 parquet 校验齐全 |
| 重建数据集 | `mug_purple_box_lerobot_v3`,重新编号 0–49,**61,990 帧**,无截断 |
| 划分 | seed 1000,validation = 新编号 **[9, 24, 29, 40, 49]**,train 45 条 |
| 归一化统计 | 由 **45 条训练集**经 lerobot `aggregate_stats` 聚合;验证集不参与 |
| 加载核验 | lerobot 0.4.4 `LeRobotDataset` 实测:50 集 / 61,990 帧 / 30 fps,8 维 state/action,双相机,无 NaN |

原始数据未修改;源文件 SHA-256 记录在 `source_readonly_manifest/source_files_sha256.json`。

## 2. 训练

| 项 | 值 |
| --- | --- |
| 机器 / 卡 | a800new(bm-220awn5),**GPU 3**(`CUDA_VISIBLE_DEVICES=3`) |
| 环境 | Python 3.10.12,lerobot **0.4.4**(与 Orin 推理端同版本),torch 2.10.0+cu128 |
| 初始化 | 官方 `lerobot/smolvla_base`(revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205`),features 改写为 chest/wrist_right/state[8]/action[8];**未从任何本项目旧 checkpoint 继续训练** |
| 超参 | 20,000 steps,batch 64,seed 1000,**save_freq 1000**(共 20 个 checkpoint),num_workers 8 |
| 训练集限定 | `--dataset.episodes` 显式传入 45 条训练编号,日志实测 `num_episodes=45` |
| 结果 | 最终训练 loss **0.005** |

## 3. 离线验证(仅 5 条 validation)

20 个 checkpoint 全部做轻量验证;7 个候选(4 个里程碑 + 轻量最优 3 个)做 15 Hz 回放。
回放使用与部署端相同的夹爪过滤器:`open<=0.40`、`close>=0.85`、连续 5 个 15 Hz tick 确认。
接触保持逻辑离线无接触信号,报告中标注为 **not simulatable offline**。

选中 checkpoint **020000** 的指标:

| 指标 | 值 |
| --- | --- |
| 首次闭合时刻误差 | **0.00 s** |
| 首次张开时刻误差 | **0.00 s** |
| 漏张开率 / 提前张开率 | **0 / 0** |
| 过滤后闭合 F1 | **1.000** |
| 夹爪开合翻转次数 | 每条 2 次(一开一合,无抖动) |
| 7 关节 MAE | **0.01517 rad**(≈0.87°) |
| 原始夹爪 MAE | 0.0048 |
| validation flow loss | 0.0424 |

### 选点依据

按交接文档第 5 节的词典序规则:①输出有限有效 ②漏张开+提前张开最低 ③夹爪 F1 最高
④事件时刻误差最小 ⑤flow loss 与 7 关节 MAE。

前四项上 002000/003000/005000/010000/015000/020000 **完全并列**(全部 0 误差、F1=1.0)。
交接文档第 4 项把 "validation flow loss 和 7 关节 MAE" 并列列出、未规定先后,本次
**以 7 关节 MAE 为准**,理由记录如下:flow matching 的 validation loss 每次评估重新采样
噪声,跨 checkpoint 不具可比性(实测随训练步数单调上升 0.025→0.042);7 关节 MAE 是
确定性指标,直接反映动作精度,实测随训练单调下降(0.024→0.015)。据此选出 020000。
该判据同样适用于本批其余任务,选点脚本见 `tools/a800_task_training/select_ckpt.py`。

## 4. Orin 部署

```text
/home/nvidia/work/telop/models/smolvla_20260827_mug_purple_box/
├── checkpoint/            # 020000(选中)
├── checkpoint_last/       # 020000(末步,本次与 best 相同)
├── vlm/                   # SmolVLM2,自旧 bundle 硬链接
├── deployment/
├── split_manifest.json / preprocessing_report.json / validation_reports/
├── start_policy_server.sh
└── SHA256SUMS
```

- a800new ↔ Orin 校验一致(摘要 `1d81030f5aff3252`)。
- 旧水瓶、旧包裹 bundle 与 profile **未做任何改动**。
- 新 profile:`tools/smolvla_mug_env.sh`,仅预设 `SMOLVLA_ORIN_BUNDLE` 后转交公共模板;
  15 Hz 与 50-action chunk 沿用模板默认。
- **部署副本已本地化**:`config.json` 的 `vlm_model_name` 与 `policy_preprocessor.json` 内
  `tokenizer_processor` 的 `tokenizer_name`(位于 steps 列表内,易漏)均改为 bundle 内
  `vlm/` 绝对路径,运行时需 `HF_HUB_OFFLINE=1`。a800new 上的原始 checkpoint 未做此改写。

## 5. 离线冒烟(不接触硬件)

```text
SMOLVLA_POLICY_SERVER_SMOKE_OK  requests=3  actions=(50, 8)
latency_s = [1.79(首次), 0.86, 0.84]
```

chunk 形状 (50, 8),数值有限;热推理 0.84 s,15 Hz 下 50 步覆盖 3.33 s,异步余量约 4 倍;
全程 `HF_HUB_OFFLINE=1`,无外网访问;未启动任何 ROS 硬件节点。

## 6. 上机(需现场人员执行)

模型与软件已就绪。ARM / MOVE / 上电 / 使能必须由能触达实体急停的现场人员操作。

```bash
cd ~/work/telop/SmolVLA-with-QGF
export SMOLVLA_ORIN_BUNDLE=/home/nvidia/work/telop/models/smolvla_20260827_mug_purple_box
export QGF_RUN_MODE=baseline QGF_EPISODE_COUNT=5
export SMOLVLA_TASK_B64=$(echo -n "把水杯放到紫色的箱子上" | base64 -w0)
export QGF_NOTES_B64=$(echo -n "mug_purple_box; ckpt=020000" | base64 -w0)
./tools/run_qgf_collection_session.sh
```

首跑注意:任务完成位姿与初始位姿 envelope 仍是旧任务参数,新值需按本任务实测标定;
首跑出现起始位姿拒绝或自动完成不触发属预期,须现场处理。
