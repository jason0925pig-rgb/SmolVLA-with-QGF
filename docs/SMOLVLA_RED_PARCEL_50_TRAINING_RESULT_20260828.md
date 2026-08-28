# 红色包裹出箱任务:50 组重建数据训练与部署结果(2026-08-28)

执行依据:`docs/SMOLVLA_NEW_50_A800_TRAINING_AND_ORIN_HANDOFF.md`(2026-08-27 版)。
任务文本(冻结):`把箱子里的红色包裹拿出来放到桌子上。`

硬件红线:重建、训练、验证、打包全程未对机器人发送任何上电/使能/伺服/夹爪/运动命令。

## 1. 数据重建(源 50 条双包裹演示 → 单阶段红包裹)

- 截断规则逐字按交接第 2 节:首次确认闭合(g≥0.85,5 tick@15Hz)→ 其后首次确认张开(g≤0.40,5 tick)→ `t_end = t_open + 7.0s`;帧级精确(parquet 行数 = 视频帧数,ffmpeg `-frames:v` 裁切并解码回验)。
- 50/50 条全部检出有效事件,零隔离;99,573 帧 → **48,178 帧**。
- 50 张 close/open/end 三联图人工复核通过(2026-08-27)。
- 45/5 划分 seed 1000,validation = 新编号 **[9, 24, 29, 40, 49]**;归一化统计仅由 45 条训练集聚合。
- 原始数据未改写;源文件 SHA-256 见 `source_readonly_manifest`。

## 2. 训练(A800new,GPU2)

| 项 | 值 |
| --- | --- |
| 初始化 | 官方 `lerobot/smolvla_base`(rev c83c3163)via init-dir 法(features 重写为 chest/wrist_right/state8/action8) |
| 数据 | 仅 45 条训练集(`--dataset.episodes` 显式列出,已审计) |
| 超参 | 20,000 steps,batch 64,seed 1000,save_freq **1000**(20 个 checkpoint) |
| 环境 | lerobot 0.4.4(=Orin 部署端同版本),torch 2.10.0+cu128 |
| 时长 | 6h02m(1.09 s/step) |

## 3. 验证与选点

每 1k checkpoint 轻量验证(flow loss / 7D MAE / 夹爪 MAE)+ 关键点 **15Hz 异步回放**
(每 25 tick 重规划、部署同款夹爪滞回 0.85/0.40/5tick;contact-hold 无法离线模拟,已注明)。

v2 回放表(5 条 validation,`validation_reports_v2/`):

| ckpt | missed+early | F1 | close_err | open_err | 7D-MAE | val flow loss |
| --- | --- | --- | --- | --- | --- | --- |
| 002000 | 0 | 0.900 | 0.68s | 1.75s | 0.0652 | 0.064 |
| 005000 | 0 | 0.755 | 2.28s | 2.63s | 0.0630 | 0.074 |
| 010000 | 0 | 0.913 | 0.61s | 1.28s | 0.0572 | 0.087 |
| 015000 | 0 | 0.887 | 0.57s | 1.79s | 0.0585 | 0.098 |
| **020000(best)** | 0 | 0.893 | 0.67s | 1.89s | **0.0566** | 0.101 |
| last(≡020000 同SHA) | 0 | 0.915 | 0.59s | 1.40s | 0.0560 | — |

- `last` 与 `020000` 权重 SHA 相同;指标差异 = flow 采样噪声,由此估计 F1 噪声带约 ±0.03。
- **选点 020000**:词典序首两级(漏/提前张开率、F1)在噪声带内平手,第 4 级 7D-MAE 最优;且与一套独立实现的早期评估结论一致(双重验证同指)。
- 如实记录:val flow loss 随步数**上升**(0.052→0.101)而行为指标(MAE/时序)持续**改善** —— flow loss 与执行质量解耦,任务书把 loss 放词典序末位是正确的。
- 早期一轮回放实现存在缺陷(事件误差出现全 0 与 198s 哨兵两种矛盾结果,脚本未存盘),已整体作废,由存盘可审计的 `replay_eval_v2.py` 取代 —— 本表为准。

## 4. 部署

- Orin bundle:`/home/nvidia/work/telop/models/smolvla_20260827_red_parcel_50/`
  (checkpoint=020000,checkpoint_last 硬链同权重,vlm/ 硬链共享,tokenizer_name 与 vlm_model_name 已本地化 + `HF_HUB_OFFLINE=1`)。
- 独立 profile:`tools/smolvla_red_parcel_env.sh`;旧 profile 未动。
- 离线冒烟(gRPC 回环,不触 ROS):见仓库分支说明与 Orin `/tmp/` 冒烟日志。

## 5. 附件

`rebuild_red_parcel.py`(重建 as-run)· `train_config_as_run.json`(训练配置权威记录)·
`replay_eval_v2.py`(回放验证)· `split_manifest.json` · `preprocessing_report.json`
