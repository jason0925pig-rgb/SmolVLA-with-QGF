# 水杯放紫盒任务:50 组数据训练与部署结果(2026-08-28)

执行依据:`docs/SMOLVLA_NEW_50_A800_TRAINING_AND_ORIN_HANDOFF.md` 的通用规则
(45/5、seed 1000、统计仅训练集、官方 base、每 1k 验证、词典序选点)。
任务文本(冻结,与采集一致):`把水杯放到紫色的箱子上`

硬件红线:全程未对机器人发送任何上电/使能/伺服/夹爪/运动命令。

## 1. 数据集(源 ep100-149 → 独立 50 集)

- 单阶段任务,**无截断**:整条保留,视频原样复制;重新编号 0-49。
- 61,990 帧;任务文本 50 条逐字一致(构建时断言)。
- 45/5 划分 seed 1000,validation = **[9, 24, 29, 40, 49]**;
  归一化统计仅由 45 条训练集聚合(官方 aggregate_stats)。
- 源文件 SHA-256 manifest 随附;Orin 原始数据未动。

## 2. 训练(A800new,GPU3)

| 项 | 值 |
| --- | --- |
| 初始化 | 官方 `lerobot/smolvla_base`(rev c83c3163),init-dir 法 |
| 数据 | 仅 45 条训练集(`--dataset.episodes` 显式列出,已审计) |
| 超参 | 20,000 steps,batch 64,seed 1000,save_freq 1000(20 checkpoints) |
| 环境 | lerobot 0.4.4,torch 2.10.0+cu128 |
| 时长 | 4h51m(1.14 step/s) |

## 3. 验证与选点

v2 15Hz 异步回放(5 条 validation,存盘脚本 `replay_eval_v2.py`,部署同款夹爪滞回):

| ckpt | missed+early | F1 | close_err | open_err | 7D-MAE |
| --- | --- | --- | --- | --- | --- |
| 002000 | 0 | 0.948 | 0.59s | 1.04s | 0.0360 |
| 005000 | 0 | 0.945 | 0.75s | 0.84s | 0.0337 |
| 010000 | 0 | 0.941 | 0.76s | 1.00s | 0.0323 |
| **015000(best)** | 0 | **0.955** | 0.75s | **0.67s** | 0.0317 |
| 020000 | 0 | 0.953 | 0.73s | 0.72s | 0.0317 |
| last(≡020000 同SHA) | 0 | 0.952 | 0.71s | 0.79s | 0.0313 |

- **选点 015000**:词典序第 2 级(F1 最高)与第 3 级(事件时刻误差最低)双冠;7D-MAE 与最优差 0.0004(噪声量级)。
- 重要更正:一轮早期回放因事件检测缺陷(198s 哨兵)曾误选 002000,该轮已整体作废;
  v2 表中 002000 实际排名靠后。经验教训:验证脚本必须存盘可审计。
- 与红包裹任务一致,val flow loss 随步数升(0.025→0.042)而行为指标持续改善,loss 不用于选点。

## 4. 部署

- Orin bundle:`/home/nvidia/work/telop/models/smolvla_20260827_mug_50/`
  (checkpoint=015000,checkpoint_last=020000 权重,vlm/ 硬链共享,tokenizer/vlm 路径本地化)。
- 独立 profile:`tools/smolvla_mug_env.sh`;其他任务 profile 未动。

## 5. 附件

`build_mug_dataset.py` · `train_config_as_run.json` · `replay_eval_v2.py` ·
`split_manifest.json` · `preprocessing_report.json`
