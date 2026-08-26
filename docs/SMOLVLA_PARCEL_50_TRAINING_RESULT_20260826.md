# 新任务 50 组演示:SmolVLA 训练与 Orin 部署结果(2026-08-26)

执行依据:SmolVLA-with-QGF 仓库 `docs/SMOLVLA_NEW_50_A800_TRAINING_AND_ORIN_HANDOFF.md`(2026-08-25)。
本文只记录**已发生并可核验**的事实;所有数字可溯源到文中列出的产物文件。

任务文本(全程逐字一致,已冻结):

```text
把箱子里面的红色包裹放到箱子外面左侧，绿色包裹放到箱子外面右侧
```

硬件红线遵守情况:训练与部署全程(2026-08-25 18:00 – 08-26 01:30)实体急停保持按下,
未发出任何上电、使能、伺服、夹爪或运动命令。冒烟测试为 gRPC 回环推理,不经过 ROS。

## 1. 源数据冻结

- 源:Orin `/home/nvidia/work/telop/onearm_Tele/lerobot_dataset` 中 `episode_index` 50–99(连续,共 50 组)。
- 每组含 parquet 轨迹 + chest/wrist_right 两路 30 FPS MP4;8 维 state/action;逐组校验无 NaN/Inf、时间戳单调。
- manifest:A800 `~/parcel_smolvla/20260825_parcel_50/manifest/MANIFEST_NEW_TASK_50_20260825.json`
  (50 组新旧编号映射、逐组帧数、任务文本)。

## 2. 传输与独立数据集构建

- 传输路径:Orin → Windows 笔记本(纯中转)→ A800,tar 流式;
  两端比对 **404 个文件 / 5,905,315,566 字节完全一致**。
- 在 A800 上抽取 ep50–99 重建**独立 LeRobot v3 数据集**(重新编号 0–49):
  `~/parcel_smolvla/20260825_parcel_50/dataset`,50 episodes / **99,573 帧**。
- 归一化统计**仅由这 50 组**的 per-episode stats 经 lerobot 官方 `aggregate_stats` 聚合生成,
  未复制旧任务统计;`episode_index`/`index`/`task_index` 相关统计已按新编号修正。
- 用 lerobot 0.4.4 `LeRobotDataset` 实际加载抽查(首/中/末帧)通过。

## 3. 训练

| 项 | 值 |
| --- | --- |
| 机器 / 卡 | A800(g0084),**仅物理 GPU 7**(`CUDA_VISIBLE_DEVICES=7`) |
| 环境 | 新建 venv `~/parcel_env`:Python 3.10.12,lerobot **0.4.4**(与 Orin 推理端同版本),torch 2.10.0+cu128 |
| 初始化 | 官方 `lerobot/smolvla_base`(hf-mirror,revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205`)。做法与旧任务一致:复制 base 权重,把 config 的 input/output features 重写为 chest / wrist_right / state[8] / action[8](init 目录法;旧 bundle 的 train_config.json 证实旧任务即此法) |
| 超参 | 20,000 steps,batch 64,seed 1000,save_freq 5000,num_workers 8(照抄旧任务已验证配方) |
| 时长 | 2026-08-25 19:22 → 08-26 00:15(约 4.9 小时,约 1.2 step/s) |
| 未做 | 未混入旧任务数据;未从旧微调权重初始化;未训练 critic;未做 LingBot 转换 |

产物:`~/parcel_smolvla/20260825_parcel_50/outputs/train_20260825_parcel50_smolvla/checkpoints/{005000,010000,015000,020000,last}`,
`train.log`,`manifest/training_provenance.json`(完整 provenance)。

## 4. 离线验证与选点

四个 checkpoint 用 `validate_smolvla_checkpoint.py`(固定帧、GPU7、离线)评估:

| checkpoint | flow loss | 动作 MAE (rad) | 夹爪二值准确率 |
| --- | --- | --- | --- |
| 005000 | 0.0193 | — | — |
| 010000 | 0.0100 | — | — |
| **015000(正选)** | 0.0089 | 0.0188 | **98.31%** |
| 020000(备用) | 0.0082 | 0.0178 | 98.25% |

- 选点由仓库工具 `select_smolvla_checkpoint.py` 完成,规则为夹爪准确率优先 → **015000**;
  与 020000 的差距为 1/1600 个样本,属噪声量级,020000 一并部署为备用。
- 夹爪方向离线核验:`0=打开,1=闭合` 语义正确,未发现方向翻转。
- 局限说明:训练未预留 episode 级验证子集(与旧任务配方保持一致),上表为固定采样的离线指标,
  不能替代真机评估。

## 5. Orin 部署(未覆盖旧模型)

```text
/home/nvidia/work/telop/models/smolvla_20260825_parcel_50/
├── checkpoint/               # 015000 正选(A800↔Orin SHA256 逐文件一致,摘要 d3372f78…)
├── checkpoint_alt_020000/    # 020000 备用(摘要 7db56d1f…)
├── vlm/                      # SmolVLM2,自旧 bundle 硬链接(内容相同,不占额外空间)
├── deployment/               # 代码快照,同上
└── start_policy_server.sh
```

- 新 profile:`SmolVLA-with-QGF/tools/smolvla_new_task_env.sh` —— 仅预设
  `SMOLVLA_ORIN_BUNDLE` 后转交公共模板;旧任务 profile 与旧 bundle **未做任何改动**。
- 15 Hz 异步约定全部沿用模板默认:`SMOLVLA_FPS=15`、`SMOLVLA_ACTIONS_PER_CHUNK=50`、
  相机保持 30 FPS、server 回环 127.0.0.1。
- **重要修正(部署副本)**:`checkpoint*/config.json` 的 `vlm_model_name` 与
  `checkpoint*/policy_preprocessor.json` 内 `tokenizer_processor` 的 `tokenizer_name`
  均已改为 bundle 内 `vlm/` 的**绝对路径**,并要求运行环境带 `HF_HUB_OFFLINE=1`。
  否则 server 端 processor 实例化会尝试联网拉 `HuggingFaceTB/SmolVLM2…` 而失败
  (tokenizer_name 位于 preprocessor JSON 的 steps 列表内,容易漏改)。
  A800 上的原始 checkpoint 未做此改写;再次部署时需同样处理。

## 6. Orin 离线冒烟(不接触硬件)

`smoke_smolvla_policy_server.py`,gRPC 回环(127.0.0.1:18080),数据集快递段帧(index 70000)+ 冻结任务文本:

```text
SMOLVLA_POLICY_SERVER_SMOKE_OK  requests=3  actions=(50, 8)
latency_s = [2.90(首次), 0.86, 0.86]
```

- 输出 chunk 形状 (50, 8),数值全部有限;
- 热推理 0.86 s,15 Hz 下 50 步块覆盖 3.33 s,异步余量约 4 倍(旧任务实测为 1.6–1.8 s);
- 全程 `HF_HUB_OFFLINE=1`,无外网访问。

## 7. 上机(需人在场,急停在手)

模型与软件已就绪;按交接文档第 9 节,ARM / MOVE / 上电 / 使能必须由现场人员执行。
Orin 终端:

```bash
cd ~/work/telop/SmolVLA-with-QGF
export SMOLVLA_ORIN_BUNDLE=/home/nvidia/work/telop/models/smolvla_20260825_parcel_50
export QGF_RUN_MODE=baseline QGF_EPISODE_COUNT=5
export SMOLVLA_TASK_B64=$(echo -n "把箱子里面的红色包裹放到箱子外面左侧，绿色包裹放到箱子外面右侧" | base64 -w0)
export QGF_NOTES_B64=$(echo -n "parcel_task; new_ckpt=015000" | base64 -w0)
./tools/run_qgf_collection_session.sh
```

切换备用 checkpoint:额外 `export SMOLVLA_SERVER_MODEL_PATH=$SMOLVLA_ORIN_BUNDLE/checkpoint_alt_020000`。

**首跑注意**:任务完成位姿与初始位姿 envelope 仍是旧任务参数(交接文档明令不得迁移旧值,
新值需按新任务实测标定)。首跑出现起始位姿拒绝或自动完成不触发属预期,须现场处理;
首次夹爪闭合必须有人盯守。

## 8. 产物索引

| 产物 | 位置 |
| --- | --- |
| 独立数据集(50 组) | A800 `~/parcel_smolvla/20260825_parcel_50/dataset` |
| 训练输出与日志 | A800 `~/parcel_smolvla/20260825_parcel_50/outputs/…`、`train.log` |
| 验证报告与选点 | A800 `~/parcel_smolvla/20260825_parcel_50/reports/` |
| provenance | A800 `~/parcel_smolvla/20260825_parcel_50/manifest/training_provenance.json` |
| Orin bundle | `/home/nvidia/work/telop/models/smolvla_20260825_parcel_50/` |
| 新任务 profile | `SmolVLA-with-QGF/tools/smolvla_new_task_env.sh` |
