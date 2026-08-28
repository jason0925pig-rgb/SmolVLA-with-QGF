# 新任务 50 组演示：SmolVLA 训练与 Orin 部署交接

更新日期：2026-08-28
适用仓库：[SmolVLA-with-QGF](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF)

## 0. 任务边界（先读）

这份交接只处理 Orin 上**新采集的同一任务 50 组主从遥操作演示**，目标是训练一个**全新的 SmolVLA 任务 checkpoint**，再把它部署到 Armstrong Orin 做推理。

- 训练初始化：使用经过核验的 **SmolVLA 官方预训练 base checkpoint**。
- 训练数据：仅使用这 50 组新演示；不可混入以前“把矿泉水放进纸箱里”任务的数据。
- 不可初始化自旧微调权重 `smolvla_onearm_20k_20260805`；该权重属于旧任务。
- 不进行 LingBot-VA 格式转换、TCP/FK 标注、VAE latent 转换或 LingBot adapter 训练。
- 不训练、也不运行 QGF/IQL critic；新任务先以普通 SmolVLA baseline 方式完成闭环。
- 本交接中的 A800/Orin 准备、离线验证**不得**调用机器人上电、使能、伺服、夹爪或运动命令。

如果无法确定“这 50 组”的源目录、任务文本或 episode 编号，必须先停在第 1 节确认；不能猜测或把相邻 episode 混进来。

## 1. 冻结并核验 Orin 上的 50 组源数据

历史 SmolVLA 数据一般位于：

```text
/home/nvidia/work/telop/onearm_Tele/lerobot_dataset
```

但这不是新数据的保证路径。先在 Orin 做只读排查，并把最终选定源目录和 50 个 episode 编号写入 manifest：

```bash
find /home/nvidia/work/telop -type f \( -name 'episodes.jsonl' -o -name 'info.json' \) -print

# 将 <SOURCE_LEROBOT_V3_ROOT> 替换为确认后的目录。
python3 - <<'PY'
import json, pathlib
root = pathlib.Path('<SOURCE_LEROBOT_V3_ROOT>')
for name in ('info.json', 'episodes.jsonl', 'tasks.jsonl'):
    p = root / 'meta' / name
    print('\n===', p, '===')
    print(p.read_text(encoding='utf-8')[:4000])
PY
```

冻结条件：

1. 恰好选择 50 个完整 episode；记录它们的原始 `episode_index`，而不是假设它们连续。
2. 每组必须具有一份 parquet 轨迹和两路 RGB 视频（chest、wrist_right）。
3. 每组均有一致、准确的任务文本；若有多种文本，必须在 manifest 中逐组保留。
4. 数据接口为 8 维：7 个右臂关节（rad）+ 1 个夹爪通道；需要确认 `observation.state`、`action` 的维度和无 NaN/Inf。
5. 保留原始 `meta/` 文件；训练的归一化统计必须只由本次 50 组计算，不能复制旧任务统计。

建议生成并永久保存：

```text
<SOURCE_LEROBOT_V3_ROOT>/MANIFEST_NEW_TASK_50_20260825.json
```

其中至少写入：源绝对路径、50 个 episode index、任务文本、开始/结束时间、每组 success/failure、摄像头名/分辨率/FPS、数据集 git/文件校验摘要。

## 2. 数据格式：直接训练 SmolVLA，不做 LingBot 转换

新 50 组是 LeRobot v3 数据集时，SmolVLA 直接消费该格式：

```text
meta/{info.json,tasks.jsonl,episodes.jsonl,episodes_stats.jsonl}
data/chunk-*/file-*.parquet
videos/chunk-*/observation.images.chest/*.mp4
videos/chunk-*/observation.images.wrist_right/*.mp4
```

含义必须保持不变：

| 字段 | 本项目语义 |
| --- | --- |
| `observation.images.chest` | 胸部 RGB 相机 |
| `observation.images.wrist_right` | 右腕 RGB 相机 |
| `observation.state` | 7 关节实际反馈（rad）+ 夹爪状态，共 8 维 |
| `action` | 7 个绝对关节目标（rad）+ 夹爪通道，共 8 维 |
| 夹爪策略语义 | `0=打开`，`1=闭合` |

不用把视频降为 15 FPS。保留原始稳定的 30 FPS MP4；部署时模型以 15 Hz 取最新时间对齐的胸部/腕部图像和关节状态。动作时间基准为 15 Hz（每 66.667 ms），不是把视频加速或重编码成 15 FPS。

### 2.1 本轮复训前必须做的数据清洗（生成派生副本，不修改原件）

这不是“删几帧视频”即可。训练 agent 必须保留原始 50 组不动，并生成一个独立、可复现的清洗数据集：

```text
<RUN_ID>/source_lerobot_v3_raw/       # 只读原始副本
<RUN_ID>/clean_lerobot_v3/            # 本轮训练唯一允许指向的 dataset root
<RUN_ID>/manifest/cleaning_report.json
<RUN_ID>/manifest/episode_decisions.csv
```

每个 episode 必须按以下流程处理；若任一步失败，标记为 `reject`，写明原因，不能悄悄补数据或把坏组混入训练。

1. 以 `action` 的时间戳建立 **15 Hz policy 时间轴**。视频维持 30 FPS，只在每个 policy 时刻选时间上最近且不晚于该时刻的 RGB 帧。
2. 检查 `observation.state` 与 `action` 均为 8 维、时间戳严格递增、没有 NaN/Inf；`observation.state` 是实际关节反馈，`action` 是示教时的绝对关节目标。两者都必须原样保留，不能用 action 覆盖 state。
3. 夹爪检查只使用示教 action 的原始数值：`<=0.15` 视为打开，`>=0.85` 视为闭合，中间区间保持上次状态；连续 5 个 15 Hz 动作确认一次状态切换。确认 `0=打开、1=闭合`，不可倒置。
4. 找到“**第一次有效闭合**”的时刻 `t_close`，再找到其后第一次有效张开的时刻 `t_open`。两者任一不存在、顺序异常、或者在抓取前发生大量翻转时，该 episode 不进入本轮训练。
5. 以 `t_cut = t_open + 7.0 s` 截断：保留从 episode 开始到 `t_cut` 的所有 parquet transition、两路视频和时间戳；若原 episode 更短则完整保留。视频要按真实 presentation timestamp 截断，不能通过改变 FPS/播放速度制造 7 秒片段。
6. 重建清洗集的 parquet、MP4、`episodes.jsonl`、`tasks.jsonl`、`info.json`、`episodes_stats.jsonl` 和 episode 索引；不得只替换视频或只裁 parquet。清洗集内索引可以重新连续编号，但原始 index 必须写回 manifest。
7. 对每个清洗 episode 输出：原/新持续时间、原/新 action 数、双路视频帧数和 FPS、`t_close`、`t_open`、`t_cut`、夹爪开闭切换次数、拒绝原因（如有）、SHA256。抽查至少 5 组：MP4 时长应与最终 parquet 时间跨度一致，且播放没有倍速。

**J5 坐标规则：** 清洗与训练必须保留采集时的原始关节坐标。不得为了安全包络把训练数据的负 J5 改成等价的正 J5；安全守卫的 2π 转换是部署期逻辑，不能污染训练样本。

清洗结束后必须重新计数。如果合格 episode 少于 50 条，不得假装仍是“50 组训练”；应先报告保留/拒绝清单，再决定补采或按实际样本数调整训练。

### 2.2 固定 45/5 的 episode 级划分

从清洗后的合格 episode 中，以固定随机种子生成 `split_manifest.json`：45 条训练、5 条验证。划分单位必须是 episode，不能从同一条轨迹抽帧同时进入训练和验证。归一化统计、训练 batch 和 augmentation 只能使用 45 条训练集；5 条验证集只用于选择 checkpoint 和离线诊断。

任务文本也属于训练输入。训练前把本轮使用的**唯一精确 prompt**写入 manifest，并验证 50 条清洗数据的 task id/task text 一致；不能沿用旧水瓶 prompt，也不能让不同任务的 prompt 混在同一 checkpoint 中。

## 3. 传输至 A800：建立新的、可追溯的训练输入副本

建议在 A800 使用一个只属于本次任务的目录：

```text
/ssd/hanbo/TNNLS_2026/data/onearm_Tele/smolvla/<RUN_ID>/source_lerobot_v3
/ssd/hanbo/TNNLS_2026/data/onearm_Tele/smolvla/<RUN_ID>/manifest
/ssd/hanbo/TNNLS_2026/outputs/smolvla/<RUN_ID>
```

其中 `<RUN_ID>` 例如 `20260825_<task_slug>_50`。不要覆盖旧 QGF、旧 SmolVLA 或 LingBot 数据目录。

优先使用可续传且会校验的 `rsync`：

```bash
rsync -aH --info=progress2 --partial --append-verify \
  <SOURCE_LEROBOT_V3_ROOT>/ \
  zwwl_user3@175.102.130.70:/ssd/hanbo/TNNLS_2026/data/onearm_Tele/smolvla/<RUN_ID>/source_lerobot_v3/
```

若 Orin 到 A800 没有可用路由，可以让 Windows 笔记本作为**纯文件中转**：先从 Orin 以 SFTP/rsync/scp 拉到本地，再用 A800 私钥上传。私钥绝不进入 git，也不复制到 Orin 数据目录。大文件优先使用 WinSCP 的 SFTP 队列与断点续传。

传输后必须在源和目标分别生成同一份文件清单，并比对：

```bash
cd <DATASET_ROOT>
find . -type f -printf '%P\t%s\n' | LC_ALL=C sort > FILE_MANIFEST.tsv
sha256sum FILE_MANIFEST.tsv
```

此外，对抽样 episode 解码双路 MP4、读 parquet，并检查时间戳单调、8 维 state/action、无 NaN/Inf。未经这些检查，不开始训练。

## 4. A800 上训练新的 SmolVLA checkpoint

### 4.1 环境与显卡选择

训练前先确认 A800 机器的空闲卡，且记录 CUDA、PyTorch、LeRobot 和 base checkpoint 的版本。只使用明确空闲的 GPU；不要为了本任务抢占正在跑实验的卡。

```bash
nvidia-smi
python - <<'PY'
import torch
print('torch=', torch.__version__)
print('cuda=', torch.version.cuda)
print('gpu_count=', torch.cuda.device_count())
PY
```

使用与已成功旧 SmolVLA 训练兼容的 LeRobot 环境/版本；确认官方 base 模型 ID、revision/commit、许可证与本地完整性后，写入 `<RUN_ID>/manifest/training_provenance.json`。本仓库当前不能据文件准确断言某个外部 base 模型 ID，因此不能凭空填写。

### 4.2 关键训练要求

1. 新建输出目录：`/ssd/hanbo/TNNLS_2026/outputs/smolvla/<RUN_ID>/`。
2. 初始化自官方 base，而非任何本项目旧任务 checkpoint。
3. 数据集根目录只指向第 2.1 节生成的 `clean_lerobot_v3`；严禁直接训练原始目录。
4. 归一化统计只能由 45 条训练 episode 产生；保存 stats、processor/preprocessor、policy config 和完整 checkpoint。不得复制旧水瓶、旧包裹、订书器或水杯任务的统计。
5. 采用一次固定随机种子并记录；可用 20,000 steps、batch size 64 作为旧项目的**起始试验配置**，但训练 agent 应根据显存、日志和过拟合迹象如实记录调整，不能把它写成已验证的最佳值。
6. 数据仅 50 组，使用第 2.2 节固定的 45/5 episode 级划分；5 条验证集用于 checkpoint 选择、loss 和时序诊断，不能取代后续真机泛化评估。
7. 保存 `last` 与按验证指标选择的 `best`，并记录二者对应 step。

先做一次 100--500 step 的 smoke train，确认数据能读、loss 有限、双路视频和 8D action/state 均被真正消费后，才启动正式训练。正式训练可从 20,000 steps、batch size 64 开始，但必须保存实际发生的 checkpoint（建议至少 5k/10k/15k/20k 或等价里程碑）；不得只凭随机抽帧准确率就宣布模型可部署。

训练完成后，目标目录至少应含：

```text
<RUN_ID>/
  checkpoint/                 # 可由 LeRobot 直接加载的 SmolVLA 权重
  config.json / policy config
  preprocessor/processor 配置与 normalization stats
  training_provenance.json
  train.log
  metrics.jsonl 或等价训练曲线
  validation_summary.json
  SHA256SUMS
```

现有工具可作为实现参考：

- [`tools/prepare_smolvla_deployment_bundle.py`](../tools/prepare_smolvla_deployment_bundle.py)
- [`tools/validate_smolvla_checkpoint.py`](../tools/validate_smolvla_checkpoint.py)
- [`tools/smoke_smolvla_policy_server.py`](../tools/smoke_smolvla_policy_server.py)
- 旧任务结果记录：[`docs/SMOLVLA_TRAINING_RESULT_20260805.md`](SMOLVLA_TRAINING_RESULT_20260805.md)

不要复用旧任务的 `episodes_stats.jsonl`、任务 completion pose、初始位姿 envelope、软目标范围或成功率结论。

### 4.3 离线验收必须包含 15 Hz 时序指标

过去的“夹爪逐帧准确率约 98%”只说明抽样动作拟合不错，不能证明模型会在抓住物体后、放到目标位置后正确释放。验证集必须在与部署相同的 **15 Hz 动作时间轴**上完整回放，而不是把 30 FPS 视频逐帧当作动作标签。

对 5 条 held-out episode，至少输出以下指标，并按 checkpoint 对比：

| 指标 | 要回答的问题 |
| --- | --- |
| 7 关节 action MAE / max error | 手臂输出是否偏离示教，尤其是首条 action。 |
| 首个完整 action chunk 的 min/max、相邻差分 | 是否有 2π 坐标跳变、异常大首帧或 chunk 内突跳。 |
| 夹爪 open/close F1 | 开和关是否都被识别，避免“长时间保持闭合”刷高普通准确率。 |
| 首次闭合时刻误差 | 是否在接近物体时闭合，而非过早/过晚。 |
| 放置后首次张开时刻误差 | 是否会在任务完成后释放。 |
| 提前张开率、漏张开率、状态翻转次数 | 是否在运输中松开、放置后不松开或反复开合。 |

评测必须离线模拟与 Orin 完全相同的夹爪状态机：`open <= 0.15`、`close >= 0.85`、连续 5 个 15 Hz 动作确认，并纳入最短驻留/接触保持时间。报告既要包含模型的裸 gripper 输出，也要包含过滤后真机会实际接收到的开/关序列。对于每个候选 checkpoint，必须用真实验证首观测推理完整首 action chunk；若发现关节坐标跨 2π、超出该任务数据范围或夹爪时序不合理，标记为离线不合格，不进入真机阶段。

## 5. 将新 checkpoint 部署回 Orin（先离线，绝不覆盖旧模型）

在 Orin 创建单独模型目录：

```text
/home/nvidia/work/telop/models/smolvla_<RUN_ID>
```

从 A800 复制完整 deployment bundle 后在两端校验 SHA256。保留旧目录：

```text
/home/nvidia/work/telop/models/smolvla_onearm_20k_20260805
```

新任务建议新增 profile，例如：

```text
/home/nvidia/work/telop/SmolVLA-with-QGF/tools/smolvla_new_task_env.sh
```

它应独立配置：

```bash
export SMOLVLA_ORIN_BUNDLE=/home/nvidia/work/telop/models/smolvla_<RUN_ID>
export SMOLVLA_SERVER_MODEL_PATH=${SMOLVLA_ORIN_BUNDLE}/checkpoint
export SMOLVLA_FPS=15
export SMOLVLA_ACTIONS_PER_CHUNK=50
```

不要修改原有 `tools/smolvla_orin_env.sh` 的旧模型路径；这样旧任务仍可复现。

运行时的已有参考代码：

- [`tools/smolvla_orin_env.sh`](../tools/smolvla_orin_env.sh)：公共 Orin 环境变量模板。
- [`tools/start_smolvla_policy_server.sh`](../tools/start_smolvla_policy_server.sh)：policy server。
- [`tools/start_smolvla_ros_client.sh`](../tools/start_smolvla_ros_client.sh)：ROS2 async policy client。
- [`tools/ubuntu_smolvla_stack.sh`](../tools/ubuntu_smolvla_stack.sh)：stack 生命周期参考。
- [`lerobot_robot_armstrong_ros2/src/lerobot_robot_armstrong_ros2/configuration_armstrong_ros2.py`](../lerobot_robot_armstrong_ros2/src/lerobot_robot_armstrong_ros2/configuration_armstrong_ros2.py)：观测、动作、夹爪策略配置。
- [`lerobot_robot_armstrong_ros2/src/lerobot_robot_armstrong_ros2/armstrongros2.py`](../lerobot_robot_armstrong_ros2/src/lerobot_robot_armstrong_ros2/armstrongros2.py)：相机/关节与动作桥接。
- [`lerobot_robot_armstrong_ros2/src/lerobot_robot_armstrong_ros2/smolvla_guard.py`](../lerobot_robot_armstrong_ros2/src/lerobot_robot_armstrong_ros2/smolvla_guard.py)：夹爪滞回与保护门。
- [`servo_controller/config/smolvla_first_rollout.yaml`](../servo_controller/config/smolvla_first_rollout.yaml)：底层安全配置参考。

旧任务代码中任何“水瓶/纸箱”专用 completion pose、初始姿态 envelope、任务关节范围都不得直接迁移。新任务必须根据新 50 组计算并单独落在新 profile 中；在未验证前，宁可不启用自动完成判定。

## 6. 固定 30 FPS 相机、15 Hz 异步推理的约束

目标频率：

| 层 | 频率/周期 | 说明 |
| --- | --- | --- |
| 胸部、右腕 RGB 采集与录制 | 30 FPS / 33.33 ms | 视频保持真实时间，不能倍速或丢失时间基准。 |
| SmolVLA 观测/动作时间基准 | 15 Hz / 66.667 ms | 每个 policy tick 使用最新且时间戳新鲜的双相机图像与真实关节状态。 |
| SmolVLA action chunk | 当前约定 50 actions | 对应约 3.33 s 的 15 Hz 动作；加载后必须核验新 checkpoint 的实际输出 shape。 |
| Orin 到控制器平滑发送 | 125 Hz / 8 ms | 仅在经过人工现场批准后，底层以插值方式发送；不是模型推理频率。 |

异步逻辑应保持如下：首块 action chunk 预加载成功后才允许显示“可以运动”；从开始执行当前块的第一步就异步请求下一块；使用新的真实观测重新推理；队列不足或观测过期时保持/停机，不发送陈旧动作。必须记录每块推理耗时、入队/出队时间、队列深度、实际执行动作与关节反馈，方便判断 Orin 是否真能支撑 15 Hz。

## 7. 夹爪：保持已验证的策略语义

当前普通 SmolVLA 运行时的策略语义为：

```text
policy gripper <= 0.15  => 打开
policy gripper >= 0.85  => 闭合
0.15 < policy gripper < 0.85 => 保持上一次状态
```

现有配置还包含连续帧确认、最短状态驻留和接触后保持；以当前代码为准，不要另写一套隐式阈值。注意 ROS 安全夹爪服务使用的是 `requested_open`，而 policy 内部用的是 `gripper_closed`，两者是相反语义，改代码时必须显式转换并做单元测试。

新 checkpoint 的夹爪输出分布可先离线绘图检查；若 0/1 方向错误，必须在离线样本中发现和修复，不能通过真机试错来判断。

## 8. 完全离线的验收清单（不接触硬件）

训练/部署 agent 可以完成以下工作：

- 解码 50 组数据、跑 parquet/视频/时间戳完整性检查；
- 从新 checkpoint 读取模型，在 A800 和/或 Orin 上对保存的离线观测做前向推理；
- 核对双路相机键名、8D state/action、输出 chunk shape、动作范围和夹爪映射；
- 使用 `validate_smolvla_checkpoint.py`、`smoke_smolvla_policy_server.py` 做本地或 loopback 健康检查；
- 验证 bundle 的 checksum、模型配置和归一化统计均来自新任务；
- 以 `dry_run=true`、全部 hardware authorization 为 false 的方式检查 ROS2 配置加载（如需要）。

以下行为一律禁止：

```text
power_on / set_powered_on(true)
set_robot_enabled(true)
set_motion_enabled(true)
servo mode enter / servo_j
夹爪移动或使能
任何会使机器人运动的 launch/脚本步骤
```

## 9. 明天到现场后的人工边界

离线验收通过只表示“模型与软件已准备好”，不表示允许自动运动。现场必须由在机器人旁、可触达实体急停的人完成：确认障碍物/工作区、检查急停、检查相机、检查新 checkpoint 路径，然后按已有受控流程启动。

首次真机应只做短时、低风险的人在场测试；训练 agent 不得自动执行 ARM、MOVE 或任何上电/使能步骤。若任务、夹爪方向、初始姿态 envelope、动作范围任一项未确认，保持 motion gate 关闭。

## 附录 A：已发生 bug 与复训/部署硬性防线

以下问题均已在本项目中真实发生。User2 在本轮复训、导出和离线部署检查中必须逐项留痕；不能用真机上电来替代这些检查。

| 已发生问题 | 已知根因/表现 | 本轮必须执行的防线 |
| --- | --- | --- |
| J5 坐标相差 2π | 部分任务的示教 J5 是原始负角；旧运行时向模型输入安全守卫使用的等价正角，造成模型归一化输入错误、首 chunk 出现异常大动作。 | 清洗和训练保持采集原始关节坐标；训练前用一条当前真实反馈与数据范围逐 joint 对照。部署时才可在安全守卫内部做 2π 等价检查/回映射，绝不能把该转换混入训练数据。当前水杯/订书器运行 profile 以 `SMOLVLA_CANONICALIZE_POLICY_OBSERVATION=false` 保留原始模型坐标；新任务必须根据自己的数据决定，不能盲抄。 |
| 首动作异常但普通验证 MAE 很小 | 原先离线指标按随机帧/动作采样，未覆盖真实首观测的完整 action chunk 和夹爪状态机。 | 强制执行第 4.3 节的 15 Hz 整段回放、首 chunk min/max/差分、首次闭合/释放时刻和状态翻转指标。 |
| 初始位姿 envelope 在 MOVE 前误拒绝 | 旧水瓶任务的关节范围被复用；即使尚未执行模型动作，也会以 J3/J5 不在旧范围而拒绝。 | 从新清洗数据独立生成首姿态范围/容差 manifest；不得复用水瓶任务 completion pose、初始范围或软目标。当前水杯/订书器现场 profile 的临时初始容差是 20°，仅是运行兼容设置，不是新训练数据的统计结论。 |
| checkpoint、prompt 或任务 profile 串台 | 曾出现界面显示订书器、实际加载红包裹 checkpoint 且模型收到红包裹 prompt；也出现过包裹任务界面正确而模型仍收到旧水瓶 prompt。 | 每次 ARM 之前打印并校验：profile 名、绝对 bundle/checkpoint 路径、精确 prompt、dataset root、checkpoint SHA256。启动器不得继承上次 shell 的 bundle/task 环境变量；服务就绪日志必须包含所选 checkpoint 身份。 |
| 不同任务 rollout 数据混入同一根目录 | 旧默认根目录为水瓶 QGF rollout，混用会污染 episode 编号、任务文本和统计。 | 每个任务都用独立 dataset root、独立 cohort 和独立编号 manifest；训练输入只能来自本轮派生的 clean 数据集。 |
| 夹爪快速开合或长期不释放 | 裸输出在阈值附近波动；普通准确率可能被大量“保持闭合”帧掩盖。 | 训练数据清洗记录开/关事件与翻转次数；离线回放使用 `open<=0.15`、`close>=0.85`、连续 5 个 15 Hz 动作确认及现有驻留/接触保持逻辑。必须同时保存裸输出与过滤后实际命令。 |
| 30 FPS 视频和 15 Hz 动作被错误按同一帧率处理 | 这会造成动作/图像错位、视频倍速或错误的逐视频帧监督。 | 保留 30 FPS 原视频和 PTS；所有训练、切段、评测以 15 Hz action timestamp 为主，图像按时间戳取最近且不晚于该 action 的帧。 |
| policy server 生命周期异常 | 曾出现 stale `127.0.0.1:8080`、新 server 退出后错误附着旧 server、ROS publisher 数量短暂不稳定、`action_enabled` 状态监控误报。 | 离线 launcher 要验证新 server PID、端口、模型路径与 readiness；首 action chunk ready 后才允许后续人工 MOVE。不要再以“恰好一个 ROS publisher”的瞬时数量作为唯一安全条件；改用受控启动/停止和身份校验。 |
| 试图以删安全守卫解决停机 | 运行中 gate stop、范围拒绝和反馈异常有不同根因；移除保护会把软件错误变成硬件风险。 | 保留保护门；先离线定位是首姿态、模型坐标、chunk、状态机还是服务生命周期问题。任何放宽阈值都必须按任务 profile 记录原因和数值，不能全局放宽。 |

本轮训练不要求、也不允许 User2 控制机器人硬件。训练完成后的 Orin bundle 只能通过纯离线/loopback 检查；现场 ARM/MOVE 由机器人旁的操作者在完成独立核对后执行。

## 10. 交付物

完成本交接后，应提交：

1. 原始/清洗派生数据的 manifest、50 个 episode 清单、逐组清洗决定、文件校验结果与 `cleaning_report.json`；
2. 固定 45/5 split manifest，以及 A800 上独立的新 SmolVLA 训练目录、训练日志、base revision、随机种子、训练/验证划分和 best checkpoint；
3. 自包含的 Orin deployment bundle（含 checkpoint、配置、preprocessor、normalization stats、SHA256）；
4. 新任务独立的 Orin env/profile，不能覆盖旧任务 profile；
5. 离线 smoke-test 与第 4.3 节时序评测报告：双路图像、8D state/action、15 Hz 计划、50-action chunk、夹爪裸输出/状态机输出、首 chunk 和时序指标；
6. 一份“尚未进行真机运动”的明确状态说明，以及现场人工验证步骤。
