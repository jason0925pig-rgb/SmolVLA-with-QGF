# RTX 4090 单 Q Critic 训练、留档与 Orin 部署交接

更新日期：2026-08-29

这份文档交给执行训练的 agent。目标是用新任务的 **50 条真实 baseline rollout**，在 `192.168.2.110` 的第二张 RTX 4090 上复刻上一次水瓶任务的视觉 IQL 单 critic 流程。除 episode 划分从 `90/10` 缩为 `45/5` 外，模型输入、奖励、网络、超参数、checkpoint 选择和部署接口都必须与水瓶 Q 保持一致。

训练 agent 只处理数据、训练、离线验证与文件传输。禁止上电、使能、进入伺服、控制夹爪或发送机械臂动作。

## 1. 三条铁律

1. `192.168.2.110` 有两张 RTX 4090，**只能使用物理 GPU 1**。所有 GPU 命令都必须显式设置 `CUDA_VISIBLE_DEVICES=1`；进程内出现的 `cuda:0` 只是被屏蔽后的逻辑编号，必须另外保存 `nvidia-smi` PID 快照证明它对应物理 GPU 1。
2. `~/projects/lingbot` 只属于 LingBot 推理。Q 数据、环境、代码、特征、训练产物和日志全部放在独立 SSD 根目录 `/opt/qgf_real_robot`，不得写入或修改 LingBot 目录。
3. 4090 只做训练/推理，绝不控制机器人。机械臂 SDK、ROS2 控制、上电、使能、伺服、安全门和实体急停处理始终只在 Orin；训练 agent 不得从 `.110` 启动任何机器人控制节点。

## 2. 已核验的机器与网络事实

| 项目 | 已核验值（2026-08-29） |
| --- | --- |
| 4090 主机 | `walle@192.168.2.110`，hostname `walle` |
| GPU | 2 x RTX 4090 24 GB；当前查询时两卡空闲，但训练前必须重新检查 |
| 允许使用 | 仅物理 GPU 1，即 `CUDA_VISIBLE_DEVICES=1` |
| 驱动 | `535.183.01`，两卡一致 |
| 系统 Python | 3.8.10；不可直接用于本项目，需独立 Python 3.10 环境 |
| SSD | `KINGSTON SKC3000D4096G`，3.7 TB NVMe |
| 根分区 `/` | ext4，约 2.7 TB，当前剩余约 231.5 GB |
| `/home` | ext4，当前仅剩约 27.5 GB，不足以保存 Q 数据 |
| Orin 到 `.110` | Orin `eth0` 1 Gbps 全双工，实测约 23.4 MiB/s，ping 约 0.3--0.7 ms |
| 笔记本到 A800 | 当前 Wi-Fi 路径小样本实测约 1.7 MiB/s，仅作当天网络对比 |

当前磁盘余量并不宽裕。复制前必须计算训练实际需要的六类文件总大小，并确保：

```text
目标空闲空间 >= 训练输入大小 + 预计 visual feature cache 大小 x 2 + 20 GiB
```

不满足就停止并报告，不能删 Orin 原始数据、不能悄悄漏传 episode，也不能把数据塞进 `/home`。上一次视觉训练实际不读取 `*.mjpeg`、ROS bag 或其它诊断文件；为节省 SSD，只复制第 5 节列出的**完整训练输入**，并在 manifest 中明确记录未复制的非训练文件。

## 3. 为什么使用 `/32` 精确路由，以及 Orin 为什么是跳板

现场存在两套机器人，其 Orin/机械臂控制器使用了相同的 `192.168.2.x` 地址。ROS Domain ID 只能隔离 ROS2 discovery，不能阻止基于固定控制器 IP 的 SDK 连错机器人，因此必须在 IP 路由和物理网口层隔离。

当前新 Orin 的关键配置为：

```text
eth4 = 192.168.2.170/24    # 本机机器人控制侧
eth0 = 192.168.2.171/32    # 只连接 4090 侧
192.168.2.110 dev eth0     # 仅给 .110 的精确 host route
```

`/32` 不声明整个 `192.168.2.0/24` 都在 `eth0`，只配合显式的 `.110` host route 使用。这样发往机械臂控制器的流量继续走本机控制口 `eth4`，发往 4090 的流量固定走 `eth0`，不会因为两台机器人地址相同而从错误网口 ARP/控制另一台机器人。其目的不是加速，而是降低“同 IP 两套机器人连在一起时控错设备”的风险。

当前验证结果是：笔记本 `.131` 能连接 Orin，但不能直接连接 `.110`；Orin `.171` 能稳定 ping/SSH `.110`。因此以后将 Orin 作为管理跳板：

```powershell
ssh armstrong-orin
```

进入 Orin 后：

```bash
ssh walle@192.168.2.110
```

也可从笔记本使用：

```powershell
ssh -J armstrong-orin walle@192.168.2.110
```

私钥、密码不得写入 Git。建议为 Orin -> `.110` 创建一把专用 Ed25519 密钥并仅安装公钥，以便 `rsync` 续传；不要复制其它服务器私钥。

## 4. 固定 SSD 目录与一次性准备

固定根目录：

```text
/opt/qgf_real_robot
```

`walle` 属于 sudo 组。第一次由现场授权人员执行：

```bash
sudo mkdir -p /opt/qgf_real_robot
sudo chown -R walle:walle /opt/qgf_real_robot
```

本次建议冻结以下 ID；若实际日期/任务不同，agent 可以改变叶子目录名，但根目录不能改变：

```bash
export QGF_SSD_ROOT=/opt/qgf_real_robot
export QGF_DATASET_ID=red_parcel_baseline50_20260829
export QGF_RUN_ID=red_parcel_single_q_45_5_20260829
```

最终目录必须是：

```text
/opt/qgf_real_robot/
├── repos/SmolVLA-with-QGF/             # 固定 commit 的代码
├── envs/visual_iql_py310/              # 独立环境，不污染 LingBot
├── policy_bundles/red_parcel_clean/    # 产生这批 rollout 的同一 SmolVLA bundle
├── datasets/red_parcel_baseline50_20260829/
│   ├── raw_episodes/                    # 训练真正读取的 50 条数据
│   ├── source_episode_map.json
│   ├── source_SHA256SUMS
│   └── transfer.log
└── runs/red_parcel_single_q_45_5_20260829/
    ├── manifest/
    ├── features/
    ├── outputs/
    ├── logs/
    ├── environment/
    ├── commands/
    ├── checksums/
    └── deployment_bundle/
```

## 5. Orin 上冻结 50 条官方 baseline rollout

红色包裹启动器当前保存根目录为：

```text
/home/nvidia/work/telop/red_parcel_real_rollouts/episodes
```

不要在“看见 50 个目录”时立刻传输。先根据本次 official notes/cohort 冻结准确的 50 条，排除 pilot、调阈值、异常停止和手工测试 episode。每条训练 episode 必须有：

```text
episode_metadata.json
transitions.parquet
normalized_policy_chunks.parquet
policy_observations.parquet
chest.mp4
wrist_right.mp4
```

必须验证：

- 恰好 50 条，任务 prompt、baseline checkpoint 和相机键一致；
- outcome 只允许 `success` 或 `failure`，两类都存在；
- 两路 MP4 可解码，parquet 非空，时间戳单调；
- normalized action chunk 为 `[50, 8]`，state 为 8 维；
- 保存每条源 episode 编号、outcome、文件大小和 SHA256；
- 失败/急停数据只有在操作员明确保存为 failure 且文件完整时才能进入训练；被删除的 episode 不得复原混入。

如果 50 条不是连续编号，不能用软链接后保留旧 metadata 冒充连续编号。应在复制出的训练数据中做可追溯重编号，同时更新 metadata，并保存 `source_episode_map.json`；绝不能修改 Orin 原件。

## 6. 从 Orin 直接复制到 `.110` SSD

先在 `.110` 创建目标目录：

```bash
mkdir -p /opt/qgf_real_robot/datasets/red_parcel_baseline50_20260829/raw_episodes
```

由 Orin 直接推送到 `.110`，不要经过笔记本或 A800。下面只复制训练会读取的完整输入；实际 episode 范围必须替换为冻结 manifest 中的 50 条：

```bash
rsync -a --partial --append-verify --info=progress2 --prune-empty-dirs \
  --include='episode_*/' \
  --include='episode_*/episode_metadata.json' \
  --include='episode_*/transitions.parquet' \
  --include='episode_*/normalized_policy_chunks.parquet' \
  --include='episode_*/policy_observations.parquet' \
  --include='episode_*/chest.mp4' \
  --include='episode_*/wrist_right.mp4' \
  --exclude='*' \
  /home/nvidia/work/telop/red_parcel_real_rollouts/episodes/ \
  walle@192.168.2.110:/opt/qgf_real_robot/datasets/red_parcel_baseline50_20260829/raw_episodes/
```

如果源目录还含 pilot，上述命令范围过宽；必须改成从冻结 episode 清单逐条 rsync。传输完成后在 Orin 和 `.110` 对所选文件分别计算 SHA256，并要求文件数、总字节和逐文件摘要全部一致。23.4 MiB/s 仅是当天无落盘链路实测：100 GiB 约 75 分钟，200 GiB 约 2.5 小时；实际以 `rsync` 日志为准。

同时复制**产生这 50 条 rollout 的完全相同 SmolVLA bundle**到：

```text
/opt/qgf_real_robot/policy_bundles/red_parcel_clean
```

视觉 token 和 normalized action space 都与该 bundle 绑定；不能用旧水瓶、15k/20k 包裹旧 checkpoint 或别的任务 checkpoint 提取特征。

## 7. 固定代码、环境与 provenance

代码仓库：

```text
https://github.com/jason0925pig-rgb/SmolVLA-with-QGF
```

训练事实依据与实现：

- `docs/qgf/REAL_ROBOT_Q_CRITIC_TRAINING_FACTUAL_REPORT.md`
- `tools/qgf_episode_recorder.py`
- `tools/finalize_qgf_episode.py`
- `qgf/scripts/build_real_robot_visual_iql_manifest.py`
- `qgf/scripts/extract_smolvla_visual_features.py`
- `qgf/scripts/train_real_robot_visual_iql.py`
- `qgf/src/guided_action_flow/critics/visual_transformer_critic.py`
- `qgf/src/guided_action_flow/critics/checkpoint.py`
- `lerobot_robot_armstrong_ros2/src/lerobot_robot_armstrong_ros2/policy_server_qgf.py`

把本次使用的 Git commit 保存到 `environment/git_commit.txt`。工作区必须干净；如为 45/5 修改脚本，保存补丁、commit 和 `git diff`，不能只把临时代码留在服务器。

`.110` 驱动 535 不适合仓库 A800 setup 脚本默认安装的 CUDA 12.8 wheel。不要直接运行 `setup_a800_visual_iql_env.sh`。使用独立 Python 3.10 环境和与驱动兼容的 CUDA 12.1 PyTorch，例如：

```bash
export QGF_SSD_ROOT=/opt/qgf_real_robot
conda create -p ${QGF_SSD_ROOT}/envs/visual_iql_py310 python=3.10 -y
conda activate ${QGF_SSD_ROOT}/envs/visual_iql_py310
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ${QGF_SSD_ROOT}/repos/SmolVLA-with-QGF/qgf[torch] \
  'lerobot[smolvla]==0.4.4' av pyarrow
```

安装后保存：`pip freeze`、`conda env export`、Python/Torch/CUDA/driver、GPU 名称、Git commit、完整命令和时间。先验证：

```bash
CUDA_VISIBLE_DEVICES=1 python - <<'PY'
import torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
```

## 8. 45/5 的唯一代码差异

上一次水瓶训练使用 100 条 episode、90/10 episode 级分层划分。当前 50 条必须使用相同随机种子 `20260814` 做 45/5 episode 级、按 recorded outcome 分层划分；同一 episode 的帧不能跨 train/validation。

现有代码有两个硬编码陷阱：

1. `build_real_robot_visual_iql_manifest.py` 即使 `--val-count 5`，仍把文件命名为 `episode_split_90_10.json`，报错文字也写死 90/10；
2. `train_real_robot_visual_iql.py` 强制要求恰好 90 train + 10 validation，会拒绝 45/5。

训练 agent 必须做最小、可测试、向后兼容的修改：给训练脚本增加显式 expected train/val episode 参数，默认仍为 90/10；本次传 `45/5`。manifest 文件名改为真实的 `episode_split_45_5.json`，并测试 train/val 不相交、合计 50、两边 outcome 统计合理。除这一处外，不改变算法、网络或超参数。

如果 5 条验证集中没有 success 或没有 failure，停止训练并重新做分层划分；不能用一个只有单一 outcome 的 validation 选择 Q checkpoint。

## 9. 与水瓶任务相同的数据构造

构建 manifest：

```bash
export CUDA_VISIBLE_DEVICES=1
export QGF_REPO=/opt/qgf_real_robot/repos/SmolVLA-with-QGF
export QGF_DATA=/opt/qgf_real_robot/datasets/red_parcel_baseline50_20260829/raw_episodes
export QGF_RUN=/opt/qgf_real_robot/runs/red_parcel_single_q_45_5_20260829
export QGF_POLICY=/opt/qgf_real_robot/policy_bundles/red_parcel_clean/checkpoint

python ${QGF_REPO}/qgf/scripts/build_real_robot_visual_iql_manifest.py \
  --raw-episodes-root ${QGF_DATA} \
  --output-dir ${QGF_RUN}/manifest \
  --episode-first <FIRST_OFFICIAL_EPISODE> \
  --episode-last <LAST_OFFICIAL_EPISODE> \
  --action-horizon 50 \
  --policy-hz 15 \
  --max-transition-gap-seconds 0.1 \
  --val-count 5 \
  --split-seed 20260814 \
  2>&1 | tee ${QGF_RUN}/logs/build_manifest.log
```

输入与水瓶 Q 完全相同：

```text
s       = 当前实际 8 维状态（7 关节 rad + gripper_closed）
z       = chest + wrist_right 经冻结 SmolVLA 编码器得到的 [128,960] BF16 token
a       = SmolVLA 原始 normalized 50 x 8 action chunk
r       = 稀疏终止奖励
s', z'  = 约 50/15 = 3.333 s 后的真实 next state / next visual token
d       = done / terminated / truncated / success 的合并终止标记
```

Q 的动作输入仍是 `normalized_policy_chunks.action_chunk_normalized`，不是 `action_policy`、`action_guarded` 或 `action_executed`。后三者和真实反馈用于诊断与构造环境转移，但 QGF 在线求导发生在 normalized flow action 空间，不能改变这一点。

reward 与水瓶任务一致：成功 episode 最后 transition 为 `reward=1, success=true, terminated=true, done=true`；失败末端为 `reward=0, truncated=true, done=true`；中间 reward 为 0。一个 50 步 chunk 若跨度内触达成功终点，可成为正奖励样本。

## 10. 双路视觉特征提取

```bash
CUDA_VISIBLE_DEVICES=1 python ${QGF_REPO}/qgf/scripts/extract_smolvla_visual_features.py \
  --raw-episodes-root ${QGF_DATA} \
  --aligned-manifest ${QGF_RUN}/manifest/aligned_normalized_chunks.parquet \
  --output-dir ${QGF_RUN}/features \
  --checkpoint ${QGF_POLICY} \
  --device cuda \
  --batch-size 32 \
  2>&1 | tee ${QGF_RUN}/logs/extract_visual_features.log
```

输出必须是每个 episode 一个 `.pt` cache，视觉形状 `[N,128,960]`、动作 `[N,50,8]`、状态 `[N,8]`，没有 NaN/Inf。记录总样本数、跳过 chunk 的原因、每个 episode 的输出大小和 SHA256。不要边训练边跨网解码 Orin MP4；特征提取和训练均读取 `.110` 本地 NVMe。

## 11. 单 critic IQL 训练：除 45/5 外复刻水瓶配置

固定配置：

| 参数 | 值 |
| --- | ---: |
| ensemble size | 1 |
| uncertainty gate | disabled，`uncertainty_scale=0.0` |
| epochs | 80 |
| batch size | 16 |
| learning rate | `3e-4` |
| weight decay | `1e-4` |
| gamma | `0.99` / 50-step chunk |
| expectile | `0.7` |
| Polyak | `0.005` |
| d_model / layers / heads / dropout | `256 / 3 / 4 / 0.1` |
| seed | `20260814` |
| CUDA precision | BF16 autocast，loss FP32 |
| checkpoint selection | validation TD loss 最低的 epoch；不是固定取 epoch 80 |

训练网络与水瓶一致：在线 `Q(s,z,a)`、Polyak target Q、`V(s,z)`。Value 使用 expectile regression；Q target 为 `r + 0.99 * (1-d) * V(s',z')`。只部署 Q，value model 仅用于离线训练。

```bash
CUDA_VISIBLE_DEVICES=1 python ${QGF_REPO}/qgf/scripts/train_real_robot_visual_iql.py \
  --data-dir ${QGF_RUN}/features \
  --split-file ${QGF_RUN}/manifest/episode_split_45_5.json \
  --output-dir ${QGF_RUN}/outputs/single_qcritic \
  --ensemble-size 1 \
  --epochs 80 \
  --batch-size 16 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --gamma 0.99 \
  --expectile 0.7 \
  --polyak 0.005 \
  --d-model 256 \
  --layers 3 \
  --heads 4 \
  --dropout 0.1 \
  --seed 20260814 \
  --device cuda \
  --expected-train-episodes 45 \
  --expected-val-episodes 5 \
  2>&1 | tee ${QGF_RUN}/logs/train_single_qcritic.log
```

上面两个 `--expected-*` 参数要求第 8 节的兼容修改已经落到 Git commit；没有实现时不得删除校验绕过去。

训练前、训练中和训练后分别保存 `nvidia-smi`；核对训练 PID 只出现在物理 GPU 1。不得为了加速临时启用 GPU 0，也不得启动第二个 critic。

## 12. 训练完成验收与完整留档

至少验收：

- `training_input_summary.json` 明确记录 45/5、train/val episode 清单和样本数；
- train/val episode 无交集，两边都有正/负 outcome，validation 有正奖励 chunk；
- `critic_member_00.pt` 恰好一个，可由 `load_action_chunk_critic` 在 CPU 和 GPU 1 加载；
- checkpoint 为 `critic_arch=visual_transformer`，state/action/horizon/visual shape 分别为 8、8、50、`[128,960]`；
- selected epoch 由最低 validation TD loss 产生；同时报告 `val_q_success_mean`、`val_q_failure_mean` 和 gap，但不得把它们称为成功率；
- 无 NaN/Inf，完整训练历史保存在 checkpoint、日志和 summary 中；
- 用固定 validation batch 重载 checkpoint 后输出一致。

必须完整保留：

```text
50 条训练输入及 source mapping / SHA256
aligned manifest 与 episode_split_45_5.json
全部 visual feature cache
训练输出、best critic、training_summary.json、training_input_summary.json
所有 stdout/stderr 日志
Git commit、git diff、代码快照或 bundle
pip freeze、conda env export、driver/CUDA/GPU 快照
实际执行命令、开始/结束时间、随机种子
最终 SHA256SUMS
```

训练结束生成只读归档索引；不要删除中间产物、最后 epoch、失败日志或特征 cache。不得覆盖前一次水瓶 Q 目录。

## 13. 将 Q critic 部署回 Orin

在 Orin 创建任务独立目录，不覆盖旧水瓶 critic：

```text
/home/nvidia/work/telop/models/qgf/red_parcel_single_q_45_5_20260829/
```

从 `.110` 复制以下 deployment bundle：

```text
critic_member_00.pt
training_summary.json
training_input_summary.json
episode_split_45_5.json
training_provenance.json
SHA256SUMS
```

推荐从 `.110` 推回 Orin `.171`，或在 Orin 上通过 `.110` 拉取；复制后两端逐文件校验 SHA256。旧水瓶路径必须保留：

```text
/home/nvidia/work/telop/models/qgf/real_17_116_single_qcritic/
```

新任务运行时必须显式设置，不修改公共默认值：

```bash
export SMOLVLA_QGF_CRITIC_PATH=/home/nvidia/work/telop/models/qgf/red_parcel_single_q_45_5_20260829/critic_member_00.pt
export QGF_RUN_MODE=qgf
export QGF_BETA=<由用户指定的正数>
```

实际 guidance 系数仍是 `1 / beta`，`grad_clip_norm=1.0`，`uncertainty_scale=0.0`。同时必须显式选择产生 rollout 的红色包裹 SmolVLA bundle和准确 prompt，不能继承水瓶、旧包裹、订书器或水杯配置。

先在 Orin 做纯离线加载和 loopback policy-server smoke test，确认 checkpoint、视觉维度、50 x 8 action shape、beta 日志和无网络依赖。该阶段禁止调用机器人服务。只有离线报告交付并由现场人员复核后，才由机器人旁的操作者按受控 ARM/MOVE 流程做真机测试。

## 14. 最终交付报告必须明确回答

1. 实际用了哪 50 条 episode，成功/失败各多少，哪些被排除以及原因；
2. Orin 与 `.110` 的源/目标路径、文件数、总字节和 SHA256 是否一致；
3. 使用的 SmolVLA checkpoint、prompt、Git commit、环境和物理 GPU；
4. 45/5 的精确 episode 清单和正奖励样本数；
5. 每个 epoch 的 Q/V/validation TD loss，best epoch 及选择依据；
6. critic checkpoint 路径、大小、SHA256 和可加载测试；
7. Orin 部署路径与旧水瓶 critic 是否保持不变；
8. 明确声明训练/部署准备期间没有上电、使能、伺服或控制机器人。
