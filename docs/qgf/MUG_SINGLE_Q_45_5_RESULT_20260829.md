# 水杯任务单 Q Critic:数据传输、训练、验收与 Orin 部署结果(2026-08-29)

按 `docs/qgf/RTX4090_SINGLE_Q_CRITIC_TRAINING_HANDOFF.md` 执行,任务从红色包裹换成
**水杯**(`把水杯放到紫色的箱子上`)。本文按该文档第 14 节的八个问题逐条回答。

**全程未对任何机器人发出上电、使能、伺服、夹爪或运动命令。** 只有文件操作和 GPU 计算。

## 0. 与交接文档的三处偏离(全部已留档)

| 偏离 | 原因 |
| --- | --- |
| **用物理 GPU 0,不是文档铁律要求的 GPU 1** | 用户明确指定。且 LingBot 自己的 `~/projects/lingbot/env_gpu1.sh` 里写着 `CUDA_VISIBLE_DEVICES=1` —— 文档铁律恰好会让训练和 LingBot 推理抢同一张卡。改用 GPU 0 解掉了这个冲突。 |
| **不用 conda 新建环境,改为 clone 本机已有环境** | `.110` **没有外网**(无默认路由),文档 §7 的 `conda create` / `pip install` 一条都跑不通。改为离线 clone 本机 `pointnet` 环境(py3.10.18 + torch 2.5.1+cu121 + torchvision 0.20.1+cu121,与 §7 要求一字不差)+ 从 a800 转运的离线 wheelhouse。 |
| **lerobot 用 `--no-deps` 安装** | lerobot 0.4.4 声明 `torchvision>=0.21`,与文档钦定的 `0.20.1` 直接冲突,pip 解析必然失败。查过源码:特征提取只用 `av` + `torch` + `SmolVLAPolicy`,不碰 torchvision 与 torchcodec。安装前后核对 torch/torchvision 版本未变。 |

## 1. 实际用了哪 50 条 episode

全部 50 条(`episode_000000`–`episode_000049`),无排除。

| 项 | 值 |
| --- | --- |
| 成功 / 失败 | **29 / 21**(58.0%) |
| prompt | 全部 `把水杯放到紫色的箱子上`,唯一 |
| 相机键 | 全部 chest + wrist_right,唯一 |
| 逐条体检 | 6 个必需文件齐全非空、chunk `[50,8]`、state 8 维、时间戳严格递增、稀疏奖励结构正确(success 末帧 reward=1,failure=0,每条恰好 1 个 done)、双路 MP4 ffprobe 可解码 —— **50/50 全过** |

**ep0/ep1 曾建议按 pilot 排除**(开场两条 180 s 超时失败,之后隔 35 分钟才开始 ep2),
用户决定 50 条全用,故未排除。

### 光照分组(采集时新增的记录字段)

| 光照 | n | 成功 | 成功率 | 95% CI | 超时数 |
| --- | ---: | ---: | ---: | :---: | ---: |
| normal | 30 | 21 | 70.0% | [52.1%, 83.3%] | 3 |
| medium | 20 | 8 | 40.0% | [21.9%, 61.3%] | 9 |
| 合计 | 50 | 29 | 58.0% | [44.2%, 70.6%] | 12 |

成功率差 Fisher 双尾 p = 0.045,超时率差 p = 0.007。

> **⚠️ 这不能当作光照效应。** 光照与时段完全共线:
> `medium ep0-1 (02:51-02:57Z)` → `normal ep2-31 (03:32-05:40Z)` → `medium ep32-49 (05:54-07:09Z)`。
> 任何随时间漂移的因素(杯子摆位、操作习惯、桌面位置、机械臂热状态)都与光照绑死。
> 且最后 4 条(ep46-49)清一色 `episode_timeout_180s`,而 ep43/44/45 还是连续三条成功 ——
> 这个断崖更像某个具体条件变了,不像渐变的光照影响。
> 要干净地测光照必须交错跑(N/M/N/M…),不能一段接一段。

## 2. 传输校验

| | |
| --- | --- |
| 源 | Orin `/home/nvidia/work/telop/mug_purple_box_real_rollouts/episodes` |
| 目标 | `.110` `/opt/qgf_real_robot/datasets/mug_purple_box_baseline50_20260829/raw_episodes` |
| 文件数 / 字节 | **300 / 18,812,927,981** |
| 校验 | **逐文件 SHA256 两端完全一致(VERIFY_OK)** |
| 路径 | Orin `eth0 = 192.168.2.171/32` + 到 `.110` 的 `/32` 主机路由直推,不经笔记本或 a800 |

按文档只复制 6 类训练输入,**未复制** 64.6 GiB 的 `chest.mjpeg` / `wrist_right.mjpeg` /
`*.jsonl` / `capture_summary.json` / `monitor.log` / `recorder.log`,清单记录在
`source_episode_map.json` 的 `not_copied_non_training_files`。

空间核算(§2 公式):训练输入 17.52 GiB + 2 × feature cache 2.06 GiB + 20 GiB = 门槛 41.64 GiB,
实际可用 205 GiB。

## 3. checkpoint、prompt、commit、环境、物理 GPU

| 项 | 值 |
| --- | --- |
| SmolVLA bundle | `smolvla_20260827_mug_purple_box`,`model.safetensors` SHA256 `c56bb1b9…`,与 Orin 原件一致 |
| **训练步数** | **020000**(用权重 SHA256 与 a800 训练输出逐一比对确认) |
| prompt | `把水杯放到紫色的箱子上` |
| upstream commit | `ed0108f9544c33b0166b5b17a338bdb1ea502bb7` |
| 快照 commit / 补丁 commit | `6497b6ab…` / `948cfb1e…`(diff 存于 `environment/patch_45_5.diff`) |
| 环境 | Ubuntu 20.04.6,py3.10.18,torch 2.5.1+cu121,cudnn 90100,driver 535.183.01 |
| **物理 GPU** | **0**,PID↔UUID↔index 快照见下 |

### bundle 的 SUPERSEDED 问题(必须说明)

该 bundle 带有一份 8-28 写的 `SUPERSEDED.txt`,建议改用 `smolvla_20260827_mug_50`(步数 015000)。
原因是组员 8-29 的提交 `23e49b2` 只把 red_parcel 的路径改到 clean 版,**mug 那行没动**。

**仍然使用 020000 是正确的**:文档 §6 要求提视觉 token 的 bundle 必须与产生 rollout 的完全一致。
两者的 v2 replay 差距也在噪声内:

```
015000: F1=0.9548  closeerr=0.7467  openerr=0.6667  jmae=0.0317
020000: F1=0.9528  closeerr=0.7333  openerr=0.7200  jmae=0.0317
```

### 物理 GPU 归属实证

```
训练 PID 1172032  →  GPU-b67f7d5d-45bd-3f94-8cba-0e0d90daf168
该 UUID           →  index 0,824 MiB,79% 利用率
index 1           →  0 MiB,0%   ← 全程未碰
```

进程内看到的 `cuda:0` 只是屏蔽后的逻辑编号,上面这份 PID↔UUID↔index 对照才是物理归属实证。

## 4. 45/5 划分与正奖励样本数

seed `20260814`,按 recorded outcome 做 episode 级分层划分。

| | episode 数 | 成功/失败 | 样本数 | 正奖励 chunk |
| --- | ---: | ---: | ---: | ---: |
| train | 45 | 26 / 19 | 3606 | 56 |
| validation | **5**(ep 3, 5, 10, 18, 36) | 3 / 2 | 434 | **6** |

manifest 对齐 chunk 总数 4040,跳过账目:`missing_aligned_policy_observation` 458、
`start_transition_too_far` 5。特征缓存 50 个,总样本 4040,与 manifest 完全吻合,
形状 `[N,128,960]` / `[N,50,8]` / `[N,8]`,dtype bfloat16,无 NaN/Inf,共 1.86 GiB,
逐条 SHA256 记录在 `manifest/feature_cache_report.json`。

> **⚠️ 验证集只有 6 个正奖励 chunk。** `val_q_success_mean` 是在 6 个样本上取的平均,
> 噪声极大。文档要求"验证有正奖励 chunk"已满足,但薄到这个程度,checkpoint 选择的
> 置信度有限。

## 5. 逐 epoch 指标与 best epoch

完整 80 轮历史存于 `critic_member_00.pt['history']` 与 `logs/train_single_qcritic.log`
(注意:`training_summary.json` **不含**逐 epoch 历史,这是上游脚本的写法)。

| epoch | train_q | train_v | val_td | val_q_mean | val_succ | val_fail | gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.072451 | 0.008914 | 0.075104 | -0.2868 | -0.2839 | -0.2869 | 0.0030 |
| 5 | 0.019210 | 0.001201 | 0.035029 | -0.0794 | 0.1704 | -0.0829 | 0.2534 |
| 10 | 0.008264 | 0.001408 | 0.007788 | 0.1013 | 0.6302 | 0.0939 | 0.5363 |
| 15 | 0.008196 | 0.001844 | 0.011095 | 0.2755 | 0.9486 | 0.2661 | **0.6825** |
| **22** | 0.006817 | 0.001166 | **0.002076** | 0.4196 | 0.9616 | 0.4120 | 0.5496 |
| 25 | 0.012139 | 0.003295 | 0.010184 | 0.4245 | 0.9948 | 0.4165 | 0.5783 |
| 30 | 0.008146 | 0.002656 | 0.009669 | 0.6093 | 1.0156 | 0.6036 | 0.4120 |
| 40 | 0.006527 | 0.001142 | 0.004810 | 0.6955 | 0.9896 | 0.6914 | 0.2982 |
| 50 | 0.006158 | 0.000916 | 0.004700 | 0.7459 | 1.0156 | 0.7422 | 0.2735 |
| 60 | 0.005542 | 0.000789 | 0.003947 | 0.7261 | 0.9837 | 0.7225 | 0.2612 |
| 70 | 0.004622 | 0.000687 | 0.003268 | 0.7067 | 0.9870 | 0.7027 | 0.2843 |
| 80 | 0.006060 | 0.001299 | 0.014095 | 0.6635 | 0.9824 | 0.6590 | 0.3234 |

**选中 epoch 22**,依据是验证 TD loss 最低(0.002076),不是取最后一轮
(最后一轮 0.014095,是选中轮的 7 倍)。

> **⚠️ 两个准则不指向同一个 checkpoint。** 失败样本的 Q 从 epoch 10 的 0.0939 单调涨到
> epoch 50 的 0.7422,成功样本稳在 ~1.0,两者持续收敛 —— 这是离线 RL 的典型值高估。
> 成功/失败**分离度峰值在 epoch 15(0.6825)**,而 TD loss 准则选出的 epoch 22 是 0.5496。
> 本次严格执行文档规定的 TD loss 准则,未擅自改用分离度选择,但这个取舍存在,记录在此。

> `val_q_success_mean` / `val_q_failure_mean` / gap 是 5 条留出集上的 **Q 值分离度**,
> **不是成功率**,按文档 §12 要求不得作为成功率报告。

## 6. critic checkpoint

| 项 | 值 |
| --- | --- |
| 路径 | `outputs/single_qcritic/critic_member_00.pt` |
| 大小 | 21,942,338 B(20.9 MiB) |
| SHA256 | `ee550ff5592b5e5c2cb31c057f09d047a2a22e173e807ea5ed9ac5acf2ecd455` |
| 架构 | `visual_transformer`,state 8 / action 8 / horizon 50 / visual `[128,960]` / d256-L3-H4-drop0.1 |
| 成员数 | 1(`ensemble_member_index=0`),uncertainty gate 关闭 |

验收(§12,全过):
- 45/5 记录正确、不相交,两侧正奖励样本均 > 0
- `critic_member_*.pt` 恰好一个
- `load_action_chunk_critic` 在 CPU 与 GPU 0 均可加载
- 80 轮完整历史;`selected_epoch == argmin(val_td_loss)`;不是取最后一轮
- 权重与历史无 NaN/Inf(58 + 58 个张量)
- **两次独立重载,同一批验证样本 Q 逐位相同(max diff 0.000e+00)**

## 7. Orin 部署

```text
/home/nvidia/work/telop/models/qgf/
├── real_17_116_single_qcritic/                      # 旧水瓶,聚合哈希 ea97f51a… 部署前后一致
└── mug_purple_box_single_q_45_5_20260829/           # 本次
    ├── critic_member_00.pt
    ├── training_summary.json
    ├── training_input_summary.json
    ├── episode_split_45_5.json
    ├── training_provenance.json
    ├── SHA256SUMS
    └── orin_offline_smoke.log
```

5 个文件两端逐一 SHA256 一致;bundle 自带的 `SHA256SUMS` 在 Orin 上 `sha256sum -c` 全部 OK;
**旧水瓶 critic 哈希部署前后完全未变**。

`tools/smolvla_orin_env.sh` 第 25 行的 `SMOLVLA_QGF_CRITIC_PATH` 默认值**仍指向旧水瓶 critic,
未做任何修改** —— 按文档 §13 要求,新任务必须运行时显式覆盖。

### Orin 端离线冒烟(已通过)

维度全对;50×8 强制生效(25 步 chunk 被 `ValueError` 拒绝);
全程封锁 outbound socket,无任何网络访问;**未调用任何机器人服务**。

注意:Orin 上跑需要先设库路径,否则 torch 导入即失败:

```bash
export LD_LIBRARY_PATH=/home/nvidia/work/telop/venvs/smolvla-orin/opt/libcusparse_lt-linux-sbsa-0.5.2.1-archive/lib:$LD_LIBRARY_PATH
```

### 上机运行时必须显式导出(不改公共默认值)

```bash
export SMOLVLA_QGF_CRITIC_PATH=/home/nvidia/work/telop/models/qgf/mug_purple_box_single_q_45_5_20260829/critic_member_00.pt
export QGF_RUN_MODE=qgf
export QGF_BETA=<由用户指定的正数>          # 实际 guidance 系数 = 1/beta
export SMOLVLA_ORIN_BUNDLE=/home/nvidia/work/telop/models/smolvla_20260827_mug_purple_box
# prompt 必须是:把水杯放到紫色的箱子上
```

`grad_clip_norm=1.0`,`uncertainty_scale=0.0`。

## 8. 安全声明

数据传输、环境搭建、特征提取、训练、部署准备的**全过程中,没有对任何机器人发出
上电、使能、进入伺服、夹爪或机械臂运动命令**。全部工作只涉及文件操作与 GPU 计算。

---

## 附:输入敏感度诊断(文档未要求,主动做的)

因为 LIBERO Q 线的 E8 曾查出 critic 视觉全盲(换整个场景 Q 只动 0.4~0.8%),
上机前值得先确认这个 critic 到底在用哪些输入。以 ep3 内部真实样本的 Q 跨度
(0.00998)为标尺:

| 扰动 | mean \|ΔQ\| | 占自然跨度 |
| --- | ---: | ---: |
| **action → 随机 N(0,1)** | 0.02068 | **207%** |
| visual → 高斯噪声 | 0.00556 | 56% |
| action → 全零 | 0.00435 | 44% |
| visual → 全零 | 0.00347 | 35% |
| 换整条 episode 的 visual | 0.00136 | 14% |
| 换整条 episode 的 action | 0.00142 | 14% |
| **state → 随机 N(0,10)** | **0.0000287** | **0.29%** |

**action > visual ≫ state**。

1. **与 E8 的"视觉全盲"不是同一现象**:这里视觉是有反应的(毁掉视觉能推动 56% 的自然
   跨度),动作通道更是明显活着(207%),而 QGF 求导正是对动作求。

2. **死掉的是 state 通道**:8 维本体感受即使给荒谬值也只推动 0.29%。结构上讲得通 ——
   state 在序列里只占 1 个 token(cls 1 + state 1 + visual 128 + action 50 = 180),
   且腕部相机本就看得见手臂位姿,信息冗余。

3. **常数向量给出逐位相同的 Q 是 LayerNorm 的必然结果**,不是 bug:
   `state_proj = Sequential(LayerNorm(8), Linear(8,256))`,LayerNorm 对任何常数向量
   输出都相同。这一点已验证,排除了探针错误的可能。

4. **最该警惕的现象** —— 失败 episode 上 Q 近乎常数:

   ```
   ep3  failure  Q 均值 0.2062  std 0.0015
   ep5  failure  Q 均值 0.2074  std 0.0014
   ep10 success  Q 均值 0.6571  std 0.3376   末 chunk 0.9589
   ep18 success  Q 均值 0.6272  std 0.3531   末 chunk 0.9557
   ep36 success  Q 均值 0.7294  std 0.3270   末 chunk 0.9674
   ```

   **恰恰在最需要 QGF 介入的时候(任务正在走偏),Q 是平的**,梯度存在但工作在极平坦
   的区域。这像是离线 RL 常见的"值函数学会认 episode 身份而非评估动作"——45 条数据
   加只有终止奖励,很容易走到这一步。

   这不构成阻止上机的理由,但:上机时 **β 的选择会很敏感**;若观察到 QGF 与 baseline
   几乎无差别,以上就是现成的解释,不必再花时间猜。

## 产物位置

| 产物 | 位置 |
| --- | --- |
| 50 条训练输入 + mapping + SHA256 | `.110` `/opt/qgf_real_robot/datasets/mug_purple_box_baseline50_20260829/` |
| SmolVLA bundle(产 rollout 的那份) | `.110` `/opt/qgf_real_robot/policy_bundles/mug_purple_box/` |
| aligned manifest + `episode_split_45_5.json` + 特征报告 | `runs/…/manifest/` |
| 全部 visual feature cache(50 个,1.86 GiB) | `runs/…/features/` |
| 训练输出 / best critic / 两个 summary | `runs/…/outputs/single_qcritic/` |
| 全部日志(建 manifest / 提特征 / 训练 / 三份验收 / 两份诊断) | `runs/…/logs/` |
| 环境 provenance / 补丁 diff / nvidia-smi 四份快照 | `runs/…/environment/` |
| 实际执行的命令行 | `runs/…/commands/` |
| 部署包 | `runs/…/deployment_bundle/` |
| 离线 wheelhouse(可复现安装) | `.110` `/opt/qgf_real_robot/wheelhouse/` |
| Orin 部署 | `/home/nvidia/work/telop/models/qgf/mug_purple_box_single_q_45_5_20260829/` |
