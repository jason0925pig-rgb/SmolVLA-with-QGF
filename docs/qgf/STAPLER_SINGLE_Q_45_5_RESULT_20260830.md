# 订书机任务单 Q Critic:数据传输、训练、验收与 Orin 部署结果(2026-08-30)

对应交接文档:`docs/qgf/RTX4090_SINGLE_Q_CRITIC_TRAINING_HANDOFF.md`
姊妹文档:`docs/qgf/MUG_SINGLE_Q_45_5_RESULT_20260829.md`(水杯任务,同一套代码与超参)

run id:`stapler_into_box_single_q_45_5_20260830`
任务 prompt:`把订书机放进快递纸盒`

---

## 0. 与交接文档的偏离(全部已留档)

| # | 偏离 | 原因 | 留档位置 |
|---|---|---|---|
| 1 | 用物理 GPU 0,交接文档要求 GPU 1 | 用户明确指定;且 LingBot 自己的 `env_gpu1.sh` 把 `CUDA_VISIBLE_DEVICES` 钉在 1,用 GPU 0 反而避开推理争用 | `training_provenance.json → machine.gpu_policy_deviation` |
| 2 | 90/10 划分改成 45/5 | 只有 50 条 episode,90/10 会让验证集只剩 5 条且正奖励样本可能为 0 | patch `948cfb1e`,默认值仍是 90/10 |

GPU 归属不是靠环境变量声明,是靠 `nvidia_smi_train_during.txt` 里 PID → GPU UUID → index 的实证。

---

## 1. 实际用了哪 50 条 episode

来源:`/home/nvidia/work/telop/stapler_real_rollouts/episodes`(Orin)
落地:`/opt/qgf_real_robot/datasets/stapler_into_box_baseline50_20260830`(4090)

| 项 | 值 |
|---|---|
| episode 数 | 50(index 0–49,连续无缺口) |
| 成功 / 失败 | 28 / 22 → **baseline 成功率 56%** |
| 光照分组 | `normal` 50 条(**没有 medium 组**,与水杯不同) |
| 结束方式 | `operator_END` 38、`episode_timeout_180s` 10、`policy_gate_closed_without_completion` 2 |

对齐后可用样本 5133 条 chunk,形状 `[50, 8]`。跳过 489 条:

- `missing_aligned_policy_observation` 488
- `start_transition_too_far` 1

**注意**:订书机这 50 条全部是 `lighting=normal`,所以它**没有可复用的 medium 亮度 baseline**。水杯的扰动对比可以靠 single-arm 模式复用旧 baseline,订书机不行,必须跑 paired 20+20。

---

## 2. 传输校验

未传输(训练用不到,省空间):`chest.mjpeg`、`wrist_right.mjpeg`、`samples.jsonl`、`normalized_policy_chunks.jsonl`、`policy_observations.jsonl`、`capture_summary.json`、`monitor.log`、`recorder.log`。

已传输的每个文件都做了 **逐文件 SHA256 比对,Orin 与 4090 完全一致**。

---

## 3. checkpoint、prompt、commit、环境、物理 GPU

| 项 | 值 |
|---|---|
| 策略 bundle(Orin) | `/home/nvidia/work/telop/models/smolvla_20260827_stapler_into_box` |
| 策略 bundle(4090) | `/opt/qgf_real_robot/policy_bundles/stapler_into_box` |
| `model.safetensors` SHA256 | `7665055af789d1a7d8c0f556143f0bdc2bdc03dd5a7874118d2c40e4b6f41eb0` |
| upstream commit | `ed0108f9544c33b0166b5b17a338bdb1ea502bb7` |
| snapshot commit(含 45/5 patch) | `948cfb1e3d962ab89414d01571b1dbc5956d7678` |
| 主机 / 系统 | `walle` / Ubuntu 20.04.6 LTS |
| 驱动 / Python | 535.183.01 / Python 3.10.18 |
| GPU | `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`(物理 GPU 0) |

bundle 里两个 checkpoint 目录的 `vlm_model_name` / `tokenizer_name` 从 Orin 绝对路径改写成了 4090 路径。**权重没动** —— 改写前后 `model.safetensors` 的哈希验证过是同一个值。

---

## 4. 45/5 划分与正奖励样本数

划分策略:按记录的 outcome 做 episode 级分层,seed `20260814`。

| | episode 数 | 成功 / 失败 | 样本数 | 正奖励样本 |
|---|---|---|---|---|
| train | 45 | 25 / 20 | 4459 | 73 |
| val | 5 | 3 / 2 | 674 | 9 |

验证集 episode:`[14, 15, 17, 28, 33]`。训练/验证 episode 无交集(已断言)。

正奖励占比 73/4459 = **1.64%**,也就是 **98.4% 的训练目标是纯自举**。这是稀疏终止奖励的结构性后果,不是这次训练的问题,但它直接决定了后面第 9 节说的那件事。

---

## 5. 逐 epoch 指标与 best epoch

跑满 80 epoch,按交接文档规定的准则选点:**验证 TD loss 最小的那一轮,不是最后一轮**。

```
 epoch     val_td    q_succ    q_fail      gap
     1   0.022926   -0.2212   -0.2742   0.0529
     5   0.011279    0.1815   -0.0252   0.2068
    10   0.006296    0.6168    0.2021   0.4147
    15   0.006551    0.8225    0.3781   0.4444
    16   0.005569    0.8655    0.4104   0.4551
    17   0.004415    0.8620    0.4010   0.4609   <-- 选中(argmin val_td)
    18   0.006844    0.9145    0.4197   0.4948
    20   0.008490    0.8776    0.3595   0.5181
    22   0.007434    0.9674    0.3868   0.5807   <-- 分离度最大
    25   0.007335    0.9965    0.4676   0.5289
    30   0.005926    0.9735    0.4752   0.4983
    40   0.013470    0.9457    0.4884   0.4573
    60   0.013735    0.9861    0.6308   0.3554
    80   0.015779    1.0469    0.7043   0.3426
```

选中 **epoch 17**,`val_td_loss = 0.004415`。

**必须说明的一点**:分离度最大的是 epoch 22(gap 0.5807),不是选中的 17(gap 0.4609)。TD loss 准则和分离度准则不指向同一轮。交接文档规定用 TD loss,这里就按 TD loss 选,但把这个分歧记下来。

最后一轮(epoch 80)`val_td_loss = 0.015779`,是选中轮的 3.6 倍,`q_fail` 从 0.40 漂到 0.70 —— 后期是明显的过拟合,选中轮不是终点这件事很重要。

> `val_q_success_mean` / `val_q_failure_mean` / `gap` 是 **5 条留出 episode 上的 Q 值分离度,不是成功率**,任何地方都不能当成功率报。

---

## 6. critic checkpoint

| 项 | 值 |
|---|---|
| 架构 | `visual_transformer` |
| 配置 | `state_dim=8, action_dim=8, action_horizon=50, visual_token_dim=960, visual_tokens=128, d_model=256, num_layers=3, num_heads=4, dropout=0.1, ff_multiplier=4` |
| ensemble | 1(单 critic,`uncertainty_scale=0.0`,不确定性门控关闭) |
| 大小 | 20.9 MiB |
| SHA256 | `1773a3a361755fab648af54fbb204c9336eb7e188b61d34458f91e6fcecb425b` |

验收(`logs/acceptance.log`)七项硬检查 **全部 PASS**:

1. 45/5 记录正确、训练验证不相交
2. 训练和验证集都含正奖励样本(73 / 9)
3. 恰好一个 critic member
4. CPU 和 GPU 0 都能加载,架构与形状全对
5. 选中轮 == argmin(验证 TD loss)
6. history 与两套 state_dict 共 116 个张量全部有限,无 NaN / Inf
7. 两次独立重载在同一批数据上给出 **逐位相同** 的 Q(最大差 0.000e+00)

---

## 7. Orin 部署

目标:`/home/nvidia/work/telop/models/qgf/stapler_into_box_single_q_45_5_20260830/`

随包附带 `SHA256SUMS`,五个文件全部校验通过。**Orin 上的 `critic_member_00.pt` 与 4090 上的 SHA256 一致**(`1773a3a3…`),传输无损。

以下目录**原样保留未动**:

- `/home/nvidia/work/telop/models/qgf/real_17_116_single_qcritic/`
- `/home/nvidia/work/telop/models/qgf/mug_purple_box_single_q_45_5_20260829/`

### Orin 端离线冒烟(已通过)

`orin_offline_smoke.log` → `ORIN OFFLINE SMOKE OK`:

- checkpoint 身份与六项配置断言全 PASS
- 合成张量前向一次,输出 Q 有限
- 25 步 chunk 被正确拒绝(强制 50×8)
- 全程出站 socket 被屏蔽,没有任何网络访问
- 没有调用任何机器人服务

### 上机运行时必须显式导出(不改公共默认值)

```
SMOLVLA_QGF_CRITIC_PATH=/home/nvidia/work/telop/models/qgf/stapler_into_box_single_q_45_5_20260830/critic_member_00.pt
SMOLVLA_ORIN_BUNDLE=/home/nvidia/work/telop/models/smolvla_20260827_stapler_into_box
QGF_RUN_MODE=qgf
QGF_BETA=<见第 8 节>
SMOLVLA_QGF_GRAD_CLIP_NORM=1.0
# prompt 必须是:把订书机放进快递纸盒
```

在 Orin 上跑任何 import torch 的东西,都要先:

```
LD_LIBRARY_PATH=/home/nvidia/work/telop/venvs/smolvla-orin/opt/libcusparse_lt-linux-sbsa-0.5.2.1-archive/lib
```

---

## 8. β 的确定:0.35

引导规则是 `guided_velocity = velocity - grad / beta`,`grad` 在整个 `[50, 8]` chunk 上裁到范数 1.0。**β 是梯度上的增益,不是混合权重**,引导系数 = 1/β,**β 越小引导越强**。因此 β 只有相对于 critic 自己的梯度尺度才有意义,不能跨 critic 直接搬。

### 不能用水瓶那个数换算

水瓶 critic 的梯度 0.397 是在**水杯的特征上**测的,属于分布外。订书机 critic 自己就有 13 倍的分布外虚高:

```
订书机 critic  自己数据上 0.026056   水杯数据上 0.337063   (12.9x)
```

所以 0.397 是上界,由它推出的 β 大约会激进一个数量级。

### 用水杯的真机行为做锚点

水杯是唯一在真机上筛过 β 的配置:

| β | 每元素扰动占动作 std | 真机观察 |
|---|---|---|
| 2 | 0.02% | 完全没反应 |
| 0.05 | 0.61% | 引导生效,夹爪闭合,出现过完整成功 |
| 0.03 | 1.02% | 能用但在边缘,多条没闭合 |
| 0.001 | 30.74% | 几秒内发散(夹爪指令 11.6,J1 目标偏 2.7 rad) |

换算按**夹爪通道**做,因为夹爪是最先坏的那一路。两个 critic 在自己的验证特征上,通道 7 的每元素梯度:

```
水杯    0.000540        订书机  0.004029        (7.46x)
夹爪动作 std 两边几乎一样(0.9675 vs 0.9823)
```

所以"按绝对扰动"和"按相对扰动"两种换算只差 1.5%,结果是稳的:

| 锚点 | 换算到订书机 |
|---|---|
| 水杯 0.05(跑出过完整成功) | **0.37** |
| 水杯 0.03(边缘可用) | 0.22 |

**取 β = 0.35**,即水杯那个"确实成功过"的设置的等效点。风险是对称的:太弱则 QGF 臂与 baseline 无异、20 组白跑;太强则夹爪抓不住、看起来像"Q 有害"、同样白跑。

**这个 0.35 没在真机上验证过,是从水杯外推的。** 前 2–3 条要盯着夹爪;若明显与 baseline 无差别,降到 0.22。

---

## 9. 与水杯 Q 的对比,以及一个共有的结构性问题

按流程实际记录的指标看,订书机 critic 与水杯 critic 质量相当:

| | 选中轮 | val_td_loss | 成功/失败分离度 | 训练样本 | 正奖励 |
|---|---|---|---|---|---|
| 水瓶 critic(旧) | 14 | 0.008719 | 0.4914 | 7970 | 94 |
| 水杯 critic | 22 | 0.002076 | 0.5496 | 3606 | 56 |
| **订书机 critic** | **17** | **0.004415** | **0.4609** | **4459** | **73** |

但两个任务共有一个**结构性问题,不是这次训练造成的**:

```
订书机  28 条成功 episode:终止奖励落在最后一次松爪之后 60.2 秒(中位 56.1s,范围 16.5–131.9s)
        平均 episode 长 109.3 秒 → 55% 的时长发生在任务物理上已经完成之后
水杯    29 条成功 episode:61.1 秒后(中位 62.7s)→ 61%
```

也就是说 reward=1 标注的那一刻,机械臂早就松开物体、回到初始位姿了。critic 学到的"高 Q"状态是**回到 home 之后闲置**,而那是策略无论做什么都会到达的状态。这解释了为什么 `∂Q/∂a` 会塌下来:Q 变成了目标接近度探测器,而不是动作质量评价器。

**订书机和水杯的数值几乎一模一样(60.2s vs 61.1s),说明这是这套采集流程的系统性特征,不是某个任务的意外。**

已知的修法是把终止奖励移到真正的松爪时刻重训。**当前不采用** —— 用户明确要求"这个奖励机制所有任务都不能变",跨任务一致性优先于单个 critic 的质量。此处仅留档。

---

## 10. 安全声明

数据传输、环境搭建、特征提取、训练、部署准备全过程,**没有向任何机器人下发过上电、使能、伺服、夹爪或关节运动指令**。全部是文件操作与 GPU 计算。

---

## 产物位置

4090,`/opt/qgf_real_robot/runs/stapler_into_box_single_q_45_5_20260830/`:

```
manifest/     episode_split_45_5.json, manifest_summary.json, feature_cache_report.json
features/     visual_feature_summary.json + 逐 episode 特征
outputs/single_qcritic/   critic_member_00.pt, training_summary.json, training_input_summary.json
deployment_bundle/        上述四个 json + training_provenance.json
environment/  environment.txt, pip_freeze.txt, nvidia_smi_*.txt, bundle_model_safetensors_sha256.txt
logs/         build_manifest.log, verify_split.log, verify_features.log,
              extract_visual_features.log, train_single_qcritic.log, acceptance.log
```

Orin,`/home/nvidia/work/telop/models/qgf/stapler_into_box_single_q_45_5_20260830/`:

```
critic_member_00.pt, SHA256SUMS, orin_offline_smoke.log,
episode_split_45_5.json, training_input_summary.json,
training_summary.json, training_provenance.json
```

checkpoint 本身按 `.gitignore`(`*.pt`、`runs/`、`outputs/`)**不进 git**,只以 SHA256 在本文档和 `SHA256SUMS` 中留证。
