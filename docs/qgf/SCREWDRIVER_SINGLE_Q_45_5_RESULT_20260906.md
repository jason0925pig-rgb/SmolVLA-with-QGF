# 螺丝刀入盒 single-Q critic:训练、验收与部署

日期:2026-09-06
运行 ID:`screwdriver_into_box_single_q_45_5_20260906`
任务串:`把杯子里的螺丝刀放进纸盒里`
机器 / 卡:walle(192.168.2.110),**物理 GPU 1**(uuid `GPU-12ffc92d-b926-e552-c165-1dc890071859`)

---

## 1. 数据来源,以及产生这批 rollout 的 bundle 是怎么定下来的

50 组 baseline rollout 在 Orin `screwdriver_real_rollouts/episodes/episode_000000..049`,
今天 12:38 开采、17:01 收工。50 条全部 `policy_mode=baseline`、`qgf_beta=0`、
`comparison_cohort=screwdriver_normal`,任务串逐条一致,六件套(metadata / 三个 parquet /
两路 mp4)无缺失,合计 19.50 GiB。**26 成功 / 24 失败**。

### bundle 身份靠访问时间定,不是靠 launcher profile

提视觉特征**必须**用产生这批数据的那个策略。但 episode metadata 里不记 bundle,
而 `collect_smolvla_task_rollouts.ps1` 的 screwdriver profile(分支 `new-task-pipeline`)
里写的是 `models/smolvla_20260904_screwdriver` —— **Orin 上根本没有这个目录**。

改用访问时间判定。`/home/nvidia/work` 挂载带 `relatime`:

| bundle | `checkpoint/model.safetensors` atime |
| --- | --- |
| **smolvla_20260904_screwdriver_into_box** | **2026-09-06 12:38:59** |
| smolvla_20260827_mug_purple_box | 2026-09-05 14:29 |
| smolvla_20260828_red_parcel_clean | 2026-09-04 18:53 |
| smolvla_20260903_green_parcel_clean | 2026-09-04 11:41 |
| 其余 6 个 | 08-27 ~ 08-29 |

12:38:59 正是本次采集会话启动的时刻,且该 atime(09-06)晚于其 mtime(09-05 00:48),
在 `relatime` 语义下这是一次真实读取,不是陈旧时间戳。

**独立佐证**:把该 bundle 拷到 4090 后算出的
`model.safetensors` = `a07247422dc277abcb0a27b3bccb84e3ca498a22fcd7f48feb6a9612a69238cb`,
与 09-04 螺丝刀 ckpt 交付文档 §6 里记录的哈希**逐字符相同**。两条互不依赖的证据指向同一个 bundle。

## 2. 冻结与传输

冻结判定 PASS:50 目录 / 50 合格 / 0 问题,单一任务串,单一 camera keyset,单一 notes 模板。

传输 Orin → 4090 用 `stage_task_to_4090.sh`,别名 `walle4090` 先校验解析到
`192.168.2.110` 且 `BindAddress=192.168.2.171`(强制走有线,防串到同 IP 的另一台真机)。

```
数据    300 文件 / 20,937,863,005 B   SHA256 双端逐文件全同
bundle   34 文件 /  3,062,416,908 B   自带 SHA256SUMS 在 4090 上 -c 全过
耗时    35 分钟(限速 12000 KB/s)
```

组 bundle 时又做了**第二次独立校验**:在 4090 上把 300 个文件重新哈希,逐条对
Orin 侧写下的 `source_SHA256SUMS`,0 处不符。

### 路径本地化

`checkpoint/config.json` 的 `vlm_model_name` 与 `checkpoint/policy_preprocessor.json` 的
`tokenizer_name` 原本指向 Orin 绝对路径,改写为 4090 上的
`policy_bundles/screwdriver_into_box/vlm`。改写前后 `model.safetensors` 的 SHA256 完全一致。
`train_config.json` 里还有一处 HuggingFace 仓库名,它是训练记录、不参与 `from_pretrained`,
与红/绿/螺丝刀 ckpt 一致地未改动。

## 3. manifest 与 45/5 划分

| 项 | 值 |
| --- | --- |
| 对齐 chunk | **4831** |
| action chunk 形状 | [50, 8] |
| state 维 | 8 |
| policy_hz / 最大间隙 | 15.0 / 0.1 s |
| 因缺对齐 policy observation 跳过 | 446 |
| 划分策略 / 种子 | 按 outcome 分层 / 20260814 |
| 训练 45 条 | 23 成功 / 22 失败 |
| 留出 5 条 | `[14, 15, 16, 19, 37]` —— 3 成功 / 2 失败 |

留出集里成功和失败都有:只有一种 outcome 的验证集选不出 Q checkpoint。
训练样本 4384(正奖励 62),验证样本 447(正奖励 10)。

## 4. 视觉特征

用 §1 那个 bundle 的 `checkpoint/` 提取,50/50 缓存全部生成并通过检查:
`[N, 128, 960]` bfloat16,合计 4831 样本 / 2,382,946,204 B(2.22 GiB),
逐 episode SHA256 已记入 `checksums/feature_cache_SHA256SUMS`。

## 5. 训练与选点

配方与水杯/订书机/红包裹逐字相同(交接文档第 11 节固定,非本任务参数):
ensemble 1 · 80 epoch · batch 16 · lr 3e-4 · wd 1e-4 · gamma 0.99 · expectile 0.7 ·
polyak 0.005 · d_model 256 · 3 层 4 头 · dropout 0.1 · seed 20260814。
耗时约 7 分钟(09:43:35 → 09:50:27 UTC)。

**选中 epoch 24**,`val_td_loss = 0.004558`,是 80 轮的 argmin;末轮 epoch 80 是
`0.013041`,高 2.9 倍 —— 所以这个选择确实起了作用,不是取最后一轮。

### 两个准则不一致(与水杯同型,程度更轻)

| | epoch | val_td | 分离 gap | 成功 Q | 失败 Q |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD argmin(**已选**) | 24 | **0.004558** | 0.524 | 0.950 | 0.426 |
| 分离度峰值 | 11 | 0.009310 | **0.773** | 0.985 | 0.213 |

失败样本的 Q 从 epoch 1 的 −0.245 单调漂到 epoch 24 的 0.426、epoch 80 的 0.852
—— 离线 RL 的价值高估,分离度因此随训练塌缩(epoch 80 只剩 0.172)。
**本次按文档钦定的 TD 准则执行,未擅自改选**;但如果后续要按分离度选点,
epoch 11 是更好的候选,这个分歧必须留档。

> 分离 gap 是 5 条留出 episode 上的 Q 值分离度,**不是成功率**,不能当成功率报。

## 6. 第 12 节验收:7 组全部通过

覆盖:运行目录与 GPU 固定(`CUDA_DEVICE_ORDER=PCI_BUS_ID` / `CUDA_VISIBLE_DEVICES=1`,
torch 看到的 uuid 与 `nvidia-smi` 的物理 1 号卡逐字符相同)· 45/5 不相交且并集恰为 0..49 ·
训练/验证正奖励均大于 0 · 恰好一个 critic 成员 · CPU 与 GPU 双路加载 · 架构与全部 shape ·
选中 epoch == argmin(val_td) · history 与两套权重(各 58 个张量)无 NaN/Inf ·
**两次独立重载在同一固定 batch 上给出逐位相同的 Q(最大差 0.000e+00)**。

## 7. beta 探针:这个 critic 有可行 beta,但不是 0.5

在 5 条留出 episode 上做离线探针(50 个样本,`grad_clip_norm=1.0`,噪声种子 20260903,
`full` 模式 —— 相对扰动、ΔQ、分布内、裁剪率四项都是 EXACT,不是替身量)。

`ratio = ‖grad/beta‖ / ‖velocity_t‖`(中位),state 约定 `raw`:

| beta | ratio 中位 | ΔQ 均值 | ΔQ>0 占比 | 五门 |
| ---: | ---: | ---: | ---: | :--- |
| 0.005 | 0.0819 | 0.1073 | 1.00 | B 超上限 |
| **0.01** | **0.0504** | **0.0999** | **1.00** | **全过 ← 建议** |
| 0.02 | 0.0282 | 0.0712 | 1.00 | 全过 |
| 0.05 | 0.0113 | 0.0186 | 1.00 | A 低于下限 |
| 0.1 | 0.0057 | 0.0124 | 1.00 | A |
| **0.5** | **0.0011** | **0.0017** | 0.98 | **A** |
| 5.0 | 0.0001 | 0.0002 | 0.68 | A、C |

`deployed` 约定给出同样的建议值 0.01(ratio 0.0439)。裁剪率全程不超过 0.006,
所以 1/beta 的线性外推成立。

### 与另外三个 critic 并排

| 任务 | beta=0.5 时 ratio | 可行 beta |
| --- | ---: | ---: |
| 水杯 | 0.00005 | **无** |
| 订书机 | 0.00119 | 0.005 |
| 红包裹 | 0.00237 | 0.01 |
| **螺丝刀** | **0.00113** | **0.01** |

结论和前三次一致:**beta=0.5 下引导幅度是速度场的千分之一,真机跑出来的 QGF 与 baseline
在机制上无从区分**。螺丝刀这个 critic 本身是可用的(0.01 处五门全过、ΔQ>0 占比 1.00),
但要用就得把 beta 降到 0.01 量级。

> 探针的每一个分数都是 critic 对自己引导的评价。过 C 门只说明引导在按 critic 的要求走,
> 不说明 critic 要的是对的。ΔQ 是 5 条 episode 上 50 个样本的配对统计,episode 内强相关,
> 有效样本量接近 5 而不是 50 —— 看符号和量级,别看 p 值。

## 8. 部署到 Orin

`/home/nvidia/work/telop/models/qgf/screwdriver_into_box_single_q_45_5_20260906/`(21 MB,6 文件)

```
a5a93a2499b33d3a3f9a118c9c100afdc723b8971686e1d17d5d6a1faf347a65  critic_member_00.pt
ee8574f15539fb18eeda29042bcb5e85914f43822eefa40a46a9274c76ef0c9a  episode_split_45_5.json
42e070de17a70af802fa321d3a4708b7b3504dfb24b8bd0927b0d4cb0b2e8157  training_input_summary.json
2bec6aa5c78195890d32c2b9620f3184deb2682b349a57310821c3ac62371f00  training_provenance.json
514c0e04ad686b0a88a23e744a27c6615d64ea44787845a5af56662f13859717  training_summary.json
```

**既有 critic 全部逐字节未变**(部署前后各算一次目录树哈希):
订书机 `375e2b0e…`、水杯 `7ad60e25…`、红包裹 `c7ea8cea…`、水瓶 `ea97f51a…`。
新增且仅新增一个目录。

### 离线冒烟:44 通过 / 1 失败

通过项包括:bundle 自校验、checkpoint 身份、离线前向、50×8 形状强制(25 步 chunk 被
`ValueError` 拒绝)、**critic 对 action chunk 确实有梯度(不是 no-op)**、裁剪到范数 1、
施加量精确等于 `clipped_grad / beta`、不确定性门 == 1、ensemble == 1、
回环监听只绑 127.0.0.1、全程零对外连接与零域名解析、没有导入任何 robot/ROS 模块。

**唯一那条失败是已查明的无害漂移**:

```
FAIL  the deployed policy server still uses the gates and the beta log format
      this test reproduces (1 fragment(s) drifted)
      drifted: f"grad_clip_norm={grad_clip_norm:.8g}; uncertainty_gate=disabled"
```

冒烟脚本逐字匹配 Orin 上 `policy_server_qgf.py` 的日志格式字符串。09-05 部署
`guidance_mode` 开关时,该行在 `uncertainty_gate=disabled` 之后追加了
`; guidance_mode={...}; random_seed={...}`,于是逐字匹配落空。
备份 `policy_server_qgf.py.bak.20260905_225231` 在原文件旁边。
功能性检查(server gate 1 = `critic_arch` 为 `visual_transformer`、
server gate 2 = 八个动作通道)都通过 —— **漂移的是日志文本,不是门控逻辑**。

> 途中还有两条失败是我自己造成、已修:① `QGF_BETA`(0.5)与 `SMOLVLA_QGF_BETA`(0.01)
> 不一致 —— 冒烟脚本抓对了,现已统一为 0.01;② 我在 Windows 上用 Python 重写环境文件
> 写成了 CRLF,路径尾部带上 `\r`。两条都不是 critic 或部署的问题,但记在这里,
> 因为"用 Python 改一个要在 Linux 上 source 的文件"这个坑会再犯。

### 运行时导出

```bash
export SMOLVLA_QGF_CRITIC_PATH=/home/nvidia/work/telop/models/qgf/screwdriver_into_box_single_q_45_5_20260906/critic_member_00.pt
export QGF_RUN_MODE=qgf
export SMOLVLA_QGF_BETA=0.01          # 探针建议值;两个 BETA 名字必须一致
export QGF_BETA=0.01
export SMOLVLA_QGF_GRAD_CLIP_NORM=1.0
export SMOLVLA_ORIN_BUNDLE=/home/nvidia/work/telop/models/smolvla_20260904_screwdriver_into_box
export LD_LIBRARY_PATH=/home/nvidia/work/telop/venvs/smolvla-orin/opt/libcusparse_lt-linux-sbsa-0.5.2.1-archive/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

`collect_smolvla_task_rollouts.ps1` 的 screwdriver profile 里 `QCriticPath` 现在还是空串
(当时留空是为了让"无匹配 critic"的守卫生效)。要跑 QGF 臂,得把它填成上面那个路径,
并把 `Bundle` 从不存在的 `smolvla_20260904_screwdriver` 改成
`smolvla_20260904_screwdriver_into_box`。

## 9. 关于机器人控制的声明

**全程没有上电、没有使能、没有进入伺服模式、没有控制夹爪、没有发送任何机械臂动作。**
冻结、传输、训练、部署、冒烟都只做文件、张量和回环操作。
采集会话在本次流水线启动前已由操作员自行结束(17:01 最后一条 episode 落盘)。
真机测试须由机器人旁的操作员按受控 ARM/MOVE 流程另行进行。

## 10. 代码与环境

| 项 | 值 |
| --- | --- |
| 4090 仓库快照 | `948cfb1e3d962ab89414d01571b1dbc5956d7678` |
| 上游 commit | `ed0108f9544c33b0166b5b17a338bdb1ea502bb7` |
| python / torch | 3.10.18 / 2.5.1+cu121(cuda 12.1,cudnn 90100) |
| 驱动 | 535.183.01 |
| 环境 | `/opt/qgf_real_robot/envs/visual_iql_py310` |

与红包裹那次是同一个快照 commit,结果可直接对比。

## 附:路径清单

- 4090 运行目录 `/opt/qgf_real_robot/runs/screwdriver_into_box_single_q_45_5_20260906/`
- 4090 数据集 `/opt/qgf_real_robot/datasets/screwdriver_into_box_baseline50_20260906/`
- 4090 策略 bundle `/opt/qgf_real_robot/policy_bundles/screwdriver_into_box/`
- Orin critic `/home/nvidia/work/telop/models/qgf/screwdriver_into_box_single_q_45_5_20260906/`
- 本次运行的环境契约 `tools/qgf_4090_staging/env_screwdriver_20260906.sh`
