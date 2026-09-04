# 红色包裹 单 Q Critic(45/5)训练与部署结果

日期:2026-09-02
运行 ID:`red_parcel_single_q_45_5_20260902`
对应交接文档:`docs/qgf/RTX4090_SINGLE_Q_CRITIC_TRAINING_HANDOFF.md`

本文按该文档第 14 节的八个问题逐条回答。所有数字均来自实际执行输出,不是复述计划。

---

## 1. 用了哪 50 条,成功/失败各多少,排除了什么

`/home/nvidia/work/telop/red_parcel_real_rollouts/episodes` 下 `episode_000000` ~ `episode_000049`,**全部 50 条,一条未排除**。目录里也没有 pilot 或调试 episode。

| 项 | 值 |
| --- | --- |
| outcome | success 33 / failure 17 |
| task_prompt | `把箱子里的红色包裹拿出来放到桌子上。`,50 条**逐字一致** |
| 相机键集 | 1 种(chest + wrist_right,各含 exact_jpeg_archive) |
| 训练输入 | 300 文件 / 19,709,912,735 字节(18.36 GiB) |
| 采集时段 | 2026-09-02 04:09–10:18 UTC |
| policy_mode | 全部 `baseline`,`qgf_beta=0` —— 没有 QGF episode 混入 |

冻结时每条都逐项验过并通过:六类必需文件存在且非空、outcome 属于 {success, failure}、`action_chunk_normalized` 首行为 `[50,8]`、state 8 维、`transitions.parquet` 非空且时间戳严格递增、两路 MP4 经 ffprobe 探流成功。目录名索引与 metadata 的 `episode_index` 一致。

**必须说明的一点:这 50 条是混光照队列** —— notes 显示 normal 30 条、medium_light 10 条、dark/light 10 条。水杯那批(30 normal + 20 medium)也是混的,此处沿用先例,但 Q critic 确实是在混光照数据上训练的,报告结果时不应描述为「固定光照下训练」。

## 2. 源/目标路径、文件数、总字节、SHA256 是否一致

| | 路径 | 文件 | 字节 |
| --- | --- | ---: | ---: |
| 源(Orin) | `/home/nvidia/work/telop/red_parcel_real_rollouts/episodes` | 300 | 19,709,912,735 |
| 目标(.110) | `/opt/qgf_real_robot/datasets/red_parcel_baseline50_20260902/raw_episodes` | 300 | 19,709,912,735 |

- `SHA256 MATCH: 300 files identical on both sides`
- `VERIFY_OK files=300 bytes=19709912735`
- bundle:`BUNDLE_VERIFY_OK files=73 bytes=3994760100 name=red_parcel_clean`

传输走 Orin 的 ssh 别名 `walle4090`,脚本在传输前验证该别名解析为 `192.168.2.110` 且 `BindAddress=192.168.2.171`(第 3 节要求的 /32 口)。限速 20000 KB/s,配合 `nice -n 10` 和 `ionice -c3`,以便与 Orin 上的其它作业共存。

留档:`source_SHA256SUMS`、`source_episode_map.json`、`frozen_episode_list.txt`、`transfer.log`。

### 首次校验失败是脚本 bug,不是数据问题

第一次校验报 `VERIFY_FAIL ... dst=19709910`(约 19.7 MB,差 1000 倍)。根因:目标端字节数用 `awk` 求和后再过 `tr -dc 0-9`。Orin 是 gawk,完整打印 `19709912735`;而 .110 是 **mawk 1.3.4**,其默认 `OFMT=%.6g` 把同一个和打印成 `1.97099e+10`,`tr` 剥掉 `.`、`e`、`+` 后得到 `19709910`。

这个 bug 不只是噪音:脚本在 SHA256 比对**之前**就 FATAL 了,所以真正的逐文件校验根本没有执行。修复是给三处求和都加 `-v OFMT=%.0f -v CONVFMT=%.0f`(commit `8a4d316`)。**不能**改用 printf 格式串 —— 那会把双引号塞进外层 `ssh "..."` 命令串里破坏引号配对,而 `bash -n` 检测不出来。修复后在 .110 上实测:旧写法 `19709910`,新写法 `19709912735`,与源端一致。

## 3. checkpoint、prompt、git commit、环境、物理 GPU

| 项 | 值 |
| --- | --- |
| SmolVLA bundle | `/home/nvidia/work/telop/models/smolvla_20260828_red_parcel_clean` |
| bundle 权重 SHA256 | `24e6505a4c56edeefcfb1de282e269a6b912f25bacaf94d679248162ae9976dd` |
| task prompt | `把箱子里的红色包裹拿出来放到桌子上。` |
| 上游 commit | `ed0108f9544c33b0166b5b17a338bdb1ea502bb7` |
| .110 代码快照 commit | `948cfb1e3d962ab89414d01571b1dbc5956d7678` |
| 主机 | `walle`,Ubuntu 20.04.6,x86_64 |
| Python / Torch | 3.10.18 / 2.5.1+cu121(cuDNN 90100) |
| 驱动 | 535.183.01 |
| **物理 GPU** | **1**(`CUDA_VISIBLE_DEVICES=1`,uuid `GPU-12ffc92d-b926-e552-c165-1dc890071859`) |

### bundle 是怎么确定的

**episode metadata 没有记录 checkpoint 路径**,`recorder.log` 和 `monitor.log` 里也没有。Orin 上有三个红包裹 bundle。确定方法是:`tools/collect_smolvla_task_rollouts.ps1` 的 `red_parcel` profile 里 `Task` 串与 50 条 episode 的 `task_prompt` **逐字一致**,且该 profile 的 `DatasetRoot` 正是这批 rollout 的保存根目录。结论可靠,但这是反推,不是直接留档 —— 建议后续在 episode metadata 里直接记录 checkpoint 路径。

### 本地化

复制到 .110 的 bundle,其 `config.json` 的 `vlm_model_name` 和 `policy_preprocessor.json` 的 `tokenizer_name` 仍指向 Orin 绝对路径,直接用会报 `HFValidationError`。已改写为 .110 本地路径。`checkpoint/` 和 `checkpoint_last/` **两个目录都改**(只改前者会让 `grep -rl /home/nvidia checkpoint*/` 校验失败,也给后来者留坑)。改写前后 `model.safetensors` 的 SHA256 完全一致 —— 只动了两个 JSON 路径字段。

### GPU 选择

本次按用户要求和文档第 1 条铁律使用**物理 GPU 1**。前两次(水杯、订书机)用的是 **GPU 0**,原因记录在水杯的 provenance 里:LingBot 自己的 `env_gpu1.sh` 固定占用 GPU 1,当时为避开 LingBot 推理而改用 GPU 0。本次开跑时两张卡均为 0 MiB 空闲,无竞争。

证据:训练进程 PID 27202 出现在 PCI bus `00000000:05:00.0`,即 `nvidia-smi` 的 index 1;GPU 0 全程 0 MiB。验收脚本另外用 uuid 交叉核对了 torch 可见设备与物理 GPU 1 一致。

## 4. 45/5 的精确清单与正奖励样本数

episode 级分层划分,按 recorded outcome 分层,`--split-seed 20260814`,输出文件名 `episode_split_45_5.json`(45/5 补丁生效;原代码会写死 `episode_split_90_10.json`)。

| | 数量 | outcome |
| --- | ---: | --- |
| train | 45 | success 30 / failure 15 |
| val | 5 | success 3 / failure 2 |

**validation episode:`[7, 13, 25, 32, 46]`**

| 项 | 值 |
| --- | ---: |
| train 样本 | 4,059 |
| val 样本 | 660 |
| train 正奖励样本 | 89 |
| val 正奖励样本 | 9 |

train ∩ val = ∅,并集恰好 50 条不重不漏,两侧都同时含成功与失败。验证集含正奖励 chunk,满足第 8 节「单一 outcome 的 validation 不得用于选 checkpoint」的门槛。

视觉特征:50/50 个 `.pt` cache,形状 `[N,128,960]`,合计 2.17 GiB / 4,719 样本,无 NaN/Inf,逐 episode SHA256 已记录。manifest 对齐出 5,176 个 chunk,实际产出 4,719 个样本,差值记录在 `manifest/feature_cache_report.json`。

## 5. 每 epoch 的损失、best epoch 与选择依据

80 个 epoch 全部完成,完整历史保存在 checkpoint、`training_summary.json` 与日志中。

| 项 | 值 |
| --- | ---: |
| **selected_epoch** | **71** |
| selected_val_td_loss | 0.008534602463658388 |
| 是否 = argmin(val_td_loss) | 是,验收脚本独立复算确认 |
| 最后一个 epoch(80)的 val_td_loss | 0.010575 —— 劣于 71,证明不是简单取末轮 |

选中 epoch 处:`val_q_success_mean = 1.0148`,`val_q_failure_mean = 0.8236`,`gap = 0.1912`。

> **这个 gap 是 5 条留出 episode 上的 Q 值分离度,不是成功率,不得当作成功率报告。**

训练前中后各存了 `nvidia-smi` 快照;`GPU binding OK: PID 27202 -> uuid -> physical GPU index 1`。

## 6. critic 路径、大小、SHA256 与可加载性

| 项 | 值 |
| --- | --- |
| 路径(.110) | `runs/red_parcel_single_q_45_5_20260902/outputs/single_qcritic/critic_member_00.pt` |
| 大小 | 21,942,274 字节(20.9 MiB) |
| SHA256 | `1b51820fee0137a429d566a831ded32541ebc6e5f5255aea2f8309a0f38ffb36` |

第 12 节七组验收 **全部通过**:

- 恰好一个 `critic_member_00.pt`
- `load_action_chunk_critic` 在 **CPU 和物理 GPU 1** 上都能加载
- `critic_arch = visual_transformer`;state 8 / action 8 / horizon 50 / visual `[128,960]`;d_model 256 / layers 3 / heads 4 / dropout 0.1 / ensemble_member_index 0
- history、`model_state_dict`(58 张量)、`value_model_state_dict`(58 张量)全部 finite,无 NaN/Inf
- 固定 validation batch 上两次独立重载输出 **bit-identical(max diff 0.000e+00)**

## 7. Orin 部署路径,旧 critic 是否保持不变

部署到:`/home/nvidia/work/telop/models/qgf/red_parcel_single_q_45_5_20260902/`

包含第 13 节要求的六个文件,`SHA256SUMS` 在 Orin 端复验全部 OK:

```
1b51820fee0137a429d566a831ded32541ebc6e5f5255aea2f8309a0f38ffb36  critic_member_00.pt
beabf7da722360cd971ab975203588ab6158afc628bb309e8634b83c809362f1  episode_split_45_5.json
a9900a2d6617cd3c5abcc7fbd777edae0c642a75a56ce4b6263950c2d58fa157  training_input_summary.json
a506b52d62371dafceeb1e4425f3352a09e06b931672207ce9abbdb4db8d3443  training_provenance.json
05237712c93d3491e0ecf0ec9f2967d0080e71dd46f9158c852a2421c05bf80d  training_summary.json
```

**三个既有 critic 部署前后逐一哈希比对,byte-identical:**

| 目录 | 哈希 |
| --- | --- |
| `stapler_into_box_single_q_45_5_20260830` | `375e2b0e09a8da9c...` |
| `mug_purple_box_single_q_45_5_20260829` | `7ad60e258f4cb6ae...` |
| `real_17_116_single_qcritic`(水瓶,固定保留) | `ea97f51a67de5d75...` |

恰好新建一个目录,没有覆盖任何东西。

### Orin 离线 smoke test:通过

在 Orin 的实际推理环境(`venvs/smolvla-orin`,torch 2.4.0a0+nv24.07,CUDA 可用)中执行:

- guided velocity 形状 `(4, 50, 8)`,保持 `[B, 50, 8]` chunk 结构且 finite
- `raw|grad| = 0.0916293`,`clipped|grad| = 0.0916293`,`|guidance delta| = 0.183259`
- critic 对 action chunk **确有梯度(不是 no-op)**;梯度按 `grad_clip_norm=1` 裁剪;施加的引导恰好等于 `clipped_grad / beta`;不确定性门关闭(gate=1);ensemble=1
- 监听只绑 `127.0.0.1`,从未绑 `0.0.0.0`;beta 日志行完成回环往返(214 字节)
- **0 次外部连接或域名解析**;`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`

### 运行时必须显式导出

```bash
export SMOLVLA_QGF_CRITIC_PATH=/home/nvidia/work/telop/models/qgf/red_parcel_single_q_45_5_20260902/critic_member_00.pt
export SMOLVLA_QGF_BETA=0.5
export QGF_BETA=0.5
export SMOLVLA_QGF_GRAD_CLIP_NORM=1.0
export QGF_RUN_MODE=qgf
export SMOLVLA_ORIN_BUNDLE=/home/nvidia/work/telop/models/smolvla_20260828_red_parcel_clean
```

`guidance_coefficient = 1 / beta`,`grad_clip_norm = 1.0`,`uncertainty_scale = 0.0`。

> **交接文档第 13 节只写了 `QGF_BETA`。smoke test 显示策略服务器实际读取的是 `SMOLVLA_QGF_BETA`,且缺少 `SMOLVLA_QGF_GRAD_CLIP_NORM` 时会报错。** 两个名字都设上最稳妥。真机测试前请与现场人员确认这一点。

β 取值本身**未在本次确定**。项目内唯一有记录的 β 是 0.5(水杯 60 条 QGF 中有 33 条记录了 `qgf_beta=0.5`,其余 27 条未记录);订书机 9 月 1 日那 40 条 QGF 未记录任何 β。真机对比前应先锁定一个全局 β。

## 8. 关于机器人控制的声明

**训练与部署准备全程,没有上电、没有使能、没有进入伺服模式、没有控制夹爪、没有发送任何机械臂动作。**

- 4090 只做数据处理、特征提取与训练,从未启动任何机器人控制节点。
- Orin 侧只做文件读写、哈希与 rsync;部署脚本显式声明并实测未调用任何机器人服务。
- 离线 smoke test 装有 import 钩子,拒绝导入任何机器人/ROS 模块(实测 0 次导入),并对自身可执行体做静态自审(扫描 22,306 字符,未发现任何机器人动作调用)。

真机测试必须由机器人旁的操作员按受控 ARM/MOVE 流程另行进行。

---

## 附:与前两次任务的差异

| 项 | 水杯 | 订书机 | 红包裹(本次) |
| --- | --- | --- | --- |
| 物理 GPU | 0 | 0 | **1** |
| 训练输入 | 18 GB | 17 GB | 18.36 GB |
| selected_epoch | — | — | 71 / 80 |
| 光照 | 混(30 normal + 20 medium) | — | 混(30 normal + 10 medium + 10 dark) |

GPU 不同不影响结果可比性(同模型、同超参、同随机种子 20260814),但记录在此以免日后混淆。

---

## 附录:2026-09-04 验收复核与 acceptance.log 的更正

复核本报告时发现 `runs/red_parcel_single_q_45_5_20260902/logs/acceptance.log`
的结尾是:

```
FAILED 2 check(s):
  - CUDA_VISIBLE_DEVICES == 1 so that 'cuda' is physical GPU 1 (got '0')
  - the visible CUDA device IS physical GPU 1
    (torch GPU-b67f7d5d-... vs nvidia-smi GPU-12ffc92d-...)
```

这与正文「第 12 节七组验收全部通过」表面矛盾,必须说清楚,否则日后审计会误判本次训练用错了卡。

**这两项 FAIL 反映的是验收脚本自身进程的环境,不是训练时的环境。** 该检查读的是
"当前 python 进程可见的 CUDA 设备",而那次验收启动时没有带 `CUDA_VISIBLE_DEVICES=1`,
于是它看到 GPU 0 并如实报告与 `nvidia-smi` 的物理 GPU 1 不符。

训练本身用的是物理 GPU 1,证据独立于任何环境变量 ——
`environment/nvidia_smi_train_during.txt` 直接记录了训练期间的进程归属:

```
|  1  N/A  N/A  27202  C  .../envs/visual_iql_py310/bin/python  812MiB |
1, GPU-12ffc92d-b926-e552-c165-1dc890071859, 818 MiB, 62 %
0, GPU-b67f7d5d-45bd-3f94-8cba-0e0d90daf168,   0 MiB,  0 %
```

训练进程 PID 27202 在物理 GPU 1 上占 812 MiB、利用率 62%,GPU 0 全程 0 MiB / 0%。

2026-09-04 以正确环境重跑验收:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  python tools/qgf_4090_staging/accept_single_q_task.py
```

结果:

```
ALL HARD CHECKS PASSED  (task=red_parcel run=red_parcel_single_q_45_5_20260902)
```

新日志保存在 `logs/acceptance_gpu1_pinned.log`,与原 `acceptance.log` 并存,
两份都不删除。中间还有一次只补了 `CUDA_VISIBLE_DEVICES=1` 的运行
(`logs/acceptance_gpu1.log`),它把 FAIL 从 2 项降到 1 项,剩下的是
`CUDA_DEVICE_ORDER == PCI_BUS_ID (got None)` —— 说明脚本要求两个变量同时钉住
才认为物理卡序号被锁定。

**教训**:验收脚本必须和训练用同一套 GPU 环境变量启动,否则它检查的是自己而不是被检查对象。
后续任务运行 `accept_single_q_task.py` 时请一并导出 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。
