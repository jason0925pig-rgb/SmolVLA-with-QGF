# 螺丝刀入盒 ckpt:检查、清洗、训练与部署

日期:2026-09-04 / 09-05
运行 ID:`20260904_screwdriver_clean_50`
任务串:`把杯子里的螺丝刀放进纸盒里`

---

## 1. 数据来源与 ep250 的处理

摇操记录存在共享数据集 `onearm_Tele/lerobot_dataset`(Orin)。螺丝刀这批是 **ep200–250 共 51 条**;按要求保留 200–249,删除 250。

`ep250` 本身没有质量问题(1 个夹爪周期、45.2 s、释放后尾巴 13.5 s),纯粹是为了凑齐 50 条。它恰好是数据集的**最后一条**,所以删除不需要重排任何索引:

| | 删除前 | 删除后 |
| --- | ---: | ---: |
| total_episodes | 251 | 250 |
| total_frames | 354,278 | 352,920 |
| 最大 episode 索引 | 250 | 249 |

删掉的 4 个文件(data parquet / meta parquet / 两路 mp4)与原 `info.json` **已备份**到
`onearm_Tele/_deleted_ep250_20260904/`(42 MB),确认无误后可自行清除。

## 2. 数据检查

50 条 episode 的任务串逐条一致,结构无问题:

- 时间戳全部严格递增、`t0 = 0`、无 NaN/Inf、state/action 均为 8 维
- 时长 med 42.7 s(29.6–63.5),抓取 med 16.3 s,释放 med 29.9 s
- 释放后尾巴 med 12.2 s,最长 39.7 s —— 有明显的空闲尾巴值得截

### J5 不需要清洗

`onearm_Tele` 这个数据集在 ep69 处发生过一次 J5 分支切换(此前记录在正分支、此后在负分支,相差 2π)。螺丝刀这批全部录于切换之后:

| | 值 |
| --- | ---: |
| 逐条 J5 均值范围 | [−2.334, −1.470] |
| 逐条均值极差 | **0.629 rad** |
| 最大间隙 | **0.058 rad** |
| 分支分布 | pos 0 条 / neg 51 条 |

天然单一分支,**没有 2π 问题**。清洗脚本因此把 J5 做成**门禁而非修正**:若检测到分支分裂就直接 `assert` 失败,而不是盲目施加一个此处不需要的 ±2π。这样这份脚本用到别的数据上时不会静默出错。

### 发现:三条 episode 是重抓

| episode | 周期 1 | 周期 2 | 周期 3 |
| --- | --- | --- | --- |
| ep215 | 22.9→26.3 s(握 3.4 s) | 32.1→50.4 s(握 18.3 s) | |
| ep245 | 11.5→16.2 s(4.7 s) | 24.1→27.1 s(3.1 s) | 33.9→42.7 s(8.8 s) |
| ep249 | 19.0→26.4 s(7.4 s) | 38.9→47.4 s(8.5 s) | |

握持时间的对比很清楚:前面那些短周期是**掉落**,最后一个才是真正放入纸盒。

> ⚠️ **红包裹清洗用的规则是「第一次释放 + 7 s」,直接套用在这里是错的** —— 会把掉落那一刻当作任务完成,教模型提前松手就算成功。本次改用**最后一次确认释放 + 7 s**,并在核验脚本里加了硬检查:每条保留的片段必须延伸到最后一次释放之后。

## 3. 清洗

规则:`[0, 最后一次确认释放 + 7.0 s]`(或到 episode 结尾,取较短者)。夹爪事件判据与红/绿一致:open ≤ 0.15、close ≥ 0.85、15 Hz 上连续 5 拍确认。

保留 50 / 50,拒绝 0。

```
成品          3.87 GiB / 55,890 帧
段长          min 739 / med 1123 / max 1723 帧
保留比例      86.1%(原 64,927 帧)
释放后尾巴    min 6.33 s / med 7.00 s / max 7.00 s
划分          45 训 / 5 留(seed 1000,val = [9, 24, 29, 40, 49])
```

### 独立跑了两遍并交叉验证

先在 Orin 上清洗一遍,再把**原始** episode 传到 a800new 重跑一遍。两次核验(`verify_screwdriver.py`)均 **ALL CHECKS PASSED**,各项数字完全吻合。

> 两端 parquet 的 SHA256 **不同**,查证原因是 pyarrow 版本号嵌在文件头(`parquet-cpp-arrow version 25.0.0` vs `25.0.1`)。取三条(839 / 1723 / 1389 行,9 列)**逐元素比对,内容零差异**。只看哈希会误报成数据损坏。

核验覆盖:结构完整性、时钟从 0 起算且严格递增、`frame_index` 连续、全局 `index` 跨 episode 连续、**视频帧数逐条等于 parquet 行数**、截断点在最后一次释放之后、J5 单分支、任务串三处一致、45/5 不相交、50 条 SHA256 复算。

### 为什么传原始数据而不是传成品

清洗后体积 **3.87 GiB,比原始的 1.92 GiB 更大** —— `-g 2`(每两帧一个关键帧,为训练时精确 seek)牺牲了压缩率。红/绿也是同样情况。因此传原始、在目标机清洗,传输量减半。

## 4. 训练

配方与红/绿完全一致,只改数据集、repo_id、输出目录:

| 项 | 值 |
| --- | --- |
| 机器 / GPU | a800new,**物理 GPU 3**(PID 3216302 @ bus `00000000:33:00.0`) |
| init 权重 | `~/parcel_smolvla/models/smolvla_parcel_init` |
| batch / steps | 64 / 20000 |
| save_freq / seed | 1000 / 1000 |
| 训练集 | 45 条(5 条留出,不进训练也不进归一化统计) |
| 参数量 | 可训练 99.9M / 总 450M |
| 耗时 | 4 小时 40 分,`End of training` 正常收尾 |

训练集列表从 `split_manifest.json` 读取并断言 train/val 不相交,不用写死列表。

> 启动时 GPU 3 上有另一位用户(`zwwl_user3`)的 vLLM 服务占用 61.6 GB。核实其显存为启动时一次性预分配(60 秒内四次读数纹丝不动,`61,647 MiB` = 75% 利用率默认配置)、不会增长,剩余 20.3 GB 大于本配方实测峰值 17.4 GB,遂按原 batch 64 启动并加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 抑制碎片。实际启动时该服务已退出,全程独占。**未对他人进程做任何操作。**

## 5. checkpoint 选择

`seq_eval.py` 在 5 条留出 episode 上回放评估,准则与红/绿一致(joint_mae 为主,夹爪指标为次)。

| ckpt | joint_mae | max_err | close_s | open_s | early_open | missed | F1 | flips | pass |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 002000 | 0.0400 | 0.5618 | 0.920 | 1.440 | 0.2 | 0.0 | 0.9085 | 2.8 | True |
| 005000 | 0.0352 | 0.6066 | 0.840 | 0.773 | 0.0 | 0.0 | 0.9202 | 2.4 | True |
| 010000 | 0.0352 | 0.5197 | 0.867 | 0.920 | 0.0 | 0.0 | 0.9141 | 2.4 | True |
| **015000** | **0.0337** | 0.5478 | **0.627** | 0.867 | 0.0 | 0.0 | **0.9255** | 2.4 | True |
| 020000 | **0.0337** | 0.5196 | 0.893 | **0.800** | 0.0 | 0.0 | 0.9171 | 2.4 | True |

**joint_mae 在 015000 与 020000 处打平(0.0337)**,由次级判据决定:015000 的夹爪 F1 更高(0.9255 vs 0.9171)、抓取时序误差更小(0.627 s vs 0.893 s)。**选 015000。**

> 红/绿两次都选中 020000,本次选 015000 —— 是这批数据的实际指标决定的,不是流程差异。

## 6. 部署到 Orin

`/home/nvidia/work/telop/models/smolvla_20260904_screwdriver_into_box/`(2.8 GB)

- `checkpoint/` 865 MB,来自 `checkpoints/015000/pretrained_model`
- `vlm/` 1.9 GB

### 传输与校验

**7 个文件的 SHA256 双端逐一相同**,含 `model.safetensors = a07247422dc277abcb0a27b3bccb84e3ca498a22fcd7f48feb6a9612a69238cb`。

**VLM 未跨网传输**:各 bundle 的 `vlm/` 内容哈希相同(`574ef520d94038c2`,13 文件),从绿 bundle 就地复制,复制后重新校验一致,省去 1.9 GB 传输。

### 路径本地化

`config.json` 的 `vlm_model_name` 与 `policy_preprocessor.json` 的 `tokenizer_name` 原本指向 HuggingFace 仓库名,离线环境无法加载,已改写为 bundle 内 `vlm/` 绝对路径(`tokenizer_name` 藏在 `steps` 列表内,用递归遍历处理)。改写前后 `model.safetensors` 的 SHA256 **完全一致**。as-received 原文件备份在 `manifest/`。

`train_config.json` 中仍有一处 HuggingFace 仓库名,与红/绿 bundle 相同 —— 它是训练记录,不参与 `from_pretrained` 加载,未改动。

### 离线加载 smoke test:通过

在 Orin 实际推理环境(`venvs/smolvla-orin`,`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`):

- config OK,`type = smolvla`,`chunk_size = 50`,`n_action_steps = 50`
- VLM 从 bundle 内本地目录加载,无网络访问
- 策略加载 OK,参数 450.0M,**权重全部 finite**
- 预处理器不含 HuggingFace 仓库名,已指向本地 `vlm/`
- `SHA256SUMS` 20 条全部通过

**既有 bundle 未受影响**(部署后抽查):绿 `630e0bbd…`、红 `24e6505a…`,与各自交付时记录一致。

## 7. 关于机器人控制的声明

**数据检查、删除、清洗、训练、部署准备全程,没有上电、没有使能、没有进入伺服模式、没有控制夹爪、没有发送任何机械臂动作。** 部署与 smoke test 期间 Orin 上机器人相关进程数为 0。真机测试须由机器人旁的操作员按受控 ARM/MOVE 流程另行进行。

---

## 附:文件清单

仓库内(`tools/screwdriver/`):

| 文件 | 作用 |
| --- | --- |
| `clean_screwdriver.py` | 清洗(Orin 版,用系统 ffmpeg) |
| `clean_screwdriver_a800.py` | 同上(a800new 版,用 imageio 自带 ffmpeg) |
| `verify_screwdriver.py` | 核验,可对任一份拷贝运行 |
| `launch_screwdriver_train.sh` | GPU 3 训练启动 |

机器上:

- 数据集 `a800new:/ssd/zwwl_user2/parcel_smolvla/20260904_screwdriver_clean_50/`
- 备份副本 `orin:/home/nvidia/work/telop/screwdriver_clean_50_20260904/`
- bundle `orin:/home/nvidia/work/telop/models/smolvla_20260904_screwdriver_into_box/`
- ep250 备份 `orin:/home/nvidia/work/telop/onearm_Tele/_deleted_ep250_20260904/`
