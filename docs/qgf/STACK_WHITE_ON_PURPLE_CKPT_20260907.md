# 堆叠白盒到紫盒 ckpt:检查、清洗、训练与部署

日期:2026-09-07
运行 ID:`stack_clean_50_20260907`
任务串:`把白色盒子叠在紫色盒子上`
机器 / 卡:a800new,**物理 GPU 0**(uuid `GPU-ffb25077-5d78-c9cc-21d9-ca8edbdd6406`)

---

## 1. 数据来源与检查

摇操记录在共享数据集 `onearm_Tele/lerobot_dataset`(Orin)。堆叠这批是 **ep250–299 共 50 条**,
是该数据集当前的末尾(总 300 条),不需要删任何东西,也不需要重排索引。

50 条逐条检查,结构无问题:

- 时间戳全部严格递增、`t0 = 0`、无 NaN/Inf、state/action 均为 8 维
- 任务串 50 条完全一致,单一 camera keyset
- 时长 med 45.6 s(29.6–74.7),首次闭合 med 17.6 s,释放 med 32.2 s
- 释放后尾巴 med 12.8 s,最长 34.3 s —— 每一条都有真实的空转要截

### 与螺丝刀那批的两处不同(都是这批更简单)

| | 螺丝刀 ep200–249 | **堆叠 ep250–299** |
| --- | --- | --- |
| 夹爪周期 | 3 条重抓(215/245/249),必须用**最后一次**释放 | **50 条全是 1 个周期**,首次即最后一次 |
| J5 分支 | 单分支(极差 0.864 rad) | 单分支(极差 **0.597 rad**,最大间隙 0.097,全负) |

清洗脚本仍用"最后一次释放"的写法 —— 它在两种情况下都正确,而且**逐条记录周期数**,
将来某批出现重抓时不会静默改变切割的含义。核验脚本新增一条门禁:保留下来的每一条必须仍是恰好 1 个周期。

## 2. 传输

Orin → 本机 → a800new 流式管道(`tar | ssh | tar`)。**没有在 Orin 上安装任何私钥** ——
Orin 到 a800d 网络可达(返回 `Permission denied (publickey)`,说明 TCP 与 SSH 协商都成功),
但按既定规矩私钥只能由用户手工搬运,所以走本机中转。

```
202 个文件(50 data parquet + 50 meta parquet + 100 mp4 + info.json + tasks.parquet)
2,283,911,447 B = 2.12 GiB
3 分 15 秒
双端逐文件 SHA256 全同
```

传输前抽查了 5 条的视频帧数与 parquet 行数,全部相等。

## 3. 清洗

规则:`[0, 最后一次确认释放 + 7.0 s]`(或到 episode 结尾,取较短者)。
夹爪事件判据与红/绿/螺丝刀一致:open ≤ 0.15、close ≥ 0.85、15 Hz 上连续 5 拍确认。

保留 50 / 50,拒绝 0。

```
成品          4.21 GiB / 58,895 帧
保留比例      85.2%(原 69,101 帧)
段长          min 721 / med 1178 / max 1747 帧
切点 t_cut    min 24.0 / med 39.2 / max 58.2 s(原时长 min 29.6 / med 45.6 / max 74.7 s)
释放后尾巴    min 6.97 / med 7.00 / max 7.00 s
划分          45 训 / 5 留(seed 1000,val = [9, 24, 29, 40, 49])
```

J5 做**门禁而非修正**:检测到分支分裂就直接失败,而不是盲目施加一个此处不需要的 ±2π。
实测 spread 0.597 rad、gap 0.097 rad、pos 0 / neg 50,单分支确认。

### 核验

`verify_stacking.py` 全部通过:结构完整性、时钟从 0 起算且严格递增、`frame_index` 连续、
全局 `index` 跨 episode 连续、**视频帧数逐条等于 parquet 行数**、截断点在最后一次释放之后、
**每条恰好 1 个夹爪周期**、J5 单分支、任务串三处一致、45/5 不相交、50 条 parquet SHA256 复算。

## 4. 训练

配方与红/绿/螺丝刀完全一致,只改数据集、repo_id、输出目录:

| 项 | 值 |
| --- | --- |
| 机器 / GPU | a800new,**物理 GPU 0**(启动时 0–3 全空,别人的作业在 4–7) |
| init 权重 | `~/parcel_smolvla/models/smolvla_parcel_init` |
| batch / steps | 64 / 20000 |
| save_freq / seed | 1000 / 1000 |
| 训练集 | 45 条,`num_frames=53514`(5 条留出既不进训练也不进归一化统计) |
| 参数量 | 可训练 99.88M / 总 450M |
| 耗时 | 4 小时 40 分(18:56 → 23:44),`End of training` 正常收尾 |

锁卡是**实证**的:训练启动后抓 `nvidia-smi` 的 PID → UUID → index,pid 3914717 落在物理 0 号卡,
别人的 pid 3895001 留在 4–7 号,全程不共卡。

### 配方漂移检查(用记录的产物,不是启动脚本)

checkpoint 001000 落盘后,把 lerobot 自己写的 `train_config.json` 与螺丝刀那次逐字段比对:

```
130 个字段,差异 3 处:
  dataset.repo_id      local/stack_white_on_purple   vs  local/screwdriver_into_box
  dataset.root         .../20260907_stack_clean_50/  vs  .../20260904_screwdriver_clean_50/
  output_dir           .../train_stacking            vs  .../train_screwdriver
```

其余全同,含 `dataset.episodes`(两次都是同一组 45 个索引)。

## 5. checkpoint 选择

`seq_eval.py` 在 5 条留出 episode 上按 15 Hz 回放评估(replan 25),准则与红/绿/螺丝刀一致:
joint_mae 为主,夹爪指标为次。

| ckpt | joint_mae | F1 | close_s | open_s | pass |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 002000 | 0.0314 | **0.966** | 0.600 | 0.373 | True |
| 005000 | 0.0316 | 0.946 | 0.760 | 0.653 | True |
| 010000 | 0.0289 | 0.949 | 0.733 | **3.88** | True |
| 015000 | 0.0292 | 0.945 | 0.840 | 0.733 | True |
| **020000** | **0.0284** | 0.943 | 0.733 | 0.760 | True |

**选 020000。** 它是 joint_mae 的**严格最小值**,不像螺丝刀那次是打平后靠次级判据决出的。

需要说明的两点:

- **010000 不能选**:它的 open 误差 3.88 s 完全由 ep24 单条的 **−16.47 s** 拖出来,
  是一次漏检/极晚的开爪,不是整体表现。
- **002000 的夹爪 F1 最高(0.966)**,但 joint_mae 最差(0.0314)。按既定准则 joint_mae 优先,
  所以没有选它;如果上机发现夹爪时序是瓶颈,002000 是值得回头试的备选。

> **ep49 是这批里最难的一条**:五个 checkpoint 上它的 joint_mae 都是最高(0.040–0.043),
> 且其中三个 checkpoint 上它的开爪误差是 +1.73 s。上机首跑请重点看这一条对应的场景。

## 6. 部署到 Orin

`/home/nvidia/work/telop/models/smolvla_20260907_stack_white_on_purple/`(2.8 GB)

- `checkpoint/` 865 MB,来自 `checkpoints/020000/pretrained_model`
- `vlm/` 1.9 GB
- `manifest/` 改写前的两个 JSON 原件
- `logs/` 训练日志、评估日志、划分与清洗记录、三个脚本

### 传输与校验

**checkpoint 的 7 个文件 SHA256 双端逐一相同**,含
`model.safetensors = 069aae3bdfe7da9d1fdfdb94dcefb9b0359ebf6a7de6c2f6ccf00e999282b901`。

**VLM 未跨网传输**:Orin 上三个既有 bundle 的 `vlm/` 内容哈希完全相同,
从螺丝刀 bundle 就地复制,复制后重新算树哈希一致(`a78d5ef526a9721f`,13 文件),省去 1.9 GB 传输。

### 路径本地化

`config.json` 的 `vlm_model_name` 与 `policy_preprocessor.json` 的 `tokenizer_name` 原本是
HuggingFace 仓库名,已改写为 bundle 内 `vlm/` 绝对路径。**`tokenizer_name` 藏在 `steps` 列表里**,
只递归字典会漏掉 —— 用同时递归 dict 和 list 的遍历处理。改写前后 `model.safetensors` 的 SHA256 完全一致。
as-received 原文件备份在 `manifest/`。

`train_config.json` 中仍有一处 HuggingFace 仓库名,与红/绿/螺丝刀 bundle 相同 ——
它是训练记录,不参与 `from_pretrained` 加载,未改动。

### 离线加载 smoke test:全部通过

在 Orin 实际推理环境(`venvs/smolvla-orin`,`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`,
并在进程内拦截一切非回环出站连接):

- 结构 4 个子目录齐全,`checkpoint/` 7 个文件
- config OK,`type = smolvla`,`chunk_size = 50`,`n_action_steps = 50`
- `input_features` = chest / wrist_right / state,两路相机键名正确
- VLM 从 bundle 内本地目录加载,无网络访问
- 策略加载 OK,参数 **450.0M**,**权重全部 finite**
- 预处理器不含 HuggingFace 仓库名,已指向本地 `vlm/`
- 写入 `SHA256SUMS` 30 条
- 没有导入任何 robot/ROS 模块

**既有资产未受影响**(部署后复算):四个 bundle 各自的 `SHA256SUMS` 全过
(螺丝刀 20 条、绿 20 条、红 49 条、本次 30 条),`models/qgf/` 下 5 个 critic 哈希与交付时记录逐一相同。

## 7. 途中踩到的两个坑

**a800new 的根分区 100% 满、0 字节可用。** 训练本身没事(checkpoint 写在 `/ssd`,还有 910 GB),
但 `seq_eval.py` 里 `Dataset.from_parquet` 要往 `~/hf_cache/datasets` 写 arrow 中间文件,
而 `/home` 在满掉的那块盘上,直接 `OSError: [Errno 28]`。
**这和今早 Orin 那次摇操导出失败是同一个机制**:一次小写入打向了错误的文件系统。

第一次修的时候我把 `HF_HOME` 整体指到 `/ssd`,结果**离线找不到 VLM** ——
`~/hf_cache/hub` 里缓存着 SmolVLM2,被我一起改走了。
正确做法是**分开**:hub 缓存只读、留在原处;datasets 缓存要写、指到 `/ssd`;`TMPDIR` 一并指过去。

## 8. 顺带查清的几件事(对以后所有任务都适用)

1. **LeRobot 的归一化会绕过 `--dataset.episodes`。** `LeRobotDataset.__init__` 调
   `load_stats(self.root)` 读**全局** `meta/stats.json`,无视 episode 过滤。
   所以清洗脚本必须自己把那个全局文件**只按训练集**算 —— 我们的确实如此,
   实测本批 `stats.count = 53514` 正是 45 条的帧数(全 50 条是 58895)。
   要是哪天有人图省事按全量算,再小心的启动参数也挡不住泄漏。

2. **`meta/info.json` 的 `splits` 不是划分。** 它是连续区间字段,只能写 `{"train": "0:50"}`,
   于是把 5 条留出也标成了训练集。本次运行没事(launcher 显式传了 45 个索引),
   但任何相信它的下游都会静默训到留出集。**已修**:清洗脚本现在在数据旁边写
   `HELDOUT_README.txt` 点名留出集,核验脚本反向解析比对。**权威是 `split_manifest.json`。**

3. **训练期间 lerobot 根本不做验证。** `env = None`,所以 `eval_freq` 形同虚设,
   5 条留出在训练过程中什么也没做 —— 它的全部价值在训练结束后由 `seq_eval.py` 离线回放兑现。
   这四次运行都是如此。

4. **LR 调度跑不完。** `cosine_decay_with_warmup`,warmup 1000、**decay 30000**,但 `steps = 20000`,
   所以停在余弦下降的三分之二处,末轮 LR 约 1e-5 而不是设定的 2.5e-6。四次运行一致,不影响可比性,
   但这是个此前没人指出的配置事实。

5. **只训练 action expert 与 state projection**(`freeze_vision_encoder = true`、
   `train_expert_only = true`),可训练 99.88M / 总 450M。

6. **"四个 ckpt 配方一致"这句话只对训练配置成立,对数据管线不成立。**
   `cleaning_report.json` 显示红(clean)/螺丝刀/堆叠用的都是 `[0, 最后一次释放 + 7 s]`、头部不动;
   **绿包裹用的是完全不同的规则**(per-episode 自适应起点 + 不截尾)。写对比表必须说明。
   另外**红包裹有两个 run 且不等价**:`red_parcel_out_table` 的 J5 是 −2.329/+5.121
   (两个分支都在、未修),`red_parcel_clean` 是单分支;归一化是全数据集 MEAN_STD,
   那一个通道在两者里完全不同,**out_table 那份不能拿来比**。

7. a800new 的 lerobot 0.4.4 是**本地 fork**(config 里有上游没有的 `rabc_*` 键,本次 `use_rabc=False` 未生效),
   **fork 的 git SHA 没有记录在任何 run 目录里** —— 这是个复现性缺口,值得补。

## 9. 关于机器人控制的声明

**数据检查、传输、清洗、训练、部署准备全程,没有上电、没有使能、没有进入伺服模式、
没有控制夹爪、没有发送任何机械臂动作。** 采集会话在本流水线启动前已由操作员自行结束。
smoke test 在进程内拦截了一切非回环出站连接,并确认没有导入任何 robot/ROS 模块。
真机测试须由机器人旁的操作员按受控 ARM/MOVE 流程另行进行。

---

## 附:文件清单

仓库内(`tools/stacking/`):

| 文件 | 作用 |
| --- | --- |
| `clean_stacking_a800.py` | 清洗(a800new 版,用 imageio 自带 ffmpeg) |
| `verify_stacking.py` | 核验,可对任一份拷贝运行 |
| `launch_stacking_train.sh` | GPU 0 训练启动 |

机器上:

- 数据集 `a800new:/ssd/zwwl_user2/parcel_smolvla/20260907_stack_clean_50/`
- 源副本 `a800new:/ssd/zwwl_user2/stack_src/lerobot_dataset/`
- bundle `orin:/home/nvidia/work/telop/models/smolvla_20260907_stack_white_on_purple/`
