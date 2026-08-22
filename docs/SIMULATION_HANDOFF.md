# Armstrong / SmolVLA 仿真交接说明

> 面向：搭建本项目仿真环境的同学。  
> 更新：2026-08-22。  
> 原则：本文件只写已经从仓库、配置和已保存实验产物核实到的信息；没有可靠来源的内容明确标成“待确认”。

## 1. 先给结论

此前的仿真主线是 **SmolVLA 0.45B + LIBERO（MuJoCo）**，而不是把当前 Armstrong 真机完整地导入某个仿真器。历史上还准备了 LIBERO-plus 与 LIBERO-PRO 的兼容评估入口。

- 推荐首先复现：`LIBERO / libero_spatial / task id 3`。
- 仿真 policy checkpoint：官方 `lerobot/smolvla_libero`（历史 manifest 有时将同一来源记为 `HuggingFaceVLA/smolvla_libero`）。
- 真机运行 checkpoint：本地训练得到的 `smolvla_onearm_20k_20260805`，它与 LIBERO checkpoint **不是同一个模型**，也不能直接替换。
- 当前真机实际为：右侧 **7 关节机械臂 + 1 个二值夹爪通道**，推理观测为胸部和右腕两路 RGB。
- 现有 `walle.urdf` / `robot_7dofs.urdf` 是有价值的机械臂几何起点，但不是当前“右臂 + CTAG2F120 夹爪 + 两相机 + 已标定 TCP”的最终统一模型。正式高保真仿真仍需补齐 CTAG2F120、安装转接件、TCP 和相机外参。

## 2. 代码和资料入口

| 内容 | 地址 / 说明 |
|---|---|
| SmolVLA、QGF、LIBERO 仿真、真机 policy bridge | [SmolVLA-with-QGF](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF) |
| 真机 ROS 2 控制、Windows 主臂遥操、URDF/mesh/CAD | [One-Arm-Teleoperation](https://github.com/jason0925pig-rgb/One-Arm-Teleoperation) |
| QGF 原始参考实现 | [guided-action-flow](https://github.com/chenchaosheng24-design/guided-action-flow) |
| 本仓库的仿真复现步骤 | [`qgf/docs/smolvla_qgf_simulation_reproduction.md`](../qgf/docs/smolvla_qgf_simulation_reproduction.md) |
| 真机 Q critic 的事实性训练报告 | [`docs/qgf/REAL_ROBOT_Q_CRITIC_TRAINING_FACTUAL_REPORT.md`](qgf/REAL_ROBOT_Q_CRITIC_TRAINING_FACTUAL_REPORT.md) |

本文件不包含内网 IP、密码、私钥、机器人 serial number、模型权重或数据集文件；它们不应提交 Git。

## 3. 历史上使用过的仿真环境、checkpoint 与任务

### 3.1 可复现的推荐环境

| 项目 | 已固定/推荐值 |
|---|---|
| simulator | MuJoCo，通过 LIBERO |
| baseline policy | `lerobot/smolvla_libero`，SmolVLA 0.45B |
| 首个 smoke task | `libero_spatial`，task id `3` |
| action chunk | 50 steps |
| 固定的上游版本 | LeRobot `6a788fbdb02cabfae60f7408636945df0b1eafa0`；LIBERO `8f1084e3132a39270c3a13ebe37270a43ece2a01`；LIBERO-plus `4976dc30028e805ff8094b55501d532c48fec182`；LIBERO-PRO `eafdb809426b13153aa1e4c42d6601844217dfec` |
| headless rendering | `MUJOCO_GL=egl` |

仿真安装、checkpoint 下载、rollout 收集、critic 训练与 baseline/QGF 公平评测的具体命令已经写在 [`qgf/docs/smolvla_qgf_simulation_reproduction.md`](../qgf/docs/smolvla_qgf_simulation_reproduction.md)。按该文档使用固定 commit，不要直接混用最新版 LeRobot/LIBERO API。

### 3.2 现有历史仿真产物（不是统一的正式 benchmark）

`E:/AAA__Github_Project/World_Model` 现在只是**历史输出和论文资料**，不再是源代码仓库。核实到的历史记录如下：

| 产物 | 环境 / checkpoint | 已记录任务或规模 | 已记录结果 | 解释 |
|---|---|---|---|---|
| `atomic_qcritic_compare_task0to4_ep1_20260714` | `libero_10`，SmolVLA LIBERO | task 0–4，各 1 条 | baseline 5/5；qcritic 5/5 | 样本仅 5 条，不能说明统计优势 |
| `atomic_qcritic_compare_task5to9_ep1_20260714` | `libero_10` | task 5–9，各 1 条 | baseline 5/5；qcritic 5/5 | 同样只是 smoke/演示 |
| `atomic_qcritic_libero90_mid20_seed11_20260714` | `libero_90` | 20 个 task、各 1 条 | baseline 11/20；qcritic 11/20 | 该组未显示 Q 提升 |
| `atomic_qcritic_libero90_hard15_seed11_20260714` | `libero_90` | 15 个困难 task、各 1 条 | baseline 0/15；qcritic 0/15 | 两者均失败 |
| `atomic_qcritic_libero90_margin2_key9_seed11_20260714` | `libero_90` | 9 个 task、各 1 条 | baseline 1/9；qcritic 1/9 | 两者相同 |

更早的 `q_guided_pipeline` manifest 使用了标签 `libero_10`、`libero_10_object`、`libero_object`、`libero_object_object`，并明确注明当时使用的是 **proxy target-action Q critic**，不是正式 learned critic。因此这些早期结果只能用于开发回溯，不能当论文结论。

当前 `qgf/README.md` 记录的、规模较大的仿真 anchors 如下（仍应在同一固定环境中重新跑一遍再作为新实验的对照）：

| 设置 | checkpoint | 样本量 | 已记录成功率 |
|---|---|---:|---:|
| LIBERO vanilla，4 suites × 5 tasks × 5 episodes | `lerobot/smolvla_libero` | 100 | 65/100 = 65.0% |
| LIBERO-plus spatial subset，10 tasks × 5 episodes | `lerobot/smolvla_libero_plus` | 50 | 39/50 = 78.0% |
| LIBERO-PRO zero-shot，4 suites × 5 tasks × 5 episodes | `lerobot/smolvla_libero` | 100 | 1/100 = 1.0% |
| `libero_spatial` task 3，seed 3000 baseline | `lerobot/smolvla_libero` | 50 | 34/50 = 68.0% |
| 同一 task/seed，K=3 critic + adaptive gate | 同上 + 3 critics | 50 | 41/50 = 82.0% |
| `libero_spatial` task 3，seed 4000 baseline | `lerobot/smolvla_libero` | 50 | 41/50 = 82.0% |
| 同一 task/seed，K=3 critic + adaptive gate | 同上 + 3 critics | 50 | 43/50 = 86.0% |

已保存 `libero_10` task 0–4 的英文任务为：

1. `put both the alphabet soup and the tomato sauce in the basket`
2. `put both the cream cheese box and the butter in the basket`
3. `turn on the stove and put the moka pot on it`
4. `put the black bowl in the bottom drawer of the cabinet and close it`
5. `put the white mug on the left plate and put the yellow and white mug on the right plate`

## 4. 真机平台：已确认的事实与待确认项

### 4.1 机器人、自由度和夹爪

| 字段 | 当前信息 | 可信度 / 备注 |
|---|---|---|
| 平台称呼 | 团队内部称为“Armstrong”右臂平台 | **名称已知；商业整机精确型号待厂商/资产标签确认** |
| 低层机械臂 SDK | JAKA/EDG 风格 SDK，控制器通过 `servo_controller` ROS 2 包访问 | 从现有 C++ 控制代码和运行日志核实；不等于已确认机械臂商业 SKU |
| 操作臂 | 右臂，7 个转动关节 | 已核实 |
| 末端夹爪 | CTAG2F120，双指夹爪，串口 Modbus 风格控制 | 型号已核实；供应商品牌名称仍建议由采购/厂商资料确认 |
| 当前 policy state/action | 7 关节 + 1 gripper = 8 维 | 已核实 |
| 灵巧手 | 旧 URDF 中存在左右灵巧手 visual mesh | **不属于当前 CTAG2F120 真机末端，不能拿来替代当前夹爪仿真** |

### 4.2 模型文件、URDF、mesh 与 CAD

| 资源 | 仓库内位置 | 可用于什么 | 重要限制 |
|---|---|---|---|
| 双臂/历史机器人描述 | `One-Arm-Teleoperation/walle_description/urdf/walle.urdf` | 基座、左右 JAKA 风格臂、相机/旧末端 visual 的起点 | 含旧灵巧手，非当前 CTAG2F120 统一模型 |
| 7 DOF 历史描述 | `One-Arm-Teleoperation/teleop_tupian/teleop/conf/robot_7dofs.urdf` | 右臂关节链、link 变换、collision/visual mesh 的最佳起点 | 速度均写为 0，关节范围是候选 CAD/显示范围，不能直接当首次真机安全软限位 |
| 机械臂 STL | `One-Arm-Teleoperation/walle_description/meshes/` | 右臂 `base_link_jaka_right.STL`、`r1.STL`…`r7.STL` 等 | 可导入 MuJoCo/Isaac/MoveIt；应先做单位和坐标检查 |
| 实际夹爪 CAD | `One-Arm-Teleoperation/changingtek-robotics-ctag2f120-gripper-cad-model.step` | 为 CTAG2F120 制作新的 visual/collision mesh | 需自行转换/简化为 collision mesh，并补齐安装位姿 |
| 主臂（不是从臂）CAD | `One-Arm-Teleoperation/mechanical/Feetech_servo/Config3_STL/` 等 | ZLink2 主臂机械结构参考 | 与 Armstrong 从臂仿真无关 |
| Xacro | 未发现可作为当前整机统一描述的 Xacro | — | 待新建 |

**建议的仿真交付件：** 以 `robot_7dofs.urdf` 为基础，单独创建 `armstrong_right_ctag2f120.urdf.xacro`：右臂 7 DOF + CTAG2F120 mesh/collision + flange→gripper 固定变换 + TCP + 两台相机光学 frame。生成后的 URDF 才是后续 motion planning、sim2real 和 camera extrinsic 标定应使用的唯一真相源。

### 4.3 关节名称、顺序、范围与正方向

真机 ROS 2 控制层发布的名字和顺序是：

```text
right_joint1, right_joint2, right_joint3, right_joint4,
right_joint5, right_joint6, right_joint7
```

旧 URDF 中同一右臂链的名字是 `r-j1` … `r-j7`；映射关系为按序一一对应。URDF 各 revolute joint 的轴均为其局部 frame 的 `+Z`。因此**仿真正方向应先采用 URDF local +Z，再通过一次真实小角度校准验证**，不要只按“世界坐标的顺/逆时针”猜测符号。

| 控制序号 / ROS name | URDF name | 候选下限 (rad) | 候选上限 (rad) | 候选范围 (deg) |
|---|---|---:|---:|---:|
| J1 / `right_joint1` | `r-j1` | -6.2832 | 6.2832 | -360 … 360 |
| J2 / `right_joint2` | `r-j2` | -1.8325 | 1.8325 | -105 … 105 |
| J3 / `right_joint3` | `r-j3` | -6.2832 | 6.2832 | -360 … 360 |
| J4 / `right_joint4` | `r-j4` | -2.5307 | 0.5235 | -145 … 30 |
| J5 / `right_joint5` | `r-j5` | -6.2832 | 6.2832 | -360 … 360 |
| J6 / `right_joint6` | `r-j6` | -1.8325 | 1.8325 | -105 … 105 |
| J7 / `right_joint7` | `r-j7` | -6.2832 | 6.2832 | -360 … 360 |

这些数值目前来自 checked-in `full_teleop_attended.yaml` 与历史 URDF。它们在真机控制中作为**宽安全边界**使用，但 J1/J3/J5/J7 的 ±360° 很可能是 CAD 显示范围；厂商真实软限位、连续旋转能力、最大速度与最大加速度仍必须以控制器/厂家资料为准。现有配置的 `0.90 rad/s` 与 `0.60 rad/s²` 是本次 attended teleoperation 限制，不是设备铭牌上限。

### 4.4 CTAG2F120 的控制抽象

当前程序不是直接输出夹爪几何开口距离，而是一个归一化状态通道：

| 字段 | 值 |
|---|---|
| policy action 第 8 维 | `gripper_closed_normalized`，范围 [0, 1]；0=开、1=关 |
| 实体端点 | open position = 2000；closed position = 12000（设备 position units，非米） |
| ROS 夹爪命令 | Bool；`true` 表示 open，注意与 policy 的 `closed_normalized` 方向相反 |
| 当前 checked-in 力/速度设置 | force=75%、open speed=100%、closed speed=75% |
| 夹爪 TCP | 待确认；不能由“端点 position=2000/12000”推出来 |

因此仿真里可以先建一个 1-DOF prismatic 或 open/close gripper proxy；只有需要接触质量、抓取稳定性或精确端点位置时，才需要把 STEP 变成完整可动夹爪模型。

## 5. ROS 2 驱动和控制接口

### 5.1 ROS 2 版本与控制包

- 当前新机部署目标是 Ubuntu 22.04 + **ROS 2 Humble**。
- 真机安全控制节点：`servo_controller/safe_one_arm_servo`。
- 夹爪安全节点：`servo_controller/safe_gripper_controller`。
- SmolVLA ROS 2 adapter：`lerobot_robot_armstrong_ros2`。
- 当前主控为 Orin；它接收 policy action 后才对机器人 SDK 发出控制指令。

核心已用 topic / service：

| 类别 | 名称 | 类型/含义 |
|---|---|---|
| joint feedback | `/right_arm/joint_states` | `sensor_msgs/JointState`，7 rad joint positions |
| joint target | `/right_arm/teleop_joint_command` | 7 rad joint target |
| executed joint command | `/right_arm/executed_joint_command` | 安全处理后的实际执行目标 |
| gripper command | `/right_arm/gripper_command` | 归一化/离散开闭接口 |
| executed gripper | `/right_arm/executed_gripper_command` | 实际接受的夹爪命令 |
| gripper contact | `/right_arm/gripper_contact` | 夹爪接触/力矩到达相关状态 |
| motion gate | `/right_arm/motion_enabled` | 运动门状态 |
| model gate | `/smolvla/set_enabled` | 启停 model action 的服务 |
| software stop | `/teleop/stop_request` | 软件 STOP 请求 |

仿真控制器需要实现等价的 7D position control + 1D gripper control。若直接接现有 SmolVLA adapter，至少要提供 `/right_arm/joint_states`、对应相机 topic 和 joint command 消费端。

## 6. 相机、图像、动作、反馈与时间

### 6.1 真机正在使用的观测相机

当前训练/rollout pipeline **只选择两台相机**，顺序固定为 `[chest, wrist_right]`：

| 顺序 | ROS topic | 设备序列号 | 安装位置 | 规格 |
|---:|---|---|---|---|
| 1 | `/camera_chest/color/image_raw/compressed` | `CP8284100034` | 胸部，俯视桌面/工作区 | RGB compressed JPEG，1280×720，30 FPS |
| 2 | `/camera_wrist/color/image_raw/compressed` | `CPCD75300083` | 右手腕/末端 | RGB compressed JPEG，1280×720，30 FPS |

其它已安装胸部相机不在当前 dataset profile 内；头部相机当时尚未装入当前 pipeline。内参、畸变参数、精确外参、相机坐标相对法兰/TCP 的变换：**未在仓库中找到可交付的标定文件，待标定/索取。**

### 6.2 动作与采样频率

| 层级 | 内容 | 频率/时间基准 |
|---|---|---|
| 两路视频 | 胸部、右腕 RGB | 30 FPS，原始 JPEG archive + MP4 |
| SmolVLA policy | 8D action，预测 50-step chunk | 15 Hz action timebase（50 步 chunk） |
| robot servo | 7D joint position 接口 | 125 Hz / 8 ms 内部 SDK servo loop |
| robot feedback publish | `JointState` | 20 Hz 配置值 |
| action unit | 前 7 维为 radian joint position targets；第 8 维为 [0,1] closed-normalized gripper | 非 TCP delta action |

**30 FPS video 与 15 Hz action 并不冲突。** 每份数据都必须靠 message header timestamp/记录时间对齐；训练 policy/Q critic 时以明确的目标时间戳选取相邻或最近图像，而不是默认“第 n 帧视频=第 n 个动作”。如为简化新仿真数据，可统一使仿真 control 与 observations 同为 15 Hz，或保持 30 FPS 图像并明确每两个 video frame 采样一帧到 policy timebase。

### 6.3 必须保存的时间戳和数据

每条 rollout 至少保留：

1. 每张 RGB 图像：ROS `header.stamp`、frame index、camera name。
2. 每一条机器人 state：`JointState.header.stamp`、7D joint position、gripper state/contact。
3. 每一条 action：policy 生成时间、原始 policy action、guarded/safe action、底层实际发送 action 和执行反馈时间。
4. 每个 episode：开始/结束 UTC、语言 task、随机种子、success/failure、`done`/`terminated`/`truncated` 原因。

现有 QGF 真实 rollout 除视频外还会保留原始 `normalized_policy_chunks`、policy observation、raw action、guarded action、executed action、joint/gripper/contact、timestamps 等文件。不要只保存最终 MP4 与 parquet。

## 7. SmolVLA 数据与归一化参数：必须随 checkpoint 一起保存

真机当前启动时引用的模型 bundle 名称为 `smolvla_onearm_20k_20260805`。它是“矿泉水放入纸箱”右臂任务的本地 fine-tuned SmolVLA bundle；训练数据是本项目采集的 LeRobot-format 双 RGB + 8D state/action 轨迹。**完整数据集路径和准确训练集清单不在 Git 中，必须从模型/数据存储机单独交付。**

必须成套归档下列项目，缺一项都可能导致仿真或真机动作尺度错误：

- checkpoint 权重和它的 `config.json` / processor config；
- checkpoint 使用的 dataset statistics / normalization 字典（state/action 的 mean/std 或 min/max 与 feature schema）；
- LeRobot dataset 的 `meta/info.json`、`meta/episodes*.jsonl`、尤其 `meta/episodes_stats.jsonl`；
- 训练使用的 action 定义（7 rad joint + gripper closed-normalized）、image preprocessing、camera key 顺序；
- checkpoint 文件 hash、数据集 revision、训练命令、Git commit；
- 对 QGF：`normalized_policy_chunks` 的归一化空间定义。Q guidance 是在 **SmolVLA sampler 的 normalized 50×8 action chunk** 中优化，不是在物理 rad/position units 上直接求梯度。

现有真实记录确认过的夹爪反归一化端点为 open=2000、closed=12000。仿真若用 [0,1] 夹爪 DOF，仍须维持 `0=open, 1=closed` 的 policy 语义。

## 8. 现有真机任务与成功率应如何引用

真机的主要任务语言为：

```text
把矿泉水放进纸箱里。
```

英文可写为：`Put the bottle of water into the cardboard box.`

真机 rollout 的成功率随光照、背景、起始姿态、episode 编号范围、baseline/QGF 模式和运行版本变化，不能在没有明确 cohort 的情况下给一个全局百分比。请从每轮 collector 生成的 episode metadata/summary 统计，并报告：

```text
mode, perturbation, checkpoint hash, Q critic hash, beta,
successes / completed episodes, excluded/discarded episodes, seeds or episode IDs.
```

历史 LIBERO 成功率见第 3.2 节，它们不能外推为 Armstrong 真机成功率。

## 9. 给仿真同学的最短执行路线

1. 克隆 [SmolVLA-with-QGF](https://github.com/jason0925pig-rgb/SmolVLA-with-QGF)，严格按 `qgf/docs/smolvla_qgf_simulation_reproduction.md` 固定 LeRobot/LIBERO commit。
2. 下载 `lerobot/smolvla_libero`，先跑 `libero_spatial` task 3 单条 baseline；确认 `eval_info.json` 与视频生成。
3. 跑同一 task 的 50 条 baseline rollout，检查 state、`action_policy`、`action_env`、reward、success、done 都存在。
4. 训练 critic 后，用相同 task、same seed、same episode count 运行 baseline/QGF。
5. 只有需要做 Armstrong 数字孪生时，再把 `robot_7dofs.urdf` + `walle_description/meshes` + CTAG2F120 STEP 组合成新的统一 URDF，并标定 flange→gripper→TCP 与 cameras。
6. 将仿真关节顺序严格映射成 `right_joint1`…`right_joint7`，首先使用第 4.3 节范围，但在真机首次接入前由控制器/厂家确认真实软限位。

## 10. 仍需由硬件方提供/确认的清单

- [ ] Armstrong 整机和右臂的准确商业品牌、型号、序列号、厂家手册。
- [ ] 厂家认可的 7 轴软限位、最大速度、最大加速度、是否连续旋转。
- [ ] CTAG2F120 的厂家资料：行程、最大/可设定夹持力、力矩到达语义、碰撞模型。
- [ ] 法兰到 CTAG2F120 base 的 6D 固定变换、TCP 定义和工具质量/惯量。
- [ ] 胸部与右腕相机的内参、畸变、相对 robot base/flange/TCP 的外参。
- [ ] 真实用于 fine-tune `smolvla_onearm_20k_20260805` 的数据集版本、训练命令和全部 normalization stats。
- [ ] 真机 controller API 对 position/velocity/servo mode 的精确语义和错误码文档。

在以上项目齐全之前，可以可靠复现 LIBERO 的算法流程，也可以建立用于接口联调的简化 Armstrong kinematic model；但不能声称已得到可用于碰撞、抓取力或精确视觉 sim2real 的高保真数字孪生。
