# Armstrong 单右臂 SmolVLA 部署说明

本文只适用于当前项目的单右臂配置：7 个 Armstrong 关节、CTAG2F120 夹爪、胸部 RGB 相机和右腕 RGB 相机。训练数据为 LeRobot v3、30 FPS、8 维状态与 8 维绝对动作，夹爪约定为 `0=张开、1=闭合`。

## 1. 部署边界

- 模型推理在独立 NVIDIA GPU 主机运行；机器人侧 Ubuntu 22.04 / ROS 2 Humble 只负责采集观测、执行安全检查和发布动作。
- 连接 ROS 客户端默认是“只观察”，不会发布动作。
- `start` 只启动相机和控制节点，不上电、不使能。
- `arm` 才会上电、使能、打开夹爪并进入伺服模式，但策略动作门仍关闭。
- `enable-policy` 会让机器人真正按照模型输出运动，必须由现场人员看护，并把实体急停放在手边。
- 训练结束后的离线验证不能代替第一次真机低速验证。

## 2. 固定的数据接口

| 内容 | ROS 2 接口或模型键 |
| --- | --- |
| 七轴状态 | `/right_arm/joint_states` |
| 夹爪执行状态 | `/right_arm/executed_gripper_command` |
| 胸部图像 | `/camera_chest/color/image_raw/compressed` → `observation.images.chest` |
| 右腕图像 | `/camera_wrist/color/image_raw/compressed` → `observation.images.wrist_right` |
| 七轴绝对目标 | `/right_arm/teleop_joint_command` |
| 夹爪目标 | `/right_arm/gripper_command` |
| 策略急停 | `/teleop/stop_request` |
| 策略动作门 | `/smolvla/set_enabled` |

模型的七轴动作单位是弧度，顺序严格为 `right_joint1` 至 `right_joint7`；第八维为夹爪开闭状态。不得交换顺序或把脉冲值直接送给模型。

## 3. 推理服务器

在保存部署包的 GPU 主机上运行。推理进程必须从部署包根目录启动，因为模型配置使用包内相对路径 `vlm`。

```bash
cd /path/to/smolvla_onearm_deployment

export SMOLVLA_PHYSICAL_GPU=4
export SMOLVLA_SERVER_HOST=0.0.0.0
export SMOLVLA_SERVER_PORT=8080
export SMOLVLA_VENV=/path/to/lerobot-v0.4.4-venv

/path/to/SmolVLA-with-QGF/tools/start_smolvla_policy_server.sh
```

机器人保持关机时，可以另开终端用数据集中的一帧验证完整 gRPC 推理链路：

```bash
CUDA_VISIBLE_DEVICES=4 /path/to/lerobot-venv/bin/python \
  deployment/tools/smoke_smolvla_policy_server.py \
  --server 127.0.0.1:8080 \
  --checkpoint checkpoint \
  --dataset-root /path/to/lerobot_dataset
```

看到 `SMOLVLA_POLICY_SERVER_SMOKE_OK` 才表示服务握手、模型加载、图像/状态预处理、动作推理、后处理和动作返回都已跑通。该检查不会导入 ROS，也不会连接机器人。

部署包中也提供了一条会自动启动并清理测试服务的命令：

```bash
SMOLVLA_PHYSICAL_GPU=4 SMOLVLA_VENV=/path/to/lerobot-venv \
  deployment/tools/test_smolvla_bundle.sh \
  "$PWD" /path/to/lerobot_dataset
```

这条命令在前台运行。终端关闭即停止推理服务。`CUDA_VISIBLE_DEVICES=4` 后，程序内部显示的 `cuda:0` 就是物理 4 号卡。

## 4. 机器人侧首次安装

在新机器人 Ubuntu 22.04 / ROS 2 Humble 上：

```bash
cd /home/nvidia/work/telop/SmolVLA-with-QGF
git pull --ff-only origin main

source /opt/ros/humble/setup.bash
colcon build --symlink-install \
  --packages-select servo_controller one_arm_teleop_bridge

mkdir -p third_party
# 把部署包中的 software/lerobot-v0.4.4 放到：
# third_party/lerobot-v0.4.4

./tools/setup_smolvla_client_ubuntu.sh
```

安装脚本只配置 CPU 侧客户端，不启动相机、不连接控制器，也不会上电或运动。

## 5. 第一次真机运行顺序

先确认机器人开机、急停已释放、右臂周围无人员和障碍物、夹爪为空、实体急停触手可及。用四个终端依次运行。

### 终端 A：模型服务器

按第 3 节启动，并保持窗口打开。

### 终端 B：机器人安全控制栈

```bash
cd /home/nvidia/work/telop/SmolVLA-with-QGF
./tools/ubuntu_smolvla_stack.sh start
./tools/ubuntu_smolvla_stack.sh status
```

此时机器人不应运动。

### 终端 C：ROS 2 与模型服务器客户端

```bash
cd /home/nvidia/work/telop/SmolVLA-with-QGF

export SMOLVLA_SERVER_ADDRESS=<推理服务器有线IP>:8080
export SMOLVLA_SERVER_MODEL_PATH=/path/on/server/smolvla_onearm_deployment/checkpoint
export SMOLVLA_TASK='把矿泉水放进纸箱里。'

./tools/start_smolvla_ros_client.sh
```

客户端连接成功后仍不会发布动作。确认 `/smolvla/status` 中 `action_enabled=0`。

### 终端 D：分阶段授权

```bash
cd /home/nvidia/work/telop/SmolVLA-with-QGF

# 会上电、使能、打开夹爪并进入伺服模式；尚不执行模型动作
./tools/ubuntu_smolvla_stack.sh arm

# 最后一次检查现场；这条命令之后机器人会运动
./tools/ubuntu_smolvla_stack.sh enable-policy
```

第一次只做低速短距离测试。任何异常先按实体急停，再运行停止命令。

## 6. 停止

```bash
cd /home/nvidia/work/telop/SmolVLA-with-QGF
./tools/ubuntu_smolvla_stack.sh disable-policy
./tools/ubuntu_smolvla_stack.sh stop
```

停止顺序为：关闭策略门并发布 STOP、退出伺服、关闭夹爪执行、机器人去使能、机器人下电、停止进程。实体急停始终高于软件停止。

## 7. 当前安全约束

- 策略开始位姿必须落在 50 组示范的起始位姿包络内，且夹爪必须张开。
- 每个模型目标必须为 8 个有限数值；NaN/Inf 会立即拒绝。
- 七轴目标被限制在示范任务包络内；明显越界会关闭策略门并发送 STOP。
- 任一关节的单次绝对目标与实时关节状态相差超过 `0.25 rad` 会拒绝执行。
- 控制器仍执行真实关节限位、速度/加速度斜坡、反馈超时、命令超时和控制周期看门狗。
- 首次策略部署速度为每轴 `0.10 rad/s`，加速度为 `0.30 rad/s²`。
- 夹爪模型值 `<=0.35` 判为张开，`>=0.65` 判为闭合，中间区域保持上一状态。

## 8. 上线前必须实机确认的事项

以下内容无法在机器人关机时完成，因此不能被离线训练替代：

1. 机器人侧能通过有线网络访问推理服务器的 `<IP>:8080`。
2. 两路相机名称、画面方向与训练数据一致，稳定达到 30 FPS。
3. 实时七轴状态顺序和单位与数据集一致。
4. 初始姿态检查能够通过，且模型动作门开启前机器人保持静止。
5. 先空载、低速、短时运行；验证 STOP、Ctrl+C 和实体急停。
6. 观察模型是否在任务外输出、是否频繁触发 `0.25 rad` 目标差保护，再决定是否调整任何参数。

不要为了“让它能跑”而绕过上述保护。模型训练损失低只表示拟合数据，不等于真机行为已经安全。
