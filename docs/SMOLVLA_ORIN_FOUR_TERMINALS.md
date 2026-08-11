# SmolVLA 在公司 Jetson AGX Orin 上的四终端运行方法

> 本文保留用于逐进程调试。现场正常运行请改用单窗口入口
> `tools/start_smolvla_orin.cmd`；它会按正确顺序启动、等待状态恢复，并在
> `Ctrl+C` 后自动退出伺服、去使能和下电。不要同时使用两套启动方式。

这份流程固定用于 `nvidia@192.168.2.170`（Jetson AGX Orin、Ubuntu 22.04、ROS 2 Humble）。模型服务、ROS 客户端和机器人接口全部在 Orin 本机运行；策略通信地址是 `127.0.0.1:8080`，不经过 Windows 或 Wi-Fi。

## 运行前检查

1. 现场人员在机器人旁，实体急停触手可及。
2. 确认右臂、夹爪、胸部相机和右腕相机周围无人员及障碍物。
3. Windows PowerShell 登录 Orin：`ssh armstrong-orin`。
4. 每个终端都先执行：

```bash
cd /home/nvidia/work/telop/SmolVLA-with-QGF
```

第一次或更新环境后，可执行只读检查：

```bash
./tools/check_smolvla_orin_runtime.sh
```

该检查不启动 ROS、相机，不连接机器人控制器，也不会上电、使能或运动。

当前 Orin 参数为 30 Hz、每次 50 个动作、收到动作块后立即预取下一块。离线预热推理约 0.86–0.91 秒，接入真实 ROS 图像后约 1.2–1.5 秒；50 个动作在 30 Hz 下覆盖约 1.67 秒。模型客户端允许最长 1.0 秒的状态/相机短时抖动，底层机械臂安全节点的命令、反馈和控制周期保护保持不变。

## 终端 1：Orin 本机模型服务器（机器人不会动）

```bash
cd /home/nvidia/work/telop/SmolVLA-with-QGF
./tools/start_smolvla_orin_policy_server.sh
```

保持该终端打开。模型只在 Orin 的 GPU 0 上加载，服务只监听 `127.0.0.1:8080`。

## 终端 2：相机与安全控制节点（机器人不会动）

```bash
cd /home/nvidia/work/telop/SmolVLA-with-QGF
./tools/ubuntu_smolvla_stack.sh start
./tools/ubuntu_smolvla_stack.sh status
```

`start` 只启动两路相机、右臂安全节点和夹爪安全节点。它不会请求机器人上电、使能、拖动或伺服运动。

## 终端 3：ROS 观察与模型客户端（机器人仍不会动）

```bash
cd /home/nvidia/work/telop/SmolVLA-with-QGF
export SMOLVLA_TASK='把矿泉水放进纸箱里。'
./tools/start_smolvla_orin_ros_client.sh
```

客户端会采集七轴、夹爪及两路 RGB 图像并向本机策略服务器请求动作，但 `/smolvla/set_enabled` 默认是关闭的，因此动作不会发布给机器人。

## 终端 4：现场分阶段授权

先检查状态：

```bash
cd /home/nvidia/work/telop/SmolVLA-with-QGF
./tools/ubuntu_smolvla_stack.sh status
```

下面命令会给右臂上电、使能、打开夹爪并进入伺服模式，但策略动作门仍然关闭：

```bash
./tools/ubuntu_smolvla_stack.sh arm
```

确认机器人保持静止、初始位姿正确、相机方向正确且实体急停触手可及后，下面命令才会让机器人按照模型输出运动：

```bash
./tools/ubuntu_smolvla_stack.sh enable-policy
```

## 停止

任何软件异常先按实体急停。正常停止在终端 4 执行：

```bash
cd /home/nvidia/work/telop/SmolVLA-with-QGF
./tools/ubuntu_smolvla_stack.sh disable-policy
./tools/ubuntu_smolvla_stack.sh stop
```

然后在终端 3、终端 1 分别按 `Ctrl+C`。`stop` 的顺序是关闭策略门、退出伺服、关闭夹爪执行、机器人去使能和下电，再停止相机与控制节点。
