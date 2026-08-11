# SmolVLA 在 Armstrong Orin 上的单窗口运行方法

正常部署不再需要四个终端。四终端版本只保留作逐进程调试；现场运行使用一个 Windows PowerShell 窗口，由脚本统一管理模型服务、相机、机器人接口和模型客户端。

## 启动前

1. 现场人员必须在机器人旁，实体急停触手可及。
2. 清空右臂、夹爪及相机周围的人员和障碍物。
3. 机器人开机，松开急停，并确认控制器网线正常。
4. Windows 已配置 SSH 别名 `armstrong-orin`，且能运行 `ssh armstrong-orin`。
5. 不要同时运行旧的四终端流程、主从遥操作或厂商演示程序。

## 一条命令启动

在 Windows PowerShell 执行：

```powershell
cd E:\AAA__Github_Project\SmolVLA-with-QGF
.\tools\start_smolvla_orin.cmd
```

如需指定语言任务：

```powershell
.\tools\start_smolvla_orin.cmd -Task "把矿泉水放进纸箱里。"
```

脚本只在当前窗口交互，并按以下顺序推进：

1. 启动 Orin 本地模型服务、两路相机和只读机器人接口；机器人不动。
2. 显示 `OBSERVATION_STACK_READY_NO_MOTION` 后等待操作者输入 `ARM`。
3. 输入 `ARM` 后，右臂上电、使能，夹爪打开；随后脚本等待关节数据恢复。
4. 数据恢复后才启动模型观测客户端，避免上电/使能期间的同步 SDK 调用被误报为关节状态超时。
5. 显示 `SMOLVLA_READY_NO_POLICY_MOTION` 后等待操作者输入 `MOVE`。
6. 输入 `MOVE` 后模型才开始控制机器人。
7. 运行时按 `Ctrl+C`，或输入 `STOP` 并回车，脚本会关闭模型动作门、退出伺服、关闭夹爪执行、去使能并下电。

不要在看到相应提示前提前输入 `ARM` 或 `MOVE`。任何软件异常先按实体急停。

## 日志位置

所有后台进程仍有独立日志，不会因合并终端而丢失：

```text
/tmp/one_arm_smolvla_1000/policy_server.log
/tmp/one_arm_smolvla_1000/policy_client.log
/tmp/one_arm_smolvla_1000/arm.log
/tmp/one_arm_smolvla_1000/gripper.log
/tmp/one_arm_smolvla_1000/cameras.log
```

MobaXterm 和 PowerShell 都只是 SSH 客户端；无论从哪边启动，进程与日志都在同一台 Orin 上，因此可以从另一条 SSH 连接检查。
