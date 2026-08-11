# SmolVLA with QGF

This is the canonical repository for the SmolVLA baseline, Armstrong model
deployment, Guided Action Flow (QGF), critic training, and baseline/QGF
experiments. Hardware teleoperation and dataset capture live separately in
[`One-Arm-Teleoperation`](https://github.com/jason0925pig-rgb/One-Arm-Teleoperation).

## Repository layout

- `qgf/`: Guided Action Flow and critic source, training/evaluation scripts,
  tests, and algorithm notes.
- `lerobot_robot_armstrong_ros2/`: LeRobot adapter for the Armstrong right arm.
- `tools/`: SmolVLA training, validation, policy-server, Orin client, and
  staged real-robot launchers.
- `servo_controller/config/smolvla_first_rollout.yaml`: model-rollout safety
  configuration. The ROS2 controller implementation remains in the teleop repo.
- `docs/`: baseline training/deployment results and QGF documentation.
- `experiments/`: LIBERO experiment clients, concise reports, and visual audits.

## Armstrong dependency boundary

On the Orin, the two repositories are expected to be siblings:

```text
/home/nvidia/work/telop/
├── One-Arm-Teleoperation/   # ROS2 hardware, camera and gripper nodes
└── SmolVLA-with-QGF/        # model policy, adapter and rollout launcher
```

The model launcher sources the ROS workspace from:

```bash
export TELEOP_PROJECT_ROOT=/home/nvidia/work/telop/One-Arm-Teleoperation
```

The default already points there. Override it only when the hardware workspace
is installed elsewhere.

## Run the current Orin baseline

From Windows PowerShell:

```powershell
cd E:\AAA__Github_Project\SmolVLA-with-QGF
.\tools\start_smolvla_orin.cmd
```

The staged launcher does not move the robot until the operator explicitly
passes the hardware checks and enters `ARM`, followed later by `MOVE`.

## Collect real-robot QGF rollouts

The continuous collector runs the baseline repeatedly and asks after every
round whether to save a success, save a failure, or delete the attempt:

```powershell
.\tools\collect_qgf_rollouts.cmd -EpisodeCount 20
```

It writes paper-compatible transition tables plus chest/right-wrist MP4 files.
See [`qgf/docs/real_robot_qgf_collection.md`](qgf/docs/real_robot_qgf_collection.md)
for the schema, labeling workflow, and realistic data-volume guidance.

## Data and security

Do not commit datasets, checkpoints, SSH keys, tokens, virtual environments,
runtime outputs, or server access material. These paths and common model
weight formats are ignored. Keep large artifacts on the designated SSD/model
storage and record their checksums and paths in documentation.
