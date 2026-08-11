#!/usr/bin/env bash
set -Eeuo pipefail

# Prepare only the CPU-side ROS 2 observation/action client.  This script does
# not start cameras, connect to the robot controller, power, enable or move it.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/tools/smolvla_orin_env.sh"
VENV="${SMOLVLA_CLIENT_VENV:-${HOME}/work/telop/venvs/lerobot-client}"
LEROBOT_SOURCE="${SMOLVLA_LEROBOT_SOURCE:-${PROJECT_ROOT}/third_party/lerobot-v0.4.4}"
PLUGIN_SOURCE="${PROJECT_ROOT}/lerobot_robot_armstrong_ros2"
PYTHON="${SMOLVLA_CLIENT_PYTHON:-/usr/bin/python3}"

[[ -x "${PYTHON}" ]] || { echo "ERROR: missing ${PYTHON}" >&2; exit 2; }
[[ -r /opt/ros/humble/setup.bash ]] || {
  echo "ERROR: this client profile requires ROS 2 Humble." >&2
  exit 2
}
[[ -f "${LEROBOT_SOURCE}/pyproject.toml" ]] || {
  echo "ERROR: LeRobot v0.4.4 source is missing at ${LEROBOT_SOURCE}" >&2
  echo "Copy the deployment bundle's software/lerobot-v0.4.4 directory there first." >&2
  exit 2
}
[[ -f "${PLUGIN_SOURCE}/pyproject.toml" ]] || {
  echo "ERROR: Armstrong LeRobot plugin is missing: ${PLUGIN_SOURCE}" >&2
  exit 2
}

"${PYTHON}" -m venv --system-site-packages "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip setuptools wheel

# Keep Jetson's NVIDIA PyTorch and ROS packages from the system environment.
# The robot-side client performs no model inference.
"${VENV}/bin/python" -m pip install \
  "draccus>=0.10.0,<0.11.0" \
  "grpcio>=1.71.0,<2.0.0" \
  "protobuf>=5.29.3,<7.0.0" \
  "numpy>=1.24,<2.0"
"${VENV}/bin/python" -m pip install --no-deps -e "${LEROBOT_SOURCE}"
"${VENV}/bin/python" -m pip install --no-deps -e "${PLUGIN_SOURCE}"

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
if [[ -r "${TELEOP_PROJECT_ROOT}/install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "${TELEOP_PROJECT_ROOT}/install/setup.bash"
fi
"${VENV}/bin/python" - <<'PY'
import cv2
import grpc
import lerobot
import lerobot_robot_armstrong_ros2
import numpy
import rclpy
import torch
from lerobot.async_inference.robot_client import RobotClient
from lerobot_robot_armstrong_ros2.async_client import ArmstrongRobotClient

print("SMOLVLA_CLIENT_ENV_READY")
print(f"torch={torch.__version__}")
print(f"cuda_visible_to_client={torch.cuda.is_available()}")
print(f"opencv={cv2.__version__}")
print(f"numpy={numpy.__version__}")
print(f"async_client={ArmstrongRobotClient.__name__};upstream={RobotClient.__name__}")
PY

echo "SETUP COMPLETE: no robot or camera process was started."
