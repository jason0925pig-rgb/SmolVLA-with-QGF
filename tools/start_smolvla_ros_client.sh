#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/tools/smolvla_orin_env.sh"
ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"
ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
WORKSPACE_SETUP="${TELEOP_PROJECT_ROOT}/install/setup.bash"
VENV="${SMOLVLA_CLIENT_VENV:-${HOME}/work/telop/venvs/lerobot-client}"
SERVER_ADDRESS="${SMOLVLA_SERVER_ADDRESS:?set SMOLVLA_SERVER_ADDRESS, for example 192.168.2.110:8080}"
SERVER_MODEL_PATH="${SMOLVLA_SERVER_MODEL_PATH:?set SMOLVLA_SERVER_MODEL_PATH to the checkpoint path on the inference server}"
TASK="${SMOLVLA_TASK:-把矿泉水放进纸箱里。}"
ACTIONS_PER_CHUNK="${SMOLVLA_ACTIONS_PER_CHUNK:-10}"
CHUNK_SIZE_THRESHOLD="${SMOLVLA_CHUNK_SIZE_THRESHOLD:-0.5}"
FPS="${SMOLVLA_FPS:-30}"
AGGREGATE_FN="${SMOLVLA_AGGREGATE_FN:-weighted_average}"
POLICY_DEVICE="${SMOLVLA_POLICY_DEVICE:-cuda}"
CLIENT_DEVICE="${SMOLVLA_CLIENT_DEVICE:-cpu}"
STATE_TIMEOUT_SECONDS="${SMOLVLA_STATE_TIMEOUT_SECONDS:-0.3}"
CAMERA_TIMEOUT_SECONDS="${SMOLVLA_CAMERA_TIMEOUT_SECONDS:-0.3}"

[[ -r "${ROS_SETUP}" ]] || { echo "ERROR: missing ${ROS_SETUP}" >&2; exit 2; }
[[ -r "${WORKSPACE_SETUP}" ]] || { echo "ERROR: build the ROS workspace first" >&2; exit 2; }
[[ -x "${VENV}/bin/python" ]] || { echo "ERROR: missing ${VENV}/bin/python" >&2; exit 2; }

# shellcheck disable=SC1090
set +u
source "${ROS_SETUP}"
# shellcheck disable=SC1090
source "${WORKSPACE_SETUP}"
set -u

"${VENV}/bin/python" - <<'PY'
import cv2
import grpc
import lerobot
import lerobot_robot_armstrong_ros2
import rclpy
print("SMOLVLA_ROS_CLIENT_ENV_OK")
PY

echo "Starting observation/action client; robot action publication remains disabled."
echo "Enable only after preflight with: ros2 service call /smolvla/set_enabled std_srvs/srv/SetBool '{data: true}'"
echo "Policy server=${SERVER_ADDRESS} actions_per_chunk=${ACTIONS_PER_CHUNK} fps=${FPS}"
exec "${VENV}/bin/python" -m lerobot_robot_armstrong_ros2.async_client \
  --robot.type=armstrong_ros2 \
  --robot.id=armstrong_right \
  --robot.state_timeout_seconds="${STATE_TIMEOUT_SECONDS}" \
  --robot.camera_timeout_seconds="${CAMERA_TIMEOUT_SECONDS}" \
  --task="${TASK}" \
  --server_address="${SERVER_ADDRESS}" \
  --policy_type=smolvla \
  --pretrained_name_or_path="${SERVER_MODEL_PATH}" \
  --policy_device="${POLICY_DEVICE}" \
  --client_device="${CLIENT_DEVICE}" \
  --actions_per_chunk="${ACTIONS_PER_CHUNK}" \
  --chunk_size_threshold="${CHUNK_SIZE_THRESHOLD}" \
  --fps="${FPS}" \
  --aggregate_fn_name="${AGGREGATE_FN}"
