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
GRIPPER_OPEN_THRESHOLD="${SMOLVLA_GRIPPER_OPEN_THRESHOLD:-0.15}"
GRIPPER_CLOSE_THRESHOLD="${SMOLVLA_GRIPPER_CLOSE_THRESHOLD:-0.85}"
GRIPPER_CONFIRMATION_FRAMES="${SMOLVLA_GRIPPER_CONFIRMATION_FRAMES:-5}"
CANONICALIZE_POLICY_OBSERVATION="${SMOLVLA_CANONICALIZE_POLICY_OBSERVATION:-true}"
INITIAL_POSE_TOLERANCE_RAD="${SMOLVLA_INITIAL_POSE_TOLERANCE_RAD:-0.0}"

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

"${VENV}/bin/python" - "${GRIPPER_OPEN_THRESHOLD}" "${GRIPPER_CLOSE_THRESHOLD}" "${GRIPPER_CONFIRMATION_FRAMES}" "${INITIAL_POSE_TOLERANCE_RAD}" <<'PY'
import sys

open_threshold, close_threshold = map(float, sys.argv[1:3])
confirmation_frames = int(sys.argv[3])
initial_pose_tolerance_rad = float(sys.argv[4])
if not 0.0 <= open_threshold < close_threshold <= 1.0:
    raise SystemExit("ERROR: require 0 <= gripper open threshold < close threshold <= 1")
if confirmation_frames < 1:
    raise SystemExit("ERROR: gripper confirmation frames must be at least 1")
if initial_pose_tolerance_rad < 0.0:
    raise SystemExit("ERROR: initial pose tolerance cannot be negative")
PY

case "${CANONICALIZE_POLICY_OBSERVATION}" in
  true|false) ;;
  *)
    echo "ERROR: SMOLVLA_CANONICALIZE_POLICY_OBSERVATION must be true or false" >&2
    exit 2
    ;;
esac

echo "Starting observation/action client; robot action publication remains disabled."
echo "Enable only after preflight with: ros2 service call /smolvla/set_enabled std_srvs/srv/SetBool '{data: true}'"
echo "Policy server=${SERVER_ADDRESS} actions_per_chunk=${ACTIONS_PER_CHUNK} fps=${FPS}"
echo "Gripper filter: open<=${GRIPPER_OPEN_THRESHOLD}, close>=${GRIPPER_CLOSE_THRESHOLD}, confirmation_frames=${GRIPPER_CONFIRMATION_FRAMES}"
echo "Policy joint observation coordinates: $([[ "${CANONICALIZE_POLICY_OBSERVATION}" == "true" ]] && echo canonical || echo raw)"
echo "Initial pose tolerance: ${INITIAL_POSE_TOLERANCE_RAD} rad"
exec "${VENV}/bin/python" -m lerobot_robot_armstrong_ros2.async_client \
  --robot.type=armstrong_ros2 \
  --robot.id=armstrong_right \
  --robot.state_timeout_seconds="${STATE_TIMEOUT_SECONDS}" \
  --robot.camera_timeout_seconds="${CAMERA_TIMEOUT_SECONDS}" \
  --robot.canonicalize_policy_observation="${CANONICALIZE_POLICY_OBSERVATION}" \
  --robot.initial_envelope_overshoot_rad="${INITIAL_POSE_TOLERANCE_RAD}" \
  --robot.gripper_open_threshold="${GRIPPER_OPEN_THRESHOLD}" \
  --robot.gripper_close_threshold="${GRIPPER_CLOSE_THRESHOLD}" \
  --robot.gripper_confirmation_frames="${GRIPPER_CONFIRMATION_FRAMES}" \
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
