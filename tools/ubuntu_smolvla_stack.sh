#!/usr/bin/env bash
set -Eeuo pipefail

ACTION="${1:-status}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/tools/smolvla_orin_env.sh"
ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"
RUNTIME_DIR="/tmp/one_arm_smolvla_${UID}"
CONFIG="${PROJECT_ROOT}/servo_controller/config/smolvla_first_rollout.yaml"
GRIPPER_DEVICE="${ONE_ARM_GRIPPER_DEVICE:-/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0}"
CHEST_SERIAL="${ONE_ARM_CHEST_SERIAL:-CP8284100034}"
WRIST_SERIAL="${ONE_ARM_WRIST_SERIAL:-CPCD75300083}"

mkdir -p "${RUNTIME_DIR}"
[[ -r "/opt/ros/${ROS_DISTRO_NAME}/setup.bash" ]] || {
  echo "ERROR: ROS setup is missing for ${ROS_DISTRO_NAME}." >&2
  exit 2
}
[[ -r "${TELEOP_PROJECT_ROOT}/install/setup.bash" ]] || {
  echo "ERROR: teleoperation ROS workspace is not built: ${TELEOP_PROJECT_ROOT}/install/setup.bash" >&2
  exit 2
}
# shellcheck disable=SC1090
set +u
source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
# shellcheck disable=SC1090
source "${TELEOP_PROJECT_ROOT}/install/setup.bash"
set -u

pid_file() { printf '%s/%s.pid\n' "${RUNTIME_DIR}" "$1"; }
log_file() { printf '%s/%s.log\n' "${RUNTIME_DIR}" "$1"; }

alive() {
  local file pid
  file="$(pid_file "$1")"
  [[ -s "${file}" ]] || return 1
  pid="$(<"${file}")"
  [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null
}

start_component() {
  local name="$1"
  shift
  nohup setsid "$@" >"$(log_file "${name}")" 2>&1 < /dev/null &
  printf '%s\n' "$!" >"$(pid_file "${name}")"
  echo "started ${name}: pid=$! log=$(log_file "${name}")"
}

require_alive() {
  local name="$1"
  if ! alive "${name}"; then
    echo "ERROR: ${name} exited during startup; recent log follows:" >&2
    tail -n 80 "$(log_file "${name}")" >&2 || true
    return 1
  fi
}

stop_component() {
  local name="$1" file pid deadline
  file="$(pid_file "${name}")"
  [[ -s "${file}" ]] || return 0
  pid="$(<"${file}")"
  if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    kill -INT -- "-${pid}" 2>/dev/null || kill -INT "${pid}" 2>/dev/null || true
    deadline=$((SECONDS + 8))
    while kill -0 "${pid}" 2>/dev/null && (( SECONDS < deadline )); do sleep 0.1; done
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    fi
  fi
  rm -f -- "${file}"
}

wait_service() {
  local service="$1" deadline=$((SECONDS + ${2:-15}))
  while (( SECONDS < deadline )); do
    ros2 service list 2>/dev/null | grep -Fxq "${service}" && return 0
    sleep 0.25
  done
  echo "ERROR: service did not appear: ${service}" >&2
  return 1
}

call_bool() {
  local service="$1" value="$2" output
  output="$(timeout 15 ros2 service call "${service}" std_srvs/srv/SetBool "{data: ${value}}" 2>&1)" || {
    printf '%s\n' "${output}" >&2
    return 1
  }
  printf '%s\n' "${output}"
  grep -Eq 'success[=:][[:space:]]*(true|True)' <<<"${output}"
}

service_available() {
  local service="$1" services
  services="$(timeout 5 ros2 service list --no-daemon --spin-time 1.5 2>/dev/null)" || return 1
  grep -Fxq "${service}" <<<"${services}"
}

try_bool() {
  local service="$1" value="$2"
  if ! service_available "${service}"; then
    echo "service not running; skipping: ${service}"
    return 0
  fi
  call_bool "${service}" "${value}" || true
}

wait_status() {
  local timeout_seconds="$1"
  shift
  local args=("${PROJECT_ROOT}/tools/wait_for_string_topic.py" /right_arm/safety_status --timeout "${timeout_seconds}") item
  for item in "$@"; do args+=(--contains "${item}"); done
  python3 "${args[@]}"
}

wait_gripper_status() {
  local timeout_seconds="$1"
  shift
  local args=("${PROJECT_ROOT}/tools/wait_for_string_topic.py" /right_arm/gripper_status --timeout "${timeout_seconds}") item
  for item in "$@"; do args+=(--contains "${item}"); done
  python3 "${args[@]}"
}

publisher_count() {
  ros2 topic info "$1" 2>/dev/null | sed -n 's/^Publisher count: //p'
}

wait_for_exactly_one_publisher() {
  local topic="$1"
  local timeout_seconds="$2"
  local deadline=$((SECONDS + timeout_seconds))
  local count=""

  # DDS discovery is asynchronous.  The policy client may be fully alive
  # while its publisher has not appeared in `ros2 topic info` yet, so do not
  # reject the stack on a single early query.
  while (( SECONDS < deadline )); do
    count="$(publisher_count "${topic}")"
    if [[ "${count}" == "1" ]]; then
      echo "SMOLVLA_JOINT_PUBLISHER_READY topic=${topic} count=1"
      return 0
    fi
    sleep 0.25
  done

  count="$(publisher_count "${topic}")"
  echo "ERROR: expected exactly one SmolVLA joint publisher on ${topic} after ${timeout_seconds}s; observed count=${count:-unavailable}." >&2
  return 1
}

preflight() {
  [[ -r "${CONFIG}" ]] || { echo "ERROR: missing ${CONFIG}" >&2; return 1; }
  ping -c 1 -W 1 192.168.2.226 >/dev/null || {
    echo "ERROR: Armstrong controller 192.168.2.226 is unreachable." >&2
    return 1
  }
  [[ -e "${GRIPPER_DEVICE}" ]] || {
    echo "ERROR: CTAG2F120 path is missing: ${GRIPPER_DEVICE}" >&2
    return 1
  }
  local conflicts owners
  conflicts="$(pgrep -af 'robot_timer|test_joint_trajectory_sub|gripper_controller|safe_one_arm_servo|safe_gripper_controller|udp_leader_bridge|(^|[[:space:]/])pi0([[:space:]/]|$)' 2>/dev/null || true)"
  [[ -z "${conflicts}" ]] || {
    echo "ERROR: an existing robot/policy process is running:" >&2
    printf '%s\n' "${conflicts}" >&2
    return 1
  }
  owners="$(fuser "$(readlink -f "${GRIPPER_DEVICE}")" 2>/dev/null || true)"
  [[ -z "${owners//[[:space:]]/}" ]] || {
    echo "ERROR: gripper serial device is busy: ${owners}" >&2
    return 1
  }
}

start_stack() {
  if alive arm || alive gripper || alive cameras; then
    echo "ERROR: SmolVLA stack is already running; use status or stop." >&2
    exit 3
  fi
  preflight
  start_component cameras ros2 launch one_arm_teleop_bridge dataset_cameras.launch.py \
    primary_camera_name:=camera_chest \
    head_serial:="${CHEST_SERIAL}" \
    wrist_serial:="${WRIST_SERIAL}"
  start_component arm ros2 run servo_controller safe_one_arm_servo --ros-args \
    --params-file "${CONFIG}" \
    -p dry_run:=false \
    -p hardware_power_authorized:=true \
    -p hardware_enable_authorized:=true \
    -p hardware_motion_authorized:=true \
    -p limits_configured:=true
  start_component gripper ros2 run servo_controller safe_gripper_controller --ros-args \
    --params-file "${CONFIG}" \
    -p dry_run:=false \
    -p configuration_complete:=true
  if ! {
    sleep 2
    require_alive cameras
    require_alive arm
    require_alive gripper
    wait_service /right_arm/set_powered_on 20
    wait_service /right_arm/set_robot_enabled 20
    wait_service /right_arm/set_motion_enabled 20
    wait_service /right_arm/set_gripper_enabled 20
    python3 "${PROJECT_ROOT}/tools/check_camera_fps.py" \
      --duration 4 \
      --warmup 1 \
      --minimum-fps 27 \
      --maximum-fps 33 \
      --maximum-gap-ms 100 \
      --topic /camera_chest/color/image_raw/compressed \
      --topic /camera_wrist/color/image_raw/compressed
  }; then
    echo "ERROR: SmolVLA observation stack preflight failed; stopping all started components." >&2
    stop_component gripper
    stop_component arm
    stop_component cameras
    return 1
  fi
  echo "SMOLVLA_STACK_STARTED_OBSERVATION_ONLY"
  echo "No power, robot enable, servo mode or policy action has been requested."
}

prepare_stack() {
  echo "WARNING: this command powers/enables the right arm and opens the gripper."
  echo "It does not enter servo mode and no policy action can reach the arm."
  if wait_status 3 robot_powered_on=1 robot_enabled=1 motion_enabled=0 servo_mode_entered=0 robot_error_code=0 robot_emergency_stop=0 robot_protective_stop=0 >/dev/null 2>&1; then
    echo "Robot is already safely powered and enabled; reusing that state after ARM authorization."
  else
    if wait_status 3 robot_powered_on=1 robot_enabled=0 robot_error_code=0 robot_emergency_stop=0 robot_protective_stop=0 >/dev/null 2>&1; then
      echo "Robot is already powered but disabled; skipping the redundant power-on request."
    else
      call_bool /right_arm/set_powered_on true
      wait_status 20 robot_powered_on=1 robot_error_code=0 robot_emergency_stop=0 robot_protective_stop=0
    fi
    call_bool /right_arm/set_robot_enabled true
    wait_status 20 robot_enabled=1 robot_error_code=0 robot_emergency_stop=0 robot_protective_stop=0
  fi
  call_bool /right_arm/set_gripper_enabled true
  call_bool /right_arm/set_gripper_open true
  wait_gripper_status 10 enabled=1 requested_open=1
  echo "SMOLVLA_ROBOT_PREPARED_NO_SERVO"
}

enter_servo() {
  wait_service /smolvla/set_enabled 20
  wait_for_exactly_one_publisher /right_arm/teleop_joint_command 20
  wait_status 10 robot_powered_on=1 robot_enabled=1 robot_error_code=0 robot_emergency_stop=0 robot_protective_stop=0
  call_bool /right_arm/set_motion_enabled true
  wait_status 15 motion_enabled=1 robot_error_code=0 robot_emergency_stop=0 robot_protective_stop=0
  echo "SMOLVLA_ARMED_NO_POLICY_ACTION"
}

arm_stack() {
  prepare_stack
  enter_servo
}

enable_policy() {
  echo "WARNING: the robot WILL MOVE under SmolVLA after this service succeeds."
  call_bool /smolvla/set_enabled true
  echo "SMOLVLA_POLICY_ENABLED"
}

disable_policy() {
  try_bool /smolvla/set_enabled false
  if alive arm; then
    call_bool /right_arm/set_motion_enabled false || true
  else
    try_bool /right_arm/set_motion_enabled false
  fi
  echo "SMOLVLA_POLICY_DISABLED"
}

reset_round() {
  # Keep controller login, power and robot enable alive between dataset
  # episodes.  Only the policy/servo gates are closed, then the gripper is
  # opened for the next scene reset.
  try_bool /smolvla/set_enabled false
  call_bool /right_arm/set_motion_enabled false || true
  wait_status 10 robot_powered_on=1 robot_enabled=1 robot_error_code=0 \
    robot_emergency_stop=0 robot_protective_stop=0 motion_enabled=0
  call_bool /right_arm/set_gripper_enabled true
  call_bool /right_arm/set_gripper_open true
  wait_gripper_status 10 enabled=1 requested_open=1
  echo "SMOLVLA_ROUND_RESET_POWER_AND_ENABLE_RETAINED"
}

stop_stack() {
  try_bool /smolvla/set_enabled false
  if alive arm; then
    call_bool /right_arm/set_motion_enabled false || true
  else
    try_bool /right_arm/set_motion_enabled false
  fi
  if alive gripper; then
    call_bool /right_arm/set_gripper_enabled false || true
  else
    try_bool /right_arm/set_gripper_enabled false
  fi
  if alive arm; then
    call_bool /right_arm/set_robot_enabled false || true
    call_bool /right_arm/set_powered_on false || true
  else
    try_bool /right_arm/set_robot_enabled false
    try_bool /right_arm/set_powered_on false
  fi
  stop_component gripper
  stop_component arm
  stop_component cameras
  echo "SMOLVLA_STACK_STOPPED"
}

status_stack() {
  for name in cameras arm gripper; do
    if alive "${name}"; then echo "${name}=running pid=$(<"$(pid_file "${name}")")"; else echo "${name}=stopped"; fi
  done
  ros2 node list 2>/dev/null || true
  python3 "${PROJECT_ROOT}/tools/wait_for_string_topic.py" /right_arm/safety_status --timeout 2 || true
  if ros2 node list 2>/dev/null | grep -Fxq /armstrong_lerobot_client; then
    python3 "${PROJECT_ROOT}/tools/wait_for_string_topic.py" /smolvla/status --timeout 2 || true
  else
    echo "smolvla_client=stopped"
  fi
}

case "${ACTION}" in
  start) start_stack ;;
  prepare) prepare_stack ;;
  servo) enter_servo ;;
  arm) arm_stack ;;
  enable-policy) enable_policy ;;
  disable-policy) disable_policy ;;
  stop) stop_stack ;;
  status) status_stack ;;
  reset-round) reset_round ;;
  *) echo "Usage: $0 {start|prepare|servo|arm|enable-policy|disable-policy|reset-round|stop|status}" >&2; exit 2 ;;
esac
