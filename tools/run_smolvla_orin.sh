#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="/tmp/one_arm_smolvla_${UID}"
SERVER_PID_FILE="${RUNTIME_DIR}/policy_server.pid"
CLIENT_PID_FILE="${RUNTIME_DIR}/policy_client.pid"
SERVER_LOG="${RUNTIME_DIR}/policy_server.log"
CLIENT_LOG="${RUNTIME_DIR}/policy_client.log"
EVENT_LOG="${RUNTIME_DIR}/launcher_events.log"
STOPPING=0
STOP_SOURCE="normal_exit"
CLIENT_LOG_STREAM_PID=""
SERVER_LOG_STREAM_PID=""
COMPLETION_MONITOR_PID=""
COMPLETION_MONITOR_LOG="${RUNTIME_DIR}/completion_monitor.log"

mkdir -p "${RUNTIME_DIR}"

emit_launcher_event() {
  printf 'TELEMETRY_EVENT %s wall_time=%s\n' "$1" "$(date --iso-8601=ns)" |
    tee -a "${EVENT_LOG}"
}
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/tools/smolvla_orin_env.sh"
if [[ -n "${SMOLVLA_TASK_B64:-}" ]]; then
  export SMOLVLA_TASK="$(printf '%s' "${SMOLVLA_TASK_B64}" | base64 --decode)"
fi
export SMOLVLA_TASK="${SMOLVLA_TASK:-把矿泉水放进纸箱里。}"

set +u
# shellcheck disable=SC1091
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
# shellcheck disable=SC1091
source "${TELEOP_PROJECT_ROOT}/install/setup.bash"
set -u

alive_pid_file() {
  local file="$1" pid
  [[ -s "${file}" ]] || return 1
  pid="$(<"${file}")"
  [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null
}

start_managed() {
  local pid_file="$1" log_file="$2"
  shift 2
  nohup setsid "$@" >"${log_file}" 2>&1 < /dev/null &
  printf '%s\n' "$!" >"${pid_file}"
}

stop_managed() {
  local pid_file="$1" pid deadline
  [[ -s "${pid_file}" ]] || return 0
  pid="$(<"${pid_file}")"
  if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    kill -INT -- "-${pid}" 2>/dev/null || kill -INT "${pid}" 2>/dev/null || true
    deadline=$((SECONDS + 8))
    while kill -0 "${pid}" 2>/dev/null && (( SECONDS < deadline )); do sleep 0.1; done
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    fi
  fi
  rm -f -- "${pid_file}"
}

cleanup() {
  local exit_code=$?
  (( STOPPING == 0 )) || return
  STOPPING=1
  trap - EXIT INT TERM HUP
  echo
  emit_launcher_event "event=launcher_cleanup source=${STOP_SOURCE}"
  echo "Stopping policy motion, servo mode, gripper, robot enable and power..."
  "${PROJECT_ROOT}/tools/ubuntu_smolvla_stack.sh" stop || true
  stop_managed "${CLIENT_PID_FILE}"
  stop_managed "${SERVER_PID_FILE}"
  [[ -z "${CLIENT_LOG_STREAM_PID}" ]] || kill "${CLIENT_LOG_STREAM_PID}" 2>/dev/null || true
  [[ -z "${SERVER_LOG_STREAM_PID}" ]] || kill "${SERVER_LOG_STREAM_PID}" 2>/dev/null || true
  [[ -z "${COMPLETION_MONITOR_PID}" ]] || kill "${COMPLETION_MONITOR_PID}" 2>/dev/null || true
  echo "SMOLVLA_ALL_STOPPED_AND_POWERED_OFF"
  if (( exit_code != 0 && exit_code != 130 )); then
    echo "--- policy client log (last 40 lines) ---" >&2
    tail -n 40 "${CLIENT_LOG}" >&2 2>/dev/null || true
    echo "--- arm log (last 40 lines) ---" >&2
    tail -n 40 "${RUNTIME_DIR}/arm.log" >&2 2>/dev/null || true
  fi
  exit "${exit_code}"
}

handle_signal() {
  local signal_name="$1" exit_code="$2"
  STOP_SOURCE="signal_${signal_name}"
  emit_launcher_event "event=operator_interrupt source=${STOP_SOURCE}"
  exit "${exit_code}"
}

trap cleanup EXIT
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM
trap 'handle_signal HUP 129' HUP

start_visible_inference_logs() {
  local client_pid server_pid
  client_pid="$(<"${CLIENT_PID_FILE}")"
  server_pid="$(<"${SERVER_PID_FILE}")"
  (
    tail --pid="${client_pid}" -n 8 -F "${CLIENT_LOG}" 2>/dev/null |
      sed -u 's/^/[POLICY CLIENT] /'
  ) &
  CLIENT_LOG_STREAM_PID=$!
  (
    tail --pid="${server_pid}" -n 20 -F "${SERVER_LOG}" 2>/dev/null |
      grep --line-buffered -E \
        'Running inference|Preprocessing and inference|Action chunk|Total time|ERROR|WARNING' |
      sed -u 's/^/[POLICY SERVER] /'
  ) &
  SERVER_LOG_STREAM_PID=$!
}

wait_for_server() {
  "${SMOLVLA_ORIN_VENV}/bin/python" - "${SMOLVLA_SERVER_HOST}" "${SMOLVLA_SERVER_PORT}" <<'PY'
import socket
import sys
import time

host, port = sys.argv[1], int(sys.argv[2])
deadline = time.monotonic() + 30.0
while time.monotonic() < deadline:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            raise SystemExit(0)
    except OSError:
        time.sleep(0.25)
raise SystemExit("policy server did not open its loopback port within 30 seconds")
PY
}

wait_for_fresh_joint_stream() {
  echo "Waiting for three valid seven-joint feedback samples..."
  python3 "${PROJECT_ROOT}/tools/wait_for_joint_state.py" \
    --topic /right_arm/joint_states \
    --timeout 20 \
    --minimum-messages 3
}

pre_arm_safety_check() {
  local output
  if output="$(python3 "${PROJECT_ROOT}/tools/wait_for_string_topic.py" \
      /right_arm/safety_status \
      --timeout 4 \
      --contains connected=1 \
      --contains feedback_valid=1 \
      --contains robot_powered_on=0 \
      --contains robot_enabled=0 \
      --contains robot_emergency_stop=0 \
      --contains robot_protective_stop=0 \
      --contains robot_socket_connected=1 \
      --contains robot_error_code=0 \
      --contains motion_enabled=0 2>&1)"; then
    printf '%s\n' "${output}"
    echo "PRE_ARM_SAFETY_CHECK_OK_NO_MOTION"
    return 0
  fi

  printf '%s\n' "${output}" >&2
  if grep -q 'robot_emergency_stop=1' <<<"${output}"; then
    echo "ERROR: the physical E-stop is pressed. Release it, verify the controller error clears, then restart this launcher." >&2
  elif grep -q 'robot_protective_stop=1' <<<"${output}"; then
    echo "ERROR: robot protective stop is active. Clear the physical cause and controller protective stop, then restart." >&2
  elif ! grep -q 'robot_error_code=0' <<<"${output}"; then
    echo "ERROR: the robot controller reports a nonzero error code. Clear it on the controller before ARM." >&2
  elif ! grep -q 'robot_socket_connected=1' <<<"${output}"; then
    echo "ERROR: the Armstrong controller socket is not connected." >&2
  elif grep -Eq 'robot_powered_on=1|robot_enabled=1|motion_enabled=1' <<<"${output}"; then
    echo "ERROR: the robot is not in the required power-off, disabled, motion-off starting state." >&2
  else
    echo "ERROR: robot pre-ARM safety status was not healthy and current within four seconds." >&2
  fi
  return 1
}

if alive_pid_file "${SERVER_PID_FILE}" || alive_pid_file "${CLIENT_PID_FILE}"; then
  echo "ERROR: the one-window SmolVLA launcher is already running." >&2
  exit 3
fi

# A run that stops before the policy client starts must never print errors
# retained in the shared runtime directory from an earlier rollout.
: >"${SERVER_LOG}"
: >"${CLIENT_LOG}"
: >"${EVENT_LOG}"

echo "============================================================"
echo "SmolVLA / Armstrong single-window launcher"
echo "Task: ${SMOLVLA_TASK}"
echo "Stage 1 starts inference, cameras and observation interfaces only."
echo "The robot will NOT power on, enable or move in this stage."
echo "============================================================"

start_managed "${SERVER_PID_FILE}" "${SERVER_LOG}" \
  "${PROJECT_ROOT}/tools/start_smolvla_orin_policy_server.sh"
sleep 1
alive_pid_file "${SERVER_PID_FILE}" || {
  echo "ERROR: policy server exited during startup." >&2
  tail -n 80 "${SERVER_LOG}" >&2 || true
  exit 1
}
wait_for_server
"${PROJECT_ROOT}/tools/ubuntu_smolvla_stack.sh" start
pre_arm_safety_check

echo
echo "OBSERVATION_STACK_READY_NO_MOTION"
echo "下一步会给右臂上电、使能并打开夹爪；机械臂可能轻微落位。"
echo "确认现场无人、无障碍物、实体急停触手可及后，输入 ARM 并回车。"
echo "输入其他内容或按 Ctrl+C 会安全退出并保持机器人下电。"
read -r -p "Type ARM to continue: " answer
[[ "${answer}" == "ARM" ]] || exit 0

"${PROJECT_ROOT}/tools/ubuntu_smolvla_stack.sh" prepare
wait_for_fresh_joint_stream
echo "JOINT_STREAM_RECOVERED_AFTER_POWER_ENABLE"

# Starting the observation client only after the blocking power/enable SDK
# calls prevents their expected multi-second pause from being reported as a
# stale joint-state failure.
start_managed "${CLIENT_PID_FILE}" "${CLIENT_LOG}" \
  "${PROJECT_ROOT}/tools/start_smolvla_orin_ros_client.sh"
sleep 2
alive_pid_file "${CLIENT_PID_FILE}" || {
  echo "ERROR: policy client exited during startup." >&2
  tail -n 100 "${CLIENT_LOG}" >&2 || true
  exit 1
}
start_visible_inference_logs

python3 "${PROJECT_ROOT}/tools/wait_for_string_topic.py" /smolvla/status \
  --timeout 180 \
  --contains connected=1 \
  --contains action_enabled=0 \
  --contains observation_ready=1 || {
    echo "ERROR: the policy client never reached a fresh observation state." >&2
    tail -n 100 "${CLIENT_LOG}" >&2 || true
    exit 1
  }
echo "Waiting for the first complete 50-action policy chunk; no policy action is published yet..."
python3 "${PROJECT_ROOT}/tools/wait_for_string_topic.py" /smolvla/status \
  --timeout 180 \
  --contains connected=1 \
  --contains action_enabled=0 \
  --contains observation_ready=1 \
  --contains policy_chunk_ready=1 \
  --contains action_queue_size=50 \
  --contains expected_action_chunk_size=50 || {
    echo "ERROR: the first complete 50-action policy chunk was not preloaded." >&2
    tail -n 100 "${CLIENT_LOG}" >&2 || true
    tail -n 100 "${SERVER_LOG}" >&2 || true
    exit 1
  }
echo "FIRST_POLICY_CHUNK_READY_50_NO_MOTION"
"${PROJECT_ROOT}/tools/ubuntu_smolvla_stack.sh" servo

echo
echo "SMOLVLA_READY_NO_POLICY_MOTION"
echo "模型、两路相机、七轴反馈和夹爪已就绪，但模型动作门仍关闭。"
echo "下一步输入 MOVE 后机器人将按模型输出运动。"
echo "确认初始场景正确且实体急停触手可及，再输入 MOVE 并回车。"
read -r -p "Type MOVE to start model control: " answer
[[ "${answer}" == "MOVE" ]] || exit 0

"${PROJECT_ROOT}/tools/ubuntu_smolvla_stack.sh" enable-policy
echo
echo "SMOLVLA_RUNNING"
echo "机器人正在由模型控制。按 Ctrl+C，或输入 STOP 并回车，即刻执行有序停止、去使能和下电。"
# The adapter closes its action gate and publishes STOP before exposing
# task_completed=1. This monitor turns that safe state into a normal launcher
# exit; Ctrl+C and typed STOP remain available throughout the rollout.
: >"${COMPLETION_MONITOR_LOG}"
python3 "${PROJECT_ROOT}/tools/wait_for_string_topic.py" /smolvla/status \
  --timeout 86400 \
  --contains task_completed=1 >"${COMPLETION_MONITOR_LOG}" 2>&1 &
COMPLETION_MONITOR_PID=$!

while true; do
  if read -r -t 0.5 answer; then
    if [[ "${answer}" == "STOP" ]]; then
      STOP_SOURCE="operator_typed_stop"
      emit_launcher_event "event=operator_stop source=typed_STOP"
      break
    fi
    echo "Type STOP then Enter, or press Ctrl+C, to stop."
  fi
  if [[ -n "${COMPLETION_MONITOR_PID}" ]] && ! kill -0 "${COMPLETION_MONITOR_PID}" 2>/dev/null; then
    if wait "${COMPLETION_MONITOR_PID}"; then
      cat "${COMPLETION_MONITOR_LOG}"
      COMPLETION_MONITOR_PID=""
      STOP_SOURCE="task_completed_returned_home"
      emit_launcher_event "event=task_completed source=returned_home"
      echo "TASK_COMPLETED_RETURNED_HOME"
      echo "The grasp/release cycle completed and all seven joints returned near the captured start pose."
      break
    fi
    echo "WARNING: automatic-completion monitor stopped unexpectedly; manual STOP remains available." >&2
    cat "${COMPLETION_MONITOR_LOG}" >&2 || true
    COMPLETION_MONITOR_PID=""
  fi
done
