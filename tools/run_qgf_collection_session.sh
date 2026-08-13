#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/tools/smolvla_orin_env.sh"

DATASET_ROOT="${QGF_DATASET_ROOT:-/home/nvidia/work/telop/qgf_real_rollouts}"
TARGET_EPISODES="${QGF_EPISODE_COUNT:-20}"
# Each attended round ends after this duration; power and enable remain on for S/F.
ROLLOUT_TIMEOUT_SECONDS="${QGF_ROLLOUT_TIMEOUT_SECONDS:-180}"
TASK_B64="${SMOLVLA_TASK_B64:?SMOLVLA_TASK_B64 is required}"
NOTES_B64="${QGF_NOTES_B64:-}"
TASK="$(printf '%s' "${TASK_B64}" | base64 --decode)"
NOTES="$(printf '%s' "${NOTES_B64}" | base64 --decode)"
RUNTIME_DIR="/tmp/one_arm_smolvla_${UID}"
SERVER_PID_FILE="${RUNTIME_DIR}/policy_server.pid"
CLIENT_PID_FILE="${RUNTIME_DIR}/policy_client.pid"
SERVER_LOG="${RUNTIME_DIR}/policy_server.log"
CLIENT_LOG="${RUNTIME_DIR}/policy_client.log"
RECORDER_PID=""
MONITOR_PID=""
CURRENT_STAGING=""
STOPPING=0
SAVED=0
ATTEMPT=0

mkdir -p "${RUNTIME_DIR}" "${DATASET_ROOT}/.staging"

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
  printf '%s
' "$!" >"${pid_file}"
}

stop_managed() {
  local file="$1" pid deadline
  [[ -s "${file}" ]] || return 0
  pid="$(<"${file}")"
  if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    kill -INT -- "-${pid}" 2>/dev/null || kill -INT "${pid}" 2>/dev/null || true
    deadline=$((SECONDS + 8))
    while kill -0 "${pid}" 2>/dev/null && (( SECONDS < deadline )); do sleep 0.1; done
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  fi
  rm -f -- "${file}"
}

stop_recorder() {
  local deadline
  [[ -n "${RECORDER_PID}" ]] || return 0
  if kill -0 "${RECORDER_PID}" 2>/dev/null; then
    kill -INT "${RECORDER_PID}" 2>/dev/null || true
    deadline=$((SECONDS + 25))
    while kill -0 "${RECORDER_PID}" 2>/dev/null && (( SECONDS < deadline )); do sleep 0.1; done
    kill -TERM "${RECORDER_PID}" 2>/dev/null || true
  fi
  wait "${RECORDER_PID}" 2>/dev/null || true
  RECORDER_PID=""
}

delete_current_episode() {
  local staging_real staging_root_real
  [[ -n "${CURRENT_STAGING}" ]] || return 0
  staging_real="$(realpath -m -- "${CURRENT_STAGING}")"
  staging_root_real="$(realpath -m -- "${DATASET_ROOT}/.staging")"
  [[ "${staging_real}" == "${staging_root_real}/"* ]] || {
    echo "ERROR: refusing to delete rollout path outside ${staging_root_real}: ${staging_real}" >&2
    return 1
  }
  if ! "${SMOLVLA_ORIN_VENV}/bin/python" "${PROJECT_ROOT}/tools/finalize_qgf_episode.py" \
    --staging "${CURRENT_STAGING}" \
    --dataset-root "${DATASET_ROOT}" \
    --outcome discard \
    --task "${TASK}"; then
    echo "WARNING: finalizer discard failed; applying validated staging-only deletion." >&2
    rm -rf -- "${staging_real}"
  fi
  [[ ! -e "${staging_real}" ]] || {
    echo "ERROR: interrupted episode could not be deleted: ${staging_real}" >&2
    return 1
  }
  echo "QGF_INTERRUPTED_EPISODE_DELETED=${staging_real}"
  CURRENT_STAGING=""
}

cleanup() {
  local code=$? delete_error=0
  (( STOPPING == 0 )) || return
  STOPPING=1
  trap - EXIT INT TERM HUP
  [[ -z "${MONITOR_PID}" ]] || kill "${MONITOR_PID}" 2>/dev/null || true
  stop_recorder
  delete_current_episode || delete_error=1
  "${PROJECT_ROOT}/tools/ubuntu_smolvla_stack.sh" stop || true
  stop_managed "${CLIENT_PID_FILE}"
  stop_managed "${SERVER_PID_FILE}"
  echo "QGF_SESSION_STOPPED_POLICY_SERVO_ENABLE_AND_POWER_OFF"
  (( delete_error == 0 )) || code=40
  exit "${code}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

wait_for_server() {
  "${SMOLVLA_ORIN_VENV}/bin/python" - "${SMOLVLA_SERVER_HOST}" "${SMOLVLA_SERVER_PORT}" <<'PY'
import socket, sys, time
host, port = sys.argv[1], int(sys.argv[2])
deadline = time.monotonic() + 40
while time.monotonic() < deadline:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            raise SystemExit(0)
    except OSError:
        time.sleep(0.25)
raise SystemExit("policy server did not become ready")
PY
}

wait_observation_and_chunk() {
  python3 "${PROJECT_ROOT}/tools/wait_for_string_topic.py" /smolvla/status \
    --timeout 180 --contains connected=1 --contains action_enabled=0 \
    --contains observation_ready=1
  python3 "${PROJECT_ROOT}/tools/wait_for_string_topic.py" /smolvla/status \
    --timeout 180 --contains connected=1 --contains action_enabled=0 \
    --contains policy_chunk_ready=1 \
    --contains "expected_action_chunk_size=${SMOLVLA_ACTIONS_PER_CHUNK}"
}

call_trigger() {
  local service="$1" output
  output="$(timeout 20 ros2 service call "${service}" std_srvs/srv/Trigger '{}' 2>&1)" || {
    printf '%s
' "${output}" >&2
    return 1
  }
  printf '%s
' "${output}"
  grep -Eq 'success[=:][[:space:]]*(true|True)' <<<"${output}"
}

start_recorder() {
  local log="${CURRENT_STAGING}/recorder.log"
  mkdir -p "${CURRENT_STAGING}"
  "${SMOLVLA_ORIN_VENV}/bin/python" "${PROJECT_ROOT}/tools/qgf_episode_recorder.py" \
    --output-dir "${CURRENT_STAGING}" --task "${TASK}" \
    --sample-hz "${SMOLVLA_FPS}" --camera-fps 30 >"${log}" 2>&1 &
  RECORDER_PID=$!
  sleep 1
  kill -0 "${RECORDER_PID}" 2>/dev/null || {
    tail -n 80 "${log}" >&2 || true
    return 1
  }
}

finalize_current() {
  local outcome="$1" termination_source="$2"
  "${SMOLVLA_ORIN_VENV}/bin/python" "${PROJECT_ROOT}/tools/finalize_qgf_episode.py" \
    --staging "${CURRENT_STAGING}" --dataset-root "${DATASET_ROOT}" \
    --outcome "${outcome}" --task "${TASK}" --notes "${NOTES}" \
    --termination-source "${termination_source}"
  CURRENT_STAGING=""
}

if alive_pid_file "${SERVER_PID_FILE}" || alive_pid_file "${CLIENT_PID_FILE}"; then
  echo "ERROR: another SmolVLA launcher is already running." >&2
  exit 3
fi
[[ "${TARGET_EPISODES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: QGF_EPISODE_COUNT must be a positive integer." >&2; exit 2;
}

echo "============================================================"
echo "Continuous real-robot QGF collection"
echo "Task: ${TASK}"
echo "Target kept episodes: ${TARGET_EPISODES}"
echo "ARM and MOVE are entered only once. Power/model/cameras stay up between rounds."
echo "Ctrl+C, physical E-stop, protective stop, controller error or unexpected gate loss"
echo "deletes the entire current episode, including raw samples and videos."
echo "============================================================"

"${SMOLVLA_ORIN_VENV}/bin/python" "${PROJECT_ROOT}/tools/snapshot_qgf_provenance.py" \
  --project-root "${PROJECT_ROOT}" \
  --checkpoint "${SMOLVLA_SERVER_MODEL_PATH}" \
  --dataset-root "${DATASET_ROOT}"

: >"${SERVER_LOG}"
: >"${CLIENT_LOG}"
export SMOLVLA_POLICY_SERVER_MODULE="lerobot_robot_armstrong_ros2.policy_server_telemetry"
start_managed "${SERVER_PID_FILE}" "${SERVER_LOG}" \
  "${PROJECT_ROOT}/tools/start_smolvla_orin_policy_server.sh"
wait_for_server
"${PROJECT_ROOT}/tools/ubuntu_smolvla_stack.sh" start

read -r -p "Type ARM once to power and enable the right arm: " answer
[[ "${answer}" == "ARM" ]] || exit 0
"${PROJECT_ROOT}/tools/ubuntu_smolvla_stack.sh" prepare

start_managed "${CLIENT_PID_FILE}" "${CLIENT_LOG}" \
  "${PROJECT_ROOT}/tools/start_smolvla_orin_ros_client.sh"
sleep 2
alive_pid_file "${CLIENT_PID_FILE}" || {
  tail -n 100 "${CLIENT_LOG}" >&2 || true; exit 1;
}
wait_observation_and_chunk
"${PROJECT_ROOT}/tools/ubuntu_smolvla_stack.sh" servo

while (( SAVED < TARGET_EPISODES )); do
  ATTEMPT=$((ATTEMPT + 1))
  CURRENT_STAGING="${DATASET_ROOT}/.staging/$(date +%Y%m%d_%H%M%S)_attempt_$(printf '%04d' "${ATTEMPT}")"
  start_recorder

  if (( ATTEMPT == 1 )); then
    read -r -p "Type MOVE once to start episode 1: " answer
    [[ "${answer}" == "MOVE" ]] || exit 0
  else
    echo "NEXT_EPISODE_READY episode=$((SAVED + 1)); starting after the confirmed scene reset."
  fi

  "${PROJECT_ROOT}/tools/ubuntu_smolvla_stack.sh" enable-policy
  monitor_log="${CURRENT_STAGING}/monitor.log"
  python3 "${PROJECT_ROOT}/tools/monitor_qgf_rollout.py" \
    --startup-timeout 20 --timeout "${ROLLOUT_TIMEOUT_SECONDS}" >"${monitor_log}" 2>&1 &
  MONITOR_PID=$!
  termination_source=""
  monitor_code=0

  while kill -0 "${MONITOR_PID}" 2>/dev/null; do
    if ! alive_pid_file "${CLIENT_PID_FILE}"; then
      termination_source="policy_client_exited"
      monitor_code=30
      kill "${MONITOR_PID}" 2>/dev/null || true
      break
    fi
    if read -r -t 0.5 answer; then
      case "${answer}" in
        END)
          termination_source="operator_END"
          kill "${MONITOR_PID}" 2>/dev/null || true
          "${PROJECT_ROOT}/tools/ubuntu_smolvla_stack.sh" disable-policy
          break
          ;;
        ABORT)
          termination_source="operator_ABORT"
          exit 130
          ;;
        *) echo "Type END to finish and label this round, or ABORT/Ctrl+C to delete it." ;;
      esac
    fi
  done

  if [[ -z "${termination_source}" ]]; then
    set +e
    wait "${MONITOR_PID}"
    monitor_code=$?
    set -e
    termination_source="$(sed -n 's/^QGF_ROLLOUT_MONITOR source=//p' "${monitor_log}" | tail -n 1)"
  else
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  MONITOR_PID=""

  # A time-limited rollout is an attended, normal ending.  The monitor has
  # already observed an active policy gate, so do not treat code 27 as a
  # safety fault or invoke the EXIT trap that disables and powers off the arm.
  # The common path below disables only policy/servo motion, then asks the
  # operator whether this no-time-limit round is a success or failure.
  if [[ "${termination_source}" == "episode_timeout" && "${monitor_code}" -eq 27 ]]; then
    termination_source="episode_timeout_${ROLLOUT_TIMEOUT_SECONDS}s"
    monitor_code=0
    echo "Episode reached ${ROLLOUT_TIMEOUT_SECONDS}s. Policy motion will stop; robot power and enable stay on for your label."
  fi

  if (( monitor_code != 0 )); then
    echo "ERROR: unsafe/abnormal rollout stop: ${termination_source} (code ${monitor_code})." >&2
    echo "The current episode will be deleted and the full stack powered off." >&2
    exit "${monitor_code}"
  fi

  "${PROJECT_ROOT}/tools/ubuntu_smolvla_stack.sh" disable-policy || true
  stop_recorder
  echo "Episode ended safely: ${termination_source}"
  while true; do
    read -r -p "Label: S=save success, F=save failure, D=discard/delete: " answer
    case "${answer}" in
      S|s) finalize_current success "${termination_source}"; SAVED=$((SAVED + 1)); break ;;
      F|f) finalize_current failure "${termination_source}"; SAVED=$((SAVED + 1)); break ;;
      D|d) finalize_current discard "${termination_source}"; break ;;
      *) echo "Please enter S, F or D." ;;
    esac
  done
  echo "QGF_COLLECTION_PROGRESS kept=${SAVED}/${TARGET_EPISODES} attempts=${ATTEMPT}"
  (( SAVED < TARGET_EPISODES )) || break

  read -r -p "Reset bottle/box now. Press Enter for the next round, or type Q to finish: " answer
  [[ "${answer}" != "Q" && "${answer}" != "q" ]] || break
  "${PROJECT_ROOT}/tools/ubuntu_smolvla_stack.sh" reset-round
  # Let any in-flight inference finish, then atomically drop all old queued
  # actions. The client/model processes themselves remain alive.
  sleep 3
  call_trigger /smolvla/reset_episode
  wait_observation_and_chunk
  "${PROJECT_ROOT}/tools/ubuntu_smolvla_stack.sh" servo
done

echo "QGF_COLLECTION_COMPLETE kept=${SAVED} attempts=${ATTEMPT}"
