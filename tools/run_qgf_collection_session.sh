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
COMPARISON_TAG_B64="${QGF_COMPARISON_TAG_B64:-}"
RUN_MODE="${QGF_RUN_MODE:-baseline}"
INITIAL_MODE="${QGF_INITIAL_MODE:-${RUN_MODE}}"
QGF_BETA="${QGF_BETA:-0}"
BASELINE_TARGET="${QGF_BASELINE_EPISODE_COUNT:-0}"
QGF_TARGET="${QGF_QGF_EPISODE_COUNT:-0}"
INITIAL_SAVED_BASELINE="${QGF_INITIAL_SAVED_BASELINE:-0}"
INITIAL_SAVED_QGF="${QGF_INITIAL_SAVED_QGF:-0}"
TASK="$(printf '%s' "${TASK_B64}" | base64 --decode)"
NOTES="$(printf '%s' "${NOTES_B64}" | base64 --decode)"
COMPARISON_TAG="$(printf '%s' "${COMPARISON_TAG_B64}" | base64 --decode)"
# The Windows task-profile launcher supplies the task as base64 so Chinese
# text survives the SSH command line.  The ROS policy-client launcher reads
# SMOLVLA_TASK (not SMOLVLA_TASK_B64), so export the decoded value before the
# managed client process is created.  Without this, it silently falls back to
# the old bottle task prompt.
export SMOLVLA_TASK="${TASK}"
# A task profile must select one exact bundle and checkpoint.  Do not inherit a
# previous rollout's SMOLVLA_SERVER_MODEL_PATH: that can silently run a
# different task model even though the terminal header prints the new task.
EXPECTED_CHECKPOINT="${SMOLVLA_EXPECTED_CHECKPOINT:-${SMOLVLA_ORIN_BUNDLE}/checkpoint}"
[[ -f "${EXPECTED_CHECKPOINT}/config.json" ]] || {
  echo "ERROR: selected task checkpoint is incomplete: ${EXPECTED_CHECKPOINT}" >&2
  exit 2
}
export SMOLVLA_SERVER_MODEL_PATH="${EXPECTED_CHECKPOINT}"
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
SAVED_BASELINE="${INITIAL_SAVED_BASELINE}"
SAVED_QGF="${INITIAL_SAVED_QGF}"
ATTEMPT=0
PAIRED_MODE=0
CURRENT_MODE="${RUN_MODE}"

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
  : >"${pid_file}"
  # GNU setsid can fork when it is invoked from a background job.  Recording
  # the short-lived parent PID leaves orphan policy/client processes behind,
  # which then occupy port 8080 and retain the preceding task model.  Let the
  # new-session child record its own PID before exec so stop_managed always
  # addresses the real process group.
  nohup setsid --fork bash -c '
    pid_file="$1"
    shift
    echo "$$" >"${pid_file}"
    exec "$@"
  ' bash "${pid_file}" "$@" >"${log_file}" 2>&1 < /dev/null &
  local deadline=$((SECONDS + 3))
  while [[ ! -s "${pid_file}" ]] && (( SECONDS < deadline )); do sleep 0.05; done
  [[ -s "${pid_file}" ]] || {
    echo "ERROR: managed process did not publish its PID: $*" >&2
    return 1
  }
}

stop_managed() {
  local file="$1" pid deadline
  [[ -s "${file}" ]] || return 0
  pid="$(<"${file}")"
  if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    deadline=$((SECONDS + 5))
    while kill -0 "${pid}" 2>/dev/null && (( SECONDS < deadline )); do sleep 0.1; done
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    fi
  fi
  rm -f -- "${file}"
}

cleanup_orphaned_policy_processes() {
  local pid pgid deadline any_alive
  local -a stale_pids=()
  mapfile -t stale_pids < <(pgrep -u "${UID}" -f \
    '[l]erobot_robot_armstrong_ros2\.policy_server_(telemetry|qgf)|[l]erobot\.async_inference\.policy_server|[l]erobot_robot_armstrong_ros2\.async_client' \
    2>/dev/null || true)
  ((${#stale_pids[@]} > 0)) || return 0

  echo "WARNING: clearing ${#stale_pids[@]} orphaned SmolVLA policy process(es) from an earlier session." >&2
  for pid in "${stale_pids[@]}"; do
    [[ "${pid}" =~ ^[0-9]+$ ]] || continue
    pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
    if [[ "${pgid}" =~ ^[0-9]+$ ]]; then
      kill -TERM -- "-${pgid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    else
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done

  deadline=$((SECONDS + 5))
  while (( SECONDS < deadline )); do
    any_alive=0
    for pid in "${stale_pids[@]}"; do kill -0 "${pid}" 2>/dev/null && any_alive=1; done
    (( any_alive == 0 )) && break
    sleep 0.1
  done
  for pid in "${stale_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
      if [[ "${pgid}" =~ ^[0-9]+$ ]]; then
        kill -KILL -- "-${pgid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
      else
        kill -KILL "${pid}" 2>/dev/null || true
      fi
    fi
  done
  rm -f -- "${SERVER_PID_FILE}" "${CLIENT_PID_FILE}"
  echo "STALE_SMOLVLA_POLICY_PROCESSES_CLEARED count=${#stale_pids[@]}"
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
  cleanup_orphaned_policy_processes
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

# When changing from baseline to QGF (or back), do not let a just-stopped
# process keep the TCP port alive.  Otherwise the readiness probe could attach
# to the old server and silently retain the previous policy mode.
wait_for_server_stopped() {
  "${SMOLVLA_ORIN_VENV}/bin/python" - "${SMOLVLA_SERVER_HOST}" "${SMOLVLA_SERVER_PORT}" <<'PY'
import socket, sys, time
host, port = sys.argv[1], int(sys.argv[2])
deadline = time.monotonic() + 15
while time.monotonic() < deadline:
    try:
        with socket.create_connection((host, port), timeout=0.4):
            time.sleep(0.2)
    except OSError:
        raise SystemExit(0)
raise SystemExit("previous policy server did not release its TCP port")
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

print_comparison_stats() {
  local args=("${PROJECT_ROOT}/tools/summarize_qgf_comparison.py" --dataset-root "${DATASET_ROOT}")
  if [[ -n "${COMPARISON_TAG}" ]]; then
    args+=(--tag "${COMPARISON_TAG}")
  fi
  "${SMOLVLA_ORIN_VENV}/bin/python" "${args[@]}"
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
  local outcome="$1" termination_source="$2" notes="${NOTES}"
  if (( PAIRED_MODE )); then
    notes="${NOTES}; policy_mode=${CURRENT_MODE}"
  fi
  "${SMOLVLA_ORIN_VENV}/bin/python" "${PROJECT_ROOT}/tools/finalize_qgf_episode.py" \
    --staging "${CURRENT_STAGING}" --dataset-root "${DATASET_ROOT}" \
    --outcome "${outcome}" --task "${TASK}" --notes "${notes}" \
    --termination-source "${termination_source}"
  CURRENT_STAGING=""
}

record_abnormal_episode_then_stop() {
  local termination_source="$1" monitor_code="$2" answer

  # The monitor has already observed a safety condition.  Stop policy/servo
  # immediately; never leave the robot moving while waiting for keyboard
  # input.  Keep the recorder staging directory intact until the operator
  # explicitly decides whether the evidence should be kept or discarded.
  "${PROJECT_ROOT}/tools/ubuntu_smolvla_stack.sh" disable-policy || true
  stop_recorder
  echo ""
  echo "SAFETY_STOP_REVIEW source=${termination_source} code=${monitor_code}"
  echo "Motion is stopped. Robot shutdown will follow after you label this episode."
  echo "The final metadata will retain termination_source=${termination_source}."

  while true; do
    read -r -p "Abnormal stop: S=save success, F=save failure, D=discard/delete: " answer
    case "${answer}" in
      S|s)
        finalize_current success "${termination_source}"
        if (( PAIRED_MODE )); then
          if [[ "${CURRENT_MODE}" == "baseline" ]]; then SAVED_BASELINE=$((SAVED_BASELINE + 1)); else SAVED_QGF=$((SAVED_QGF + 1)); fi
        else
          SAVED=$((SAVED + 1))
        fi
        break
        ;;
      F|f)
        finalize_current failure "${termination_source}"
        if (( PAIRED_MODE )); then
          if [[ "${CURRENT_MODE}" == "baseline" ]]; then SAVED_BASELINE=$((SAVED_BASELINE + 1)); else SAVED_QGF=$((SAVED_QGF + 1)); fi
        else
          SAVED=$((SAVED + 1))
        fi
        break
        ;;
      D|d)
        finalize_current discard "${termination_source}"
        break
        ;;
      *) echo "Please enter S, F or D." ;;
    esac
  done
  print_comparison_stats
}

if alive_pid_file "${SERVER_PID_FILE}" || alive_pid_file "${CLIENT_PID_FILE}"; then
  echo "ERROR: another SmolVLA launcher is already running." >&2
  exit 3
fi
cleanup_orphaned_policy_processes

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

configure_policy_mode() {
  local mode="$1"
  case "${mode}" in
    baseline)
      export SMOLVLA_POLICY_SERVER_MODULE="lerobot_robot_armstrong_ros2.policy_server_telemetry"
      unset SMOLVLA_QGF_BETA
      ;;
    qgf)
    python3 - "${QGF_BETA}" <<'PY'
import math
import sys
beta = float(sys.argv[1])
if not math.isfinite(beta) or beta <= 0.0:
    raise SystemExit("ERROR: QGF_BETA must be a finite positive value in qgf mode.")
print(f"QGF_CONFIG beta={beta:.8g} coefficient=1/beta={1.0 / beta:.8g}")
PY
      export SMOLVLA_POLICY_SERVER_MODULE="lerobot_robot_armstrong_ros2.policy_server_qgf"
      export SMOLVLA_QGF_BETA="${QGF_BETA}"
      export SMOLVLA_QGF_GRAD_CLIP_NORM="${SMOLVLA_QGF_GRAD_CLIP_NORM:-1.0}"
      [[ -f "${SMOLVLA_QGF_CRITIC_PATH}" ]] || {
        echo "ERROR: QGF critic checkpoint is missing: ${SMOLVLA_QGF_CRITIC_PATH}" >&2; return 1;
      }
      ;;
    *)
      echo "ERROR: policy mode must be baseline or qgf, got ${mode}." >&2; return 1;
      ;;
  esac
}

start_policy_server() {
  : >"${SERVER_LOG}"
  start_managed "${SERVER_PID_FILE}" "${SERVER_LOG}" \
    env "SMOLVLA_ORIN_BUNDLE=${SMOLVLA_ORIN_BUNDLE}" \
      "SMOLVLA_SERVER_MODEL_PATH=${SMOLVLA_SERVER_MODEL_PATH}" \
      "${PROJECT_ROOT}/tools/start_smolvla_orin_policy_server.sh"
  wait_for_server
  # Orphaned listeners are cleared before launch.  Therefore the listener that
  # just passed wait_for_server belongs to this invocation; a launcher-PID
  # kill -0 check is not a reliable readiness test across setsid/Python handoff.
  echo "SMOLVLA_POLICY_SERVER_READY endpoint=${SMOLVLA_SERVER_HOST}:${SMOLVLA_SERVER_PORT} checkpoint=${SMOLVLA_SERVER_MODEL_PATH}"
}

start_policy_client() {
  : >"${CLIENT_LOG}"
  start_managed "${CLIENT_PID_FILE}" "${CLIENT_LOG}" \
    env "SMOLVLA_ORIN_BUNDLE=${SMOLVLA_ORIN_BUNDLE}" \
      "SMOLVLA_SERVER_MODEL_PATH=${SMOLVLA_SERVER_MODEL_PATH}" \
      "SMOLVLA_TASK=${TASK}" \
      "${PROJECT_ROOT}/tools/start_smolvla_orin_ros_client.sh"
  sleep 2
  alive_pid_file "${CLIENT_PID_FILE}" || {
    tail -n 100 "${CLIENT_LOG}" >&2 || true; return 1;
  }
  wait_observation_and_chunk
  grep -Fq "pretrained_name_or_path': '${SMOLVLA_SERVER_MODEL_PATH}'" "${CLIENT_LOG}" || {
    echo "ERROR: policy client checkpoint identity mismatch; expected ${SMOLVLA_SERVER_MODEL_PATH}." >&2
    tail -n 120 "${CLIENT_LOG}" >&2 || true
    return 1
  }
  grep -Fq "'task': '${TASK}'" "${CLIENT_LOG}" || {
    echo "ERROR: policy client task identity mismatch; expected ${TASK}." >&2
    tail -n 120 "${CLIENT_LOG}" >&2 || true
    return 1
  }
  echo "SMOLVLA_POLICY_IDENTITY_OK checkpoint=${SMOLVLA_SERVER_MODEL_PATH} task=${TASK}"
}

stop_policy_components() {
  stop_managed "${CLIENT_PID_FILE}"
  stop_managed "${SERVER_PID_FILE}"
  wait_for_server_stopped
}

start_policy_for_mode() {
  CURRENT_MODE="$1"
  configure_policy_mode "${CURRENT_MODE}"
  start_policy_server
  start_policy_client
}

case "${RUN_MODE}" in
  baseline|qgf)
    is_positive_integer "${TARGET_EPISODES}" || {
      echo "ERROR: QGF_EPISODE_COUNT must be a positive integer." >&2; exit 2;
    }
    CURRENT_MODE="${RUN_MODE}"
    ;;
  paired)
    PAIRED_MODE=1
    is_positive_integer "${BASELINE_TARGET}" || {
      echo "ERROR: QGF_BASELINE_EPISODE_COUNT must be a positive integer in paired mode." >&2; exit 2;
    }
    is_positive_integer "${QGF_TARGET}" || {
      echo "ERROR: QGF_QGF_EPISODE_COUNT must be a positive integer in paired mode." >&2; exit 2;
    }
    is_nonnegative_integer "${SAVED_BASELINE}" && (( SAVED_BASELINE <= BASELINE_TARGET )) || {
      echo "ERROR: QGF_INITIAL_SAVED_BASELINE must be an integer between 0 and the baseline target." >&2; exit 2;
    }
    is_nonnegative_integer "${SAVED_QGF}" && (( SAVED_QGF <= QGF_TARGET )) || {
      echo "ERROR: QGF_INITIAL_SAVED_QGF must be an integer between 0 and the QGF target." >&2; exit 2;
    }
    case "${INITIAL_MODE}" in baseline|qgf) CURRENT_MODE="${INITIAL_MODE}" ;; *)
      echo "ERROR: QGF_INITIAL_MODE must be baseline or qgf in paired mode." >&2; exit 2;; esac
    ;;
  *)
    echo "ERROR: QGF_RUN_MODE must be baseline, qgf or paired, got ${RUN_MODE}." >&2; exit 2;
    ;;
esac
configure_policy_mode "${CURRENT_MODE}"

echo "============================================================"
echo "Continuous real-robot QGF collection"
echo "Task: ${TASK}"
if (( PAIRED_MODE )); then
  echo "Interactive paired mode: initial=${CURRENT_MODE}; baseline_target=${BASELINE_TARGET}; qgf_target=${QGF_TARGET}"
else
  echo "Policy mode: ${RUN_MODE}"
  echo "Target kept episodes: ${TARGET_EPISODES}"
fi
[[ -z "${COMPARISON_TAG}" ]] || echo "Comparison cohort tag: ${COMPARISON_TAG}"
echo "ARM and MOVE are entered only once. Power/model/cameras stay up between rounds."
echo "Ctrl+C/ABORT deletes the current episode immediately. Physical safety stops"
echo "and policy safety guards stop motion first, then ask whether to save S/F or delete D."
echo "============================================================"

"${SMOLVLA_ORIN_VENV}/bin/python" "${PROJECT_ROOT}/tools/snapshot_qgf_provenance.py" \
  --project-root "${PROJECT_ROOT}" \
  --checkpoint "${SMOLVLA_SERVER_MODEL_PATH}" \
  --dataset-root "${DATASET_ROOT}"

start_policy_server
"${PROJECT_ROOT}/tools/ubuntu_smolvla_stack.sh" start

read -r -p "Type ARM once to power and enable the right arm: " answer
[[ "${answer}" == "ARM" ]] || exit 0
"${PROJECT_ROOT}/tools/ubuntu_smolvla_stack.sh" prepare

start_policy_client
"${PROJECT_ROOT}/tools/ubuntu_smolvla_stack.sh" servo

while true; do
  if (( PAIRED_MODE )); then
    (( SAVED_BASELINE < BASELINE_TARGET || SAVED_QGF < QGF_TARGET )) || break
  else
    (( SAVED < TARGET_EPISODES )) || break
  fi
  ATTEMPT=$((ATTEMPT + 1))
  CURRENT_STAGING="${DATASET_ROOT}/.staging/$(date +%Y%m%d_%H%M%S)_attempt_$(printf '%04d' "${ATTEMPT}")"
  start_recorder

  if (( ATTEMPT == 1 )); then
    print_comparison_stats
    read -r -p "Type MOVE once to start episode 1: " answer
    [[ "${answer}" == "MOVE" ]] || exit 0
  else
    if (( PAIRED_MODE )); then
      echo "NEXT_PAIRED_EPISODE_READY mode=${CURRENT_MODE}; starting after the confirmed scene reset."
    else
      echo "NEXT_EPISODE_READY episode=$((SAVED + 1)); starting after the confirmed scene reset."
    fi
  fi

  "${PROJECT_ROOT}/tools/ubuntu_smolvla_stack.sh" enable-policy
  monitor_log="${CURRENT_STAGING}/monitor.log"
  python3 "${PROJECT_ROOT}/tools/monitor_qgf_rollout.py" \
    --action-gate-confirmed --timeout "${ROLLOUT_TIMEOUT_SECONDS}" >"${monitor_log}" 2>&1 &
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
    record_abnormal_episode_then_stop "${termination_source}" "${monitor_code}"
    echo "Abnormal episode review completed; the full stack will now be powered off." >&2
    exit "${monitor_code}"
  fi

  "${PROJECT_ROOT}/tools/ubuntu_smolvla_stack.sh" disable-policy || true
  stop_recorder
  echo "Episode ended safely: ${termination_source}"
  while true; do
    read -r -p "Label: S=save success, F=save failure, D=discard/delete: " answer
    case "${answer}" in
      S|s)
        finalize_current success "${termination_source}"
        if (( PAIRED_MODE )); then
          if [[ "${CURRENT_MODE}" == "baseline" ]]; then SAVED_BASELINE=$((SAVED_BASELINE + 1)); else SAVED_QGF=$((SAVED_QGF + 1)); fi
        else
          SAVED=$((SAVED + 1))
        fi
        break
        ;;
      F|f)
        finalize_current failure "${termination_source}"
        if (( PAIRED_MODE )); then
          if [[ "${CURRENT_MODE}" == "baseline" ]]; then SAVED_BASELINE=$((SAVED_BASELINE + 1)); else SAVED_QGF=$((SAVED_QGF + 1)); fi
        else
          SAVED=$((SAVED + 1))
        fi
        break
        ;;
      D|d) finalize_current discard "${termination_source}"; break ;;
      *) echo "Please enter S, F or D." ;;
    esac
  done
  print_comparison_stats
  if (( PAIRED_MODE )); then
    echo "QGF_PAIRED_PROGRESS baseline=${SAVED_BASELINE}/${BASELINE_TARGET} qgf=${SAVED_QGF}/${QGF_TARGET} attempts=${ATTEMPT}"
    (( SAVED_BASELINE < BASELINE_TARGET || SAVED_QGF < QGF_TARGET )) || break

    while true; do
      read -r -p "Next policy: B=baseline, Q/G=QGF, X=finish session: " answer
      case "${answer}" in
        B|b|baseline|BASELINE)
          NEXT_MODE="baseline"
          if (( SAVED_BASELINE >= BASELINE_TARGET )); then echo "Baseline target is already complete."; else break; fi
          ;;
        G|g|Q|q|qgf|QGF)
          NEXT_MODE="qgf"
          if (( SAVED_QGF >= QGF_TARGET )); then echo "QGF target is already complete."; else break; fi
          ;;
        X|x)
          NEXT_MODE=""
          break
          ;;
        *) echo "Please enter B, Q/G or X." ;;
      esac
    done
    [[ -n "${NEXT_MODE}" ]] || break

    # Stopping the old client publishes STOP, which intentionally disables the
    # safe gripper.  Stop it first, then reset/re-enable the gripper.  The old
    # order did this backwards and left every later paired round with the
    # gripper disabled even when the model requested close.
    stop_policy_components
    "${PROJECT_ROOT}/tools/ubuntu_smolvla_stack.sh" reset-round
    start_policy_for_mode "${NEXT_MODE}"
    "${PROJECT_ROOT}/tools/ubuntu_smolvla_stack.sh" servo
    read -r -p "${CURRENT_MODE} is ready. Reset the scene, then press Enter to start the next round (or X to finish): " answer
    [[ "${answer}" != "X" && "${answer}" != "x" ]] || break
  else
    echo "QGF_COLLECTION_PROGRESS kept=${SAVED}/${TARGET_EPISODES} attempts=${ATTEMPT}"
    (( SAVED < TARGET_EPISODES )) || break
    read -r -p "Reset bottle/box now. The comparison success rates are above. Press Enter for the next round, or type Q to finish: " answer
    [[ "${answer}" != "Q" && "${answer}" != "q" ]] || break
    "${PROJECT_ROOT}/tools/ubuntu_smolvla_stack.sh" reset-round
    # Let any in-flight inference finish, then atomically drop all old queued
    # actions. The client/model processes themselves remain alive.
    sleep 3
    call_trigger /smolvla/reset_episode
    wait_observation_and_chunk
    "${PROJECT_ROOT}/tools/ubuntu_smolvla_stack.sh" servo
  fi
done

if (( PAIRED_MODE )); then
  echo "QGF_PAIRED_COLLECTION_COMPLETE baseline=${SAVED_BASELINE}/${BASELINE_TARGET} qgf=${SAVED_QGF}/${QGF_TARGET} attempts=${ATTEMPT}"
else
  echo "QGF_COLLECTION_COMPLETE kept=${SAVED} attempts=${ATTEMPT}"
fi
print_comparison_stats
