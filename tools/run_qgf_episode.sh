#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/tools/smolvla_orin_env.sh"

EPISODE_DIR="${QGF_EPISODE_DIR:?QGF_EPISODE_DIR is required}"
TASK_B64="${SMOLVLA_TASK_B64:?SMOLVLA_TASK_B64 is required}"
TASK="$(printf '%s' "${TASK_B64}" | base64 --decode)"
RECORDER_PID=""
RECORDER_LOG="${EPISODE_DIR}/recorder.log"

mkdir -p "${EPISODE_DIR}"

stop_recorder() {
  local deadline
  [[ -n "${RECORDER_PID}" ]] || return 0
  if kill -0 "${RECORDER_PID}" 2>/dev/null; then
    kill -INT "${RECORDER_PID}" 2>/dev/null || true
    deadline=$((SECONDS + 20))
    while kill -0 "${RECORDER_PID}" 2>/dev/null && (( SECONDS < deadline )); do
      sleep 0.1
    done
    if kill -0 "${RECORDER_PID}" 2>/dev/null; then
      kill -TERM "${RECORDER_PID}" 2>/dev/null || true
    fi
  fi
  wait "${RECORDER_PID}" 2>/dev/null || true
  RECORDER_PID=""
}

trap stop_recorder EXIT INT TERM HUP

set +u
# shellcheck disable=SC1091
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
# shellcheck disable=SC1091
source "${TELEOP_PROJECT_ROOT}/install/setup.bash"
set -u

"${SMOLVLA_ORIN_VENV}/bin/python" "${PROJECT_ROOT}/tools/qgf_episode_recorder.py" \
  --output-dir "${EPISODE_DIR}" \
  --task "${TASK}" \
  --sample-hz "${SMOLVLA_FPS}" \
  --camera-fps 30 >"${RECORDER_LOG}" 2>&1 &
RECORDER_PID=$!
sleep 1
kill -0 "${RECORDER_PID}" 2>/dev/null || {
  echo "ERROR: QGF recorder exited during startup." >&2
  tail -n 80 "${RECORDER_LOG}" >&2 || true
  exit 10
}

echo "QGF_RECORDER_READY_NO_MOTION episode_dir=${EPISODE_DIR}"
set +e
SMOLVLA_TASK_B64="${TASK_B64}" "${PROJECT_ROOT}/tools/run_smolvla_orin.sh"
launcher_code=$?
set -e
stop_recorder
python3 - "${EPISODE_DIR}" "${launcher_code}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1]) / "launcher_result.json"
path.write_text(json.dumps({
    "launcher_exit_code": int(sys.argv[2]),
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
}, indent=2), encoding="utf-8")
PY
echo "QGF_CAPTURE_FINISHED staging=${EPISODE_DIR} launcher_exit_code=${launcher_code}"
exit "${launcher_code}"
