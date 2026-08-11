#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_SETUP="/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/tools/smolvla_orin_env.sh"
WORKSPACE_SETUP="${TELEOP_PROJECT_ROOT}/install/setup.bash"

[[ -r "${ROS_SETUP}" ]] || { echo "ERROR: missing ${ROS_SETUP}." >&2; exit 2; }
[[ -r "${WORKSPACE_SETUP}" ]] || { echo "ERROR: missing ${WORKSPACE_SETUP}." >&2; exit 2; }
# shellcheck disable=SC1090
set +u
source "${ROS_SETUP}"
# shellcheck disable=SC1090
source "${WORKSPACE_SETUP}"
set -u
[[ "$(uname -m)" == "aarch64" ]] || { echo "ERROR: expected aarch64." >&2; exit 2; }
[[ -x "${SMOLVLA_VENV}/bin/python" ]] || { echo "ERROR: missing ${SMOLVLA_VENV}." >&2; exit 2; }
[[ -f "${SMOLVLA_SERVER_MODEL_PATH}/config.json" ]] || { echo "ERROR: missing checkpoint." >&2; exit 2; }
[[ -d "${SMOLVLA_ORIN_BUNDLE}/vlm" ]] || { echo "ERROR: missing bundled VLM." >&2; exit 2; }

"${SMOLVLA_VENV}/bin/python" "${PROJECT_ROOT}/tools/check_smolvla_orin_python.py"

echo "This check did not start ROS, cameras, robot power, robot enable or motion."
