#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/tools/smolvla_orin_env.sh"

# The QGF telemetry server publishes normalized chunks through ROS 2.  Load
# rclpy here as an explicit launcher dependency instead of relying on a parent
# shell to have sourced Humble already.
set +u
# shellcheck disable=SC1091
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
set -u

[[ "$(uname -m)" == "aarch64" ]] || {
  echo "ERROR: this launcher is only for the Jetson AGX Orin (aarch64)." >&2
  exit 2
}
[[ -f "${SMOLVLA_SERVER_MODEL_PATH}/config.json" ]] || {
  echo "ERROR: checkpoint is incomplete: ${SMOLVLA_SERVER_MODEL_PATH}" >&2
  exit 2
}

cd "${SMOLVLA_ORIN_BUNDLE}"
exec "${PROJECT_ROOT}/tools/start_smolvla_policy_server.sh"
