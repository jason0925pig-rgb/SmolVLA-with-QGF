#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/tools/smolvla_orin_env.sh"

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
