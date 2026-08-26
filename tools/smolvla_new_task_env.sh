#!/usr/bin/env bash
# Parcel-sorting task profile (new 50-demo checkpoint, 2026-08-25).
# Usage:  source tools/smolvla_new_task_env.sh   (INSTEAD of smolvla_orin_env.sh)
# It only presets the bundle path, then defers everything else to the
# common template. The old water-bottle profile stays untouched.
export SMOLVLA_ORIN_BUNDLE=/home/nvidia/work/telop/models/smolvla_20260825_parcel_50
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_here}/smolvla_orin_env.sh"
