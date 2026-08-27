#!/usr/bin/env bash
# smolvla_20260827_mug_purple_box task profile (2026-08-27). Presets the bundle then defers to the common template.
export SMOLVLA_ORIN_BUNDLE=/home/nvidia/work/telop/models/smolvla_20260827_mug_purple_box
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_here}/smolvla_orin_env.sh"
