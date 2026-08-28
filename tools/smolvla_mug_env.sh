#!/usr/bin/env bash
export SMOLVLA_ORIN_BUNDLE=/home/nvidia/work/telop/models/smolvla_20260827_mug_purple_box
# This checkpoint was trained from the controller's raw joint coordinates.
export SMOLVLA_CANONICALIZE_POLICY_OBSERVATION=false
export SMOLVLA_INITIAL_POSE_TOLERANCE_RAD=0.3490658503988659
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_here}/smolvla_orin_env.sh"
