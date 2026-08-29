#!/bin/bash
# Handoff section 9: build the aligned manifest + stratified 45/5 split.
# GPU 0 per the user's decision (see environment.txt for why).
set -eu
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1

E=/opt/qgf_real_robot/envs/visual_iql_py310
QGF_REPO=/opt/qgf_real_robot/repos/SmolVLA-with-QGF
QGF_DATA=/opt/qgf_real_robot/datasets/mug_purple_box_baseline50_20260829/raw_episodes
QGF_RUN=/opt/qgf_real_robot/runs/mug_purple_box_single_q_45_5_20260829
mkdir -p "$QGF_RUN"/{manifest,logs,commands}

CMD=(
  "$E/bin/python" "$QGF_REPO/qgf/scripts/build_real_robot_visual_iql_manifest.py"
  --raw-episodes-root "$QGF_DATA"
  --output-dir "$QGF_RUN/manifest"
  --episode-first 0
  --episode-last 49
  --action-horizon 50
  --policy-hz 15
  --max-transition-gap-seconds 0.1
  --val-count 5
  --split-seed 20260814
)

{
  echo "# executed $(date -u +%FT%TZ) on $(hostname)"
  echo "# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1"
  printf '%q ' "${CMD[@]}"
  echo
} > "$QGF_RUN/commands/build_manifest.cmd"

echo "=== running ==="
printf '%q ' "${CMD[@]}"; echo
"${CMD[@]}" 2>&1 | tee "$QGF_RUN/logs/build_manifest.log"

echo
echo "=== manifest dir ==="
ls -la "$QGF_RUN/manifest"
