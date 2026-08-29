#!/bin/bash
# Handoff section 10: dual-camera SmolVLA visual feature extraction.
# Reads only local NVMe; never decodes Orin MP4 over the network.
set -eu
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

E=/opt/qgf_real_robot/envs/visual_iql_py310
QGF_REPO=/opt/qgf_real_robot/repos/SmolVLA-with-QGF
QGF_DATA=/opt/qgf_real_robot/datasets/mug_purple_box_baseline50_20260829/raw_episodes
QGF_RUN=/opt/qgf_real_robot/runs/mug_purple_box_single_q_45_5_20260829
QGF_POLICY=/opt/qgf_real_robot/policy_bundles/mug_purple_box/checkpoint
mkdir -p "$QGF_RUN"/{features,logs,commands}

CMD=(
  "$E/bin/python" "$QGF_REPO/qgf/scripts/extract_smolvla_visual_features.py"
  --raw-episodes-root "$QGF_DATA"
  --aligned-manifest "$QGF_RUN/manifest/aligned_normalized_chunks.parquet"
  --output-dir "$QGF_RUN/features"
  --checkpoint "$QGF_POLICY"
  --device cuda
  --batch-size 32
)

{
  echo "# executed $(date -u +%FT%TZ) on $(hostname)"
  echo "# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
  printf '%q ' "${CMD[@]}"
  echo
} > "$QGF_RUN/commands/extract_visual_features.cmd"

nvidia-smi > "$QGF_RUN/environment/nvidia_smi_extract_start.txt"
echo "=== running ==="
"${CMD[@]}" 2>&1 | tee "$QGF_RUN/logs/extract_visual_features.log"
nvidia-smi > "$QGF_RUN/environment/nvidia_smi_extract_end.txt"
echo "=== features ==="
ls "$QGF_RUN/features" | head -5
ls "$QGF_RUN/features" | wc -l
du -sh "$QGF_RUN/features"
