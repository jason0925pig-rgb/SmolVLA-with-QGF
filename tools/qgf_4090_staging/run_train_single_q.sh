#!/bin/bash
# Handoff section 11: single-critic visual IQL.  Every hyperparameter is the
# water-bottle configuration; the ONLY difference is the 45/5 split.
# GPU 0 by the user's decision (LingBot pins itself to GPU 1).
set -eu
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

E=/opt/qgf_real_robot/envs/visual_iql_py310
QGF_REPO=/opt/qgf_real_robot/repos/SmolVLA-with-QGF
QGF_RUN=/opt/qgf_real_robot/runs/mug_purple_box_single_q_45_5_20260829
OUT=$QGF_RUN/outputs/single_qcritic
mkdir -p "$QGF_RUN"/{outputs,logs,commands,environment}

# --- refuse to start if the feature cache is incomplete -----------------------
N=$(ls "$QGF_RUN"/features/*.pt 2>/dev/null | wc -l)
if [ "$N" -ne 50 ]; then
  echo "FATAL: expected 50 feature caches, found $N.  Not starting training."
  exit 1
fi
echo "feature caches: $N/50"

# --- refuse to share the card with LingBot ------------------------------------
if pgrep -af lingbot | grep -v pgrep > /dev/null 2>&1; then
  echo "FATAL: a lingbot process is running.  Stop it or re-check the GPU policy."
  pgrep -af lingbot | grep -v pgrep
  exit 1
fi

CMD=(
  "$E/bin/python" "$QGF_REPO/qgf/scripts/train_real_robot_visual_iql.py"
  --data-dir "$QGF_RUN/features"
  --split-file "$QGF_RUN/manifest/episode_split_45_5.json"
  --output-dir "$OUT"
  --ensemble-size 1
  --epochs 80
  --batch-size 16
  --lr 3e-4
  --weight-decay 1e-4
  --gamma 0.99
  --expectile 0.7
  --polyak 0.005
  --d-model 256
  --layers 3
  --heads 4
  --dropout 0.1
  --seed 20260814
  --device cuda
  --expected-train-episodes 45
  --expected-val-episodes 5
)

{
  echo "# executed $(date -u +%FT%TZ) on $(hostname)"
  echo "# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0"
  echo "# uncertainty gate disabled by ensemble-size 1 (uncertainty_scale=0.0 at deploy)"
  printf '%q ' "${CMD[@]}"
  echo
} > "$QGF_RUN/commands/train_single_qcritic.cmd"

nvidia-smi > "$QGF_RUN/environment/nvidia_smi_train_before.txt"
echo "=== training start $(date -u +%FT%TZ) ==="
printf '%q ' "${CMD[@]}"; echo
"${CMD[@]}" 2>&1 | tee "$QGF_RUN/logs/train_single_qcritic.log"
RC=${PIPESTATUS[0]}
nvidia-smi > "$QGF_RUN/environment/nvidia_smi_train_after.txt"
echo "=== training end $(date -u +%FT%TZ) rc=$RC ==="
ls -la "$OUT" 2>/dev/null | head -20
exit "$RC"
