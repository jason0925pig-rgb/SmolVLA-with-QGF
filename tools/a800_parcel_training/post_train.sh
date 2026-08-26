#!/bin/bash
# After training: validate all checkpoints on GPU7, select the best, checksum it.
set -u
R=~/parcel_smolvla/20260825_parcel_50
PY=~/parcel_env/bin/python
OUT=$R/outputs/train_20260825_parcel50_smolvla
export CUDA_VISIBLE_DEVICES=7
export HF_HOME=~/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
mark() { echo "$(date '+%m-%d %H:%M:%S') $1" >> "$R/PIPELINE.log"; }

mark "POSTTRAIN_START"
mkdir -p $R/reports
REPORTS=()
for step in 005000 010000 015000 020000; do
  CK=$OUT/checkpoints/$step/pretrained_model
  [ -d "$CK" ] || CK=$OUT/checkpoints/$step
  [ -d "$CK" ] || { mark "POSTTRAIN_MISSING_$step"; continue; }
  $PY $R/validate_smolvla_checkpoint.py \
    --checkpoint "$CK" \
    --dataset-root $R/dataset \
    --repo-id local/onearm_parcel_50 \
    --device cuda \
    --seed 42 \
    --report $R/reports/validate_$step.json \
    > $R/reports/validate_$step.log 2>&1
  if [ $? -eq 0 ]; then
    mark "VALIDATE_OK_$step"
    REPORTS+=("--report" "$R/reports/validate_$step.json")
  else
    mark "VALIDATE_FAIL_$step"
  fi
done

if [ ${#REPORTS[@]} -eq 0 ]; then mark "POSTTRAIN_FAIL no_reports"; exit 1; fi
$PY $R/select_smolvla_checkpoint.py "${REPORTS[@]}" --output $R/reports/selected.json \
  > $R/reports/select.log 2>&1 || { mark "SELECT_FAIL"; exit 1; }
mark "SELECT_DONE $(grep -o '"checkpoint"[^,}]*' $R/reports/selected.json | head -1)"

# checksum the selected checkpoint directory
BEST=$(python3 -c "import json;print(json.load(open('$R/reports/selected.json')).get('checkpoint',''))" 2>/dev/null)
if [ -n "$BEST" ] && [ -d "$BEST" ]; then
  (cd "$BEST" && find . -type f -exec sha256sum {} \; | LC_ALL=C sort > SHA256SUMS)
  mark "POSTTRAIN_DONE best=$BEST"
else
  mark "POSTTRAIN_DONE_NO_BEST"
fi
