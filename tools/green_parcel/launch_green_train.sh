#!/bin/bash
# Fine-tune SmolVLA on the green-parcel segment. GPU 3 only, as requested.
#
# Same recipe as the red-parcel run (launch_trainings.sh): same init weights,
# batch 64, 20000 steps, save_freq 1000, seed 1000, 45 train episodes with the
# 5 validation episodes held out via --dataset.episodes. Only the dataset, the
# repo id and the output directory differ, so the two checkpoints stay
# comparable.
set -u
export HF_HOME=~/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

GPU=3
RUN=/ssd/zwwl_user2/parcel_smolvla/20260903_green_parcel_clean_50
DS=$RUN/clean_lerobot_v3
OUT=$RUN/outputs/train_green_parcel
INIT=~/parcel_smolvla/models/smolvla_parcel_init
REPO=local/green_parcel_clean

[ -d "$DS" ] || { echo "FATAL: dataset missing: $DS" >&2; exit 2; }
[ -d "$INIT" ] || { echo "FATAL: init weights missing: $INIT" >&2; exit 2; }
[ -f "$RUN/split_manifest.json" ] || { echo "FATAL: split manifest missing" >&2; exit 2; }

# The train list comes from the split the cleaner wrote, so the 5 validation
# episodes can never leak into training through a stale hardcoded list.
TRAIN_EPS=$(~/parcel_env/bin/python -c "
import json
s = json.load(open('$RUN/split_manifest.json'))
tr = s['train']
assert len(set(tr)) == len(tr), 'duplicate train episode'
assert not (set(tr) & set(s['val'])), 'train/val overlap'
print('[' + ','.join(str(i) for i in tr) + ']')
") || { echo "FATAL: could not build the train episode list" >&2; exit 2; }

NTR=$(~/parcel_env/bin/python -c "
import json; print(len(json.load(open('$RUN/split_manifest.json'))['train']))")
NVA=$(~/parcel_env/bin/python -c "
import json; print(len(json.load(open('$RUN/split_manifest.json'))['val']))")
TASK=$(~/parcel_env/bin/python -c "
import json; print(json.load(open('$RUN/split_manifest.json'))['task_text'])")

echo "============================================================"
echo "SmolVLA fine-tune: green parcel"
echo "============================================================"
echo "  GPU            : $GPU (physical, CUDA_VISIBLE_DEVICES)"
echo "  dataset        : $DS"
echo "  task           : $TASK"
echo "  train / val    : $NTR / $NVA"
echo "  init weights   : $INIT"
echo "  output         : $OUT"
echo "  batch 64  steps 20000  save_freq 1000  seed 1000"
echo "============================================================"

# Do NOT create $OUT here: lerobot's TrainPipelineConfig.validate() refuses to
# start when the output directory already exists and resume is off.
rm -rf "$OUT"
echo "$(date '+%m-%d %H:%M:%S') TRAIN_START gpu=$GPU green_parcel" >> "$RUN/PIPELINE.log"

CUDA_VISIBLE_DEVICES=$GPU setsid nohup ~/parcel_env/bin/lerobot-train \
  --dataset.repo_id=$REPO \
  --dataset.root=$DS \
  --dataset.episodes="$TRAIN_EPS" \
  --policy.path=$INIT \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --output_dir=$OUT \
  --batch_size=64 \
  --steps=20000 \
  --save_freq=1000 \
  --seed=1000 \
  --num_workers=8 \
  --wandb.enable=false \
  > "$RUN/train.log" 2>&1 < /dev/null &

echo "launched pid=$! log=$RUN/train.log"
