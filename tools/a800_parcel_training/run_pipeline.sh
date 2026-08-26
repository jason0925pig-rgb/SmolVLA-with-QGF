#!/bin/bash
# Parcel-task SmolVLA training pipeline: subset -> verify -> train (GPU7 only).
# Runs detached on the A800. Progress markers land in ~/parcel_smolvla/20260825_parcel_50/PIPELINE.log
set -u
R=~/parcel_smolvla/20260825_parcel_50
PY=~/parcel_env/bin/python
LOG=$R/PIPELINE.log
mark() { echo "$(date '+%m-%d %H:%M:%S') $1" >> "$LOG"; }

mark "PIPELINE_START"

# ---- 1. build the 50-episode subset dataset
$PY $R/build_parcel_subset.py > $R/subset.log 2>&1
if [ $? -ne 0 ]; then mark "SUBSET_FAIL"; exit 1; fi
mark "SUBSET_DONE"

# ---- 2. verify: load with lerobot 0.4.4, sample frames, check shapes
$PY - > $R/verify.log 2>&1 <<'PYEOF'
import numpy as np
from pathlib import Path
root = Path.home() / "parcel_smolvla/20260825_parcel_50/dataset"
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except Exception:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/onearm_parcel_50", root=str(root))
print("episodes:", ds.num_episodes, "frames:", ds.num_frames, "fps:", ds.fps)
assert ds.num_episodes == 50, ds.num_episodes
for i in [0, len(ds)//2, len(ds)-1]:
    item = ds[i]
    st = item["observation.state"]; ac = item["action"]
    assert st.shape[-1] == 8 and ac.shape[-1] == 8
    assert np.isfinite(st.numpy()).all() and np.isfinite(ac.numpy()).all()
    for cam in ["observation.images.chest", "observation.images.wrist_right"]:
        img = item[cam]
        assert img.shape[0] == 3 and img.shape[1] > 100, (cam, img.shape)
    print(f"frame {i}: ok  task={item.get('task','')[:20]}")
print("VERIFY_OK")
PYEOF
grep -q "VERIFY_OK" $R/verify.log || { mark "VERIFY_FAIL"; exit 1; }
mark "VERIFY_DONE"

# ---- 3. train: fresh checkpoint from official base, GPU7 only, 20k steps
export CUDA_VISIBLE_DEVICES=7
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/hf_cache
export TOKENIZERS_PARALLELISM=false
OUT=$R/outputs/train_20260825_parcel50_smolvla
rm -rf "$OUT"
mark "TRAIN_START gpu=7 steps=20000 batch=64 seed=42"
~/parcel_env/bin/lerobot-train \
  --dataset.repo_id=local/onearm_parcel_50 \
  --dataset.root=$R/dataset \
  --policy.path=lerobot/smolvla_base --policy.push_to_hub=false \
  --policy.device=cuda \
  --output_dir=$OUT \
  --batch_size=64 \
  --steps=20000 \
  --save_freq=5000 \
  --seed=42 \
  --num_workers=8 \
  --wandb.enable=false \
  > $R/train.log 2>&1
CODE=$?
if [ $CODE -eq 0 ]; then mark "TRAIN_DONE"; else mark "TRAIN_FAIL code=$CODE"; fi
exit $CODE
