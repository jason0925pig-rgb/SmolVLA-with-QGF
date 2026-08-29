"""Verify the cleaned red-parcel dataset per handoff 2.1 step 7."""
import glob
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path.home() / "parcel_smolvla/20260828_red_parcel_clean_50"
CLEAN = RUN / "clean_lerobot_v3"
FFMPEG = str(Path.home() / "bin/ffmpeg")
import cv2

files = sorted(glob.glob(str(CLEAN / "data/chunk-000/*.parquet")))
print("episodes:", len(files))

# ---- J5 branch check across the whole clean set
allj = []
for f in files:
    d = pd.read_parquet(f, columns=["observation.state", "action"])
    allj.append(np.stack(d["observation.state"].to_numpy())[:, 4])
j5 = np.concatenate(allj)
print(f"J5 state range: [{j5.min():+.3f}, {j5.max():+.3f}]  span={j5.max()-j5.min():.3f} rad")
print("  -> single branch" if (j5.max() - j5.min()) < 3.0 else "  -> STILL MIXED!")

# per-episode branch
branches = set()
for a in allj:
    branches.add("pos" if a.mean() > 0 else "neg")
print("  per-episode branches present:", branches)

# ---- all joints range
st_all, ac_all = [], []
for f in files:
    d = pd.read_parquet(f, columns=["observation.state", "action"])
    st_all.append(np.stack(d["observation.state"].to_numpy()))
    ac_all.append(np.stack(d["action"].to_numpy()))
st = np.concatenate(st_all); ac = np.concatenate(ac_all)
print("\nper-joint action range:")
for j in range(7):
    print(f"  J{j+1}: [{ac[:,j].min():+.3f}, {ac[:,j].max():+.3f}]")
print(f"  gripper: [{ac[:,7].min():.2f}, {ac[:,7].max():.2f}]")
print("finite:", bool(np.isfinite(st).all() and np.isfinite(ac).all()))

# ---- spot check 5 episodes: video duration vs parquet time span
print("\nspot check (video duration vs parquet span):")
rep = json.loads((RUN / "manifest/cleaning_report.json").read_text())
by_new = {e["new_index"]: e for e in rep["episodes"] if e.get("decision") == "keep"}
for new_i in [0, 12, 24, 36, 49]:
    d = pd.read_parquet(CLEAN / f"data/chunk-000/file-{new_i:03d}.parquet", columns=["timestamp"])
    span = float(d["timestamp"].iloc[-1] - d["timestamp"].iloc[0])
    rows = len(d)
    out = []
    for cam in ["observation.images.chest", "observation.images.wrist_right"]:
        p = CLEAN / f"videos/{cam}/chunk-000/file-{new_i:03d}.mp4"
        cap = cv2.VideoCapture(str(p))
        n = 0
        while cap.read()[0]:
            n += 1
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        out.append(f"{cam.split('.')[-1]}: {n}f @{fps:.1f}fps = {n/max(fps,1):.2f}s")
    e = by_new.get(new_i, {})
    print(f"  ep{new_i:>2} rows={rows} span={span:.2f}s t_cut={e.get('t_cut')} | " + " | ".join(out))

# ---- info / split consistency
info = json.loads((CLEAN / "meta/info.json").read_text())
split = json.loads((RUN / "split_manifest.json").read_text())
print("\ninfo total_episodes:", info["total_episodes"], "total_frames:", info["total_frames"])
print("split: train", len(split["train_new_indices"]), "val", split["validation_new_indices"])
print("kept/rejected:", rep["kept"], "/", rep["rejected"])
shifted = [e["old_index"] for e in rep["episodes"] if e.get("j5_shifted")]
print("J5-shifted episodes (old idx):", shifted)
