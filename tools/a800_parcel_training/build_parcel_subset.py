#!/usr/bin/env python3
"""Extract episodes 50-99 (parcel task) from the mixed 100-episode LeRobot v3
dataset into a fresh standalone dataset re-indexed 0-49.

Normalization stats are aggregated ONLY from these 50 episodes' per-episode
stats (already stored in meta/episodes parquet) using lerobot's official
aggregate_stats, so no video decoding is required.

Run on the A800 with the lerobot==0.4.4 venv.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path.home() / "parcel_smolvla/20260825_parcel_50/source_full/lerobot_dataset"
DST = Path.home() / "parcel_smolvla/20260825_parcel_50/dataset"
MANIFEST_DIR = Path.home() / "parcel_smolvla/20260825_parcel_50/manifest"

OLD_EPS = list(range(50, 100))          # source episode indices
INDEX_OFFSET = None                      # derived from ep50's dataset_from_index
TASK_TEXT_EXPect = "红色包裹"            # sanity substring

STAT_KEYS = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]


def load_episode_meta(ep: int) -> pd.DataFrame:
    return pd.read_parquet(SRC / f"meta/episodes/chunk-000/file-{ep:03d}.parquet")


def main() -> None:
    global INDEX_OFFSET
    assert SRC.is_dir(), f"source missing: {SRC}"
    if DST.exists():
        print("dst exists, removing for a clean rebuild")
        shutil.rmtree(DST)
    for sub in ["data/chunk-000", "meta/episodes/chunk-000",
                "videos/observation.images.chest/chunk-000",
                "videos/observation.images.wrist_right/chunk-000"]:
        (DST / sub).mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    # -- derive frame index offset from the first parcel episode
    m0 = load_episode_meta(50)
    INDEX_OFFSET = int(m0["dataset_from_index"].iloc[0])
    print("index offset:", INDEX_OFFSET)

    manifest = {"source_root": str(SRC), "old_episode_indices": OLD_EPS,
                "index_offset": INDEX_OFFSET, "episodes": []}
    ep_stats_list = []
    total_frames = 0
    task_text = None

    for new_i, old_i in enumerate(OLD_EPS):
        # ---- data parquet: rewrite bookkeeping columns
        d = pd.read_parquet(SRC / f"data/chunk-000/file-{old_i:03d}.parquet")
        assert int(d["episode_index"].iloc[0]) == old_i
        d["episode_index"] = new_i
        d["index"] = d["index"] - INDEX_OFFSET
        d["task_index"] = 0
        # basic integrity
        st = np.stack(d["observation.state"].to_numpy())
        ac = np.stack(d["action"].to_numpy())
        assert st.shape[1] == 8 and ac.shape[1] == 8, f"ep{old_i}: dims {st.shape} {ac.shape}"
        assert np.isfinite(st).all() and np.isfinite(ac).all(), f"ep{old_i}: NaN/Inf!"
        ts = d["timestamp"].to_numpy()
        assert (np.diff(ts) > 0).all(), f"ep{old_i}: non-monotonic timestamps"
        d.to_parquet(DST / f"data/chunk-000/file-{new_i:03d}.parquet", index=False)

        # ---- episode meta parquet: rewrite indices, keep stats columns
        e = load_episode_meta(old_i).copy()
        this_task = list(e["tasks"].iloc[0])
        assert len(this_task) == 1 and TASK_TEXT_EXPect in this_task[0], f"ep{old_i}: task {this_task}"
        if task_text is None:
            task_text = this_task[0]
        assert this_task[0] == task_text, f"ep{old_i}: inconsistent task text"
        length = int(e["length"].iloc[0])
        e["episode_index"] = new_i
        e["data/file_index"] = new_i
        e["dataset_from_index"] = e["dataset_from_index"] - INDEX_OFFSET
        e["dataset_to_index"] = e["dataset_to_index"] - INDEX_OFFSET
        for cam in ["observation.images.chest", "observation.images.wrist_right"]:
            e[f"videos/{cam}/file_index"] = new_i
        if "meta/episodes/file_index" in e.columns:
            e["meta/episodes/file_index"] = new_i
        # per-episode stats for global aggregation (nested dict form)
        stats = {}
        for col in e.columns:
            if not col.startswith("stats/"):
                continue
            _, feat, key = col.split("/", 2)
            stats.setdefault(feat, {})[key] = np.asarray(e[col].iloc[0])
        # image stats are stored flattened as (3,) in the parquet, but
        # lerobot's aggregate_stats validates image stats as (3,1,1)
        for feat, fstats in stats.items():
            if not feat.startswith("observation.images."):
                continue
            for k, v in fstats.items():
                if k != "count" and v.shape == (3,):
                    fstats[k] = v.reshape(3, 1, 1)
        # episode_index/index stats inside stats refer to OLD numbering; fix them
        for feat, off in [("episode_index", old_i - new_i), ("index", INDEX_OFFSET)]:
            if feat in stats:
                for k in ["min", "max", "mean", "q01", "q10", "q50", "q90", "q99"]:
                    if k in stats[feat]:
                        stats[feat][k] = stats[feat][k] - off
                        e[f"stats/{feat}/{k}"] = [stats[feat][k]]
        if "task_index" in stats:
            for k in ["min", "max", "mean", "q01", "q10", "q50", "q90", "q99"]:
                if k in stats["task_index"]:
                    stats["task_index"][k] = np.zeros_like(stats["task_index"][k])
                    e[f"stats/task_index/{k}"] = [stats["task_index"][k]]
        ep_stats_list.append(stats)
        e.to_parquet(DST / f"meta/episodes/chunk-000/file-{new_i:03d}.parquet", index=False)

        # ---- videos: plain copy under the new name
        for cam in ["observation.images.chest", "observation.images.wrist_right"]:
            src_v = SRC / f"videos/{cam}/chunk-000/file-{old_i:03d}.mp4"
            dst_v = DST / f"videos/{cam}/chunk-000/file-{new_i:03d}.mp4"
            assert src_v.is_file(), f"missing video {src_v}"
            shutil.copy2(src_v, dst_v)

        total_frames += length
        manifest["episodes"].append({"old_index": old_i, "new_index": new_i,
                                     "length": length, "task": this_task[0]})
        if new_i % 10 == 0:
            print(f"  ep {old_i} -> {new_i} done ({length} frames)")

    # ---- tasks.parquet: single task, index 0  (match source schema)
    t_src = pd.read_parquet(SRC / "meta/tasks.parquet")
    if t_src.index.name is not None or t_src.index.dtype == object:
        # source uses task string as index with a task_index column
        t_new = pd.DataFrame({"task_index": [0]}, index=pd.Index([task_text], name=t_src.index.name))
    else:
        t_new = pd.DataFrame({"task_index": [0], "task": [task_text]})
    t_new.to_parquet(DST / "meta/tasks.parquet")

    # ---- info.json: fix counts, keep everything else verbatim
    info = json.loads((SRC / "meta/info.json").read_text())
    info["total_episodes"] = 50
    info["total_frames"] = int(total_frames)
    info["total_tasks"] = 1
    if "total_videos" in info:
        info["total_videos"] = 100
    if "splits" in info:
        info["splits"] = {"train": "0:50"}
    (DST / "meta/info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False))

    # ---- global stats.json: official aggregation over the 50 episodes only
    agg = None
    for modpath in ["lerobot.datasets.compute_stats",
                    "lerobot.common.datasets.compute_stats"]:
        try:
            mod = __import__(modpath, fromlist=["aggregate_stats"])
            agg = mod.aggregate_stats
            break
        except Exception:
            continue
    assert agg is not None, "aggregate_stats not found in lerobot"
    import torch

    def _f64(v):
        a = np.asarray(v)
        if a.dtype == object:
            a = np.stack([np.asarray(x, dtype=np.float64) for x in a])
        return a.astype(np.float64)

    ep_stats_np = [
        {f: {k: _f64(v) for k, v in fs.items()} for f, fs in ep.items()}
        for ep in ep_stats_list
    ]
    global_stats = agg(ep_stats_np)

    def to_jsonable(o):
        if isinstance(o, dict):
            return {k: to_jsonable(v) for k, v in o.items()}
        if isinstance(o, np.ndarray):
            return o.tolist()
        try:
            import torch as _t
            if isinstance(o, _t.Tensor):
                return o.cpu().numpy().tolist()
        except Exception:
            pass
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    (DST / "meta/stats.json").write_text(json.dumps(to_jsonable(global_stats), indent=4))

    manifest["task_text"] = task_text
    manifest["total_frames"] = int(total_frames)
    (MANIFEST_DIR / "MANIFEST_NEW_TASK_50_20260825.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))

    print("\nDONE")
    print("episodes: 50, frames:", total_frames)
    print("task:", task_text)
    print("dst:", DST)


if __name__ == "__main__":
    sys.exit(main())
