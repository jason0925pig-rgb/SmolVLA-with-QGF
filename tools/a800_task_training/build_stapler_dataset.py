#!/usr/bin/env python3
"""Build the standalone stapler-task dataset from source episodes 150-199.

No truncation (single-stage task recorded as such): rows and videos are kept
whole, only re-indexed 0-49. Fixed 45/5 split with seed 1000; global
normalization stats aggregated ONLY from the 45 train episodes, per the
2026-08-27 handoff's common rules. Originals are never modified. CPU only.

Run on a800new with ~/parcel_env/bin/python.
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path.home() / "stapler_src/lerobot_dataset"
RUN = Path.home() / "parcel_smolvla/20260827_stapler_into_box_50_45train_5val"
DST = RUN / "stapler_into_box_lerobot_v3"
TASK = "把订书机放进快递纸盒"
CAMS = ["observation.images.chest", "observation.images.wrist_right"]
OLD_EPS = list(range(150, 200))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    for sub in ["data/chunk-000", "meta/episodes/chunk-000",
                *(f"videos/{c}/chunk-000" for c in CAMS)]:
        (DST / sub).mkdir(parents=True, exist_ok=True)
    (RUN / "source_readonly_manifest").mkdir(parents=True, exist_ok=True)

    m0 = pd.read_parquet(SRC / "meta/episodes/chunk-000/file-150.parquet")
    OFFSET = int(m0["dataset_from_index"].iloc[0])
    print("index offset:", OFFSET)

    ep_stats, report, src_manifest = [], [], []
    total = 0
    for new_i, old_i in enumerate(OLD_EPS):
        dsrc = SRC / f"data/chunk-000/file-{old_i:03d}.parquet"
        d = pd.read_parquet(dsrc)
        assert int(d["episode_index"].iloc[0]) == old_i
        d["episode_index"] = new_i
        d["index"] = d["index"] - OFFSET
        d["task_index"] = 0
        st = np.stack(d["observation.state"].to_numpy())
        ac = np.stack(d["action"].to_numpy())
        assert st.shape[1] == 8 and ac.shape[1] == 8
        assert np.isfinite(st).all() and np.isfinite(ac).all()
        assert (np.diff(d["timestamp"].to_numpy()) > 0).all()
        d.to_parquet(DST / f"data/chunk-000/file-{new_i:03d}.parquet", index=False)

        e = pd.read_parquet(SRC / f"meta/episodes/chunk-000/file-{old_i:03d}.parquet").copy()
        t = list(e["tasks"].iloc[0])
        assert len(t) == 1 and t[0] == TASK, f"ep{old_i}: task {t}"
        n = int(e["length"].iloc[0])
        assert n == len(d)
        e["episode_index"] = new_i
        e["data/file_index"] = new_i
        e["dataset_from_index"] = e["dataset_from_index"] - OFFSET
        e["dataset_to_index"] = e["dataset_to_index"] - OFFSET
        for cam in CAMS:
            e[f"videos/{cam}/file_index"] = new_i
        if "meta/episodes/file_index" in e.columns:
            e["meta/episodes/file_index"] = new_i

        # per-episode stats: valid as-is (no truncation); fix bookkeeping feats
        stats = {}
        for col in e.columns:
            if not col.startswith("stats/"):
                continue
            _, feat, key = col.split("/", 2)
            stats.setdefault(feat, {})[key] = np.asarray(e[col].iloc[0])
        for feat, off in [("episode_index", old_i - new_i), ("index", OFFSET)]:
            for k in ["min", "max", "mean", "q01", "q10", "q50", "q90", "q99"]:
                if k in stats.get(feat, {}):
                    stats[feat][k] = stats[feat][k] - off
                    e[f"stats/{feat}/{k}"] = [stats[feat][k]]
        # task_index in the mixed library was 9 for the mug task; force 0
        for k in ["min", "max", "mean", "q01", "q10", "q50", "q90", "q99"]:
            if k in stats.get("task_index", {}):
                stats["task_index"][k] = np.zeros_like(stats["task_index"][k])
                e[f"stats/task_index/{k}"] = [stats["task_index"][k]]
        ep_stats.append(stats)
        e.to_parquet(DST / f"meta/episodes/chunk-000/file-{new_i:03d}.parquet", index=False)

        for cam in CAMS:
            shutil.copy2(SRC / f"videos/{cam}/chunk-000/file-{old_i:03d}.mp4",
                         DST / f"videos/{cam}/chunk-000/file-{new_i:03d}.mp4")

        src_manifest.append({"old_index": old_i,
                             "data_parquet_sha256": sha256_file(dsrc),
                             "chest_mp4_sha256": sha256_file(SRC / f"videos/{CAMS[0]}/chunk-000/file-{old_i:03d}.mp4"),
                             "wrist_mp4_sha256": sha256_file(SRC / f"videos/{CAMS[1]}/chunk-000/file-{old_i:03d}.mp4")})
        report.append({"old_index": old_i, "new_index": new_i, "length": n})
        total += n
        if new_i % 10 == 0:
            print(f"  ep {old_i}->{new_i} ({n} frames)", flush=True)

    # tasks table
    t_src = pd.read_parquet(SRC / "meta/tasks.parquet")
    if t_src.index.dtype == object:
        t_new = pd.DataFrame({"task_index": [0]}, index=pd.Index([TASK], name=t_src.index.name))
    else:
        t_new = pd.DataFrame({"task_index": [0], "task": [TASK]})
    t_new.to_parquet(DST / "meta/tasks.parquet")

    info = json.loads((SRC / "meta/info.json").read_text())
    info["total_episodes"] = 50
    info["total_frames"] = int(total)
    info["total_tasks"] = 1
    if "total_videos" in info:
        info["total_videos"] = 100
    if "splits" in info:
        info["splits"] = {"train": "0:50"}
    (DST / "meta/info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False))

    # 45/5 split, seed 1000
    rng = np.random.default_rng(1000)
    val = sorted(rng.choice(np.arange(50), size=5, replace=False).tolist())
    train = [i for i in range(50) if i not in val]
    (RUN / "split_manifest.json").write_text(json.dumps({
        "seed": 1000, "rule": "rng.choice over sorted new indices 0..49, size 5, no replacement",
        "validation_new_indices": val, "train_new_indices": train, "task_text": TASK,
        "mapping": [{"old": r["old_index"], "new": r["new_index"],
                     "split": "validation" if r["new_index"] in val else "train",
                     "length": r["length"]} for r in report]}, indent=2, ensure_ascii=False))

    # global stats from 45 train eps only
    agg = None
    for mp in ["lerobot.datasets.compute_stats", "lerobot.common.datasets.compute_stats"]:
        try:
            agg = __import__(mp, fromlist=["aggregate_stats"]).aggregate_stats
            break
        except Exception:
            continue
    assert agg is not None

    def f64(v):
        a = np.asarray(v)
        if a.dtype == object:
            a = np.stack([np.asarray(x, dtype=np.float64).ravel() for x in a]).ravel()
        return a.astype(np.float64)

    def prep(stats):
        out = {}
        for feat, fs in stats.items():
            o = {}
            for k, v in fs.items():
                a = f64(v)
                if feat.startswith("observation.images.") and k != "count" and a.shape == (3,):
                    a = a.reshape(3, 1, 1)
                o[k] = a
            out[feat] = o
        return out

    global_stats = agg([prep(ep_stats[i]) for i in train])

    def jsonable(o):
        if isinstance(o, dict):
            return {k: jsonable(v) for k, v in o.items()}
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    (DST / "meta/stats.json").write_text(json.dumps(jsonable(global_stats), indent=4))

    (RUN / "source_readonly_manifest/source_files_sha256.json").write_text(json.dumps({
        "source": "Orin onearm_Tele/lerobot_dataset eps 150-199, streamed via laptop 2026-08-27",
        "episodes": src_manifest}, indent=2, ensure_ascii=False))
    (RUN / "preprocessing_report.json").write_text(json.dumps({
        "run_id": RUN.name, "truncation": "none (single-stage task)",
        "stats_scope": "global stats from the 45 TRAIN episodes only",
        "episodes": report}, indent=2, ensure_ascii=False))

    print("\nMUG_BUILD_DONE frames:", total)
    print("validation (new idx):", val)


if __name__ == "__main__":
    main()
