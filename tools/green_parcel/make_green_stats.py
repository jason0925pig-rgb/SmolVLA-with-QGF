#!/usr/bin/env python3
"""Write meta/stats.json for the cleaned green dataset.

lerobot's make_dataset() reads meta/stats.json for normalisation; without it
meta.stats[key] is None and training dies in factory.py. The red cleaner built
this at the end of its run and mine did not, so this recovers it from the
per-episode stats already stored in meta/episodes/*.parquet - no need to redo
the video re-encode.

Aggregated over the TRAIN episodes only, matching the red run: the 5 validation
episodes must not reach the normalisation statistics.
"""
import glob
import json

import numpy as np
import pandas as pd

RUN = "/ssd/zwwl_user2/parcel_smolvla/20260903_green_parcel_clean_50"
CLEAN = RUN + "/clean_lerobot_v3"

agg = None
for mp in ["lerobot.datasets.compute_stats", "lerobot.common.datasets.compute_stats"]:
    try:
        agg = __import__(mp, fromlist=["aggregate_stats"]).aggregate_stats
        break
    except Exception:
        continue
assert agg is not None, "aggregate_stats not importable"

split = json.load(open(RUN + "/split_manifest.json"))
train = split["train"]
val = split["val"]
assert not (set(train) & set(val)), "train/val overlap"
print("train %d / val %d  (stats use train only)" % (len(train), len(val)))

info = json.load(open(CLEAN + "/meta/info.json"))
feats = list(info["features"].keys())


def f64(v):
    a = np.asarray(v)
    if a.dtype == object:
        a = np.stack([np.asarray(x, dtype=np.float64).ravel() for x in a]).ravel()
    return a.astype(np.float64)


per_ep = []
for i in train:
    e = pd.read_parquet(CLEAN + "/meta/episodes/chunk-000/file-%03d.parquet" % i)
    out = {}
    for feat in feats:
        keys = [c for c in e.columns if c.startswith("stats/" + feat + "/")]
        if not keys:
            continue
        o = {}
        for c in keys:
            k = c.split("/")[-1]
            a = f64(e[c].iloc[0])
            if feat.startswith("observation.images.") and k != "count" and a.shape == (3,):
                a = a.reshape(3, 1, 1)
            o[k] = a
        out[feat] = o
    per_ep.append(out)

missing = [f for f in feats if f not in per_ep[0]]
assert not missing, "features without stats: %s" % missing
print("aggregating %d features over %d episodes" % (len(per_ep[0]), len(per_ep)))

gstats = agg(per_ep)


def js(o):
    if isinstance(o, dict):
        return {k: js(v) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


path = CLEAN + "/meta/stats.json"
open(path, "w").write(json.dumps(js(gstats), indent=4))
print("wrote", path)

chk = json.load(open(path))
print("features in stats.json:", len(chk))
for f in ("observation.state", "action", "observation.images.chest"):
    if f in chk:
        m = np.asarray(chk[f]["mean"])
        s = np.asarray(chk[f]["std"])
        print("  %-28s mean shape %-10s std finite %s" % (f, str(m.shape), bool(np.isfinite(s).all())))
bad = [f for f in chk if not np.isfinite(np.asarray(chk[f]["mean"], dtype=float)).all()]
print("non-finite means:", bad if bad else "none")
