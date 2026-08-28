#!/usr/bin/env python3
"""Rebuild the 50 dual-parcel demos into the single-stage red-parcel dataset
per docs/SMOLVLA_NEW_50_A800_TRAINING_AND_ORIN_HANDOFF.md (2026-08-27).

Truncation: first confirmed close (g>=0.85, 5 ticks @15Hz) -> first confirmed
open after it (g<=0.40, 5 ticks) -> t_end = t_open + 7.0 s.
Keeps everything from episode start to t_end. Originals are never modified.

CPU only (no CUDA). Run with the parcel_env python on the A800.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SRC = Path.home() / "parcel_smolvla/20260825_parcel_50/source_full/lerobot_dataset"
RUN = Path.home() / "parcel_smolvla/20260827_red_parcel_out_table_50_trunc7s_45train_5val"
DST = RUN / "red_parcel_out_table_lerobot_v3"
NEW_TASK = "把箱子里的红色包裹拿出来放到桌子上。"
CLOSE_T, OPEN_T, CONFIRM, TAIL_S = 0.85, 0.40, 5, 7.0
CAMS = ["observation.images.chest", "observation.images.wrist_right"]
NUM_FEATS = ["observation.state", "action", "observation.gripper_contact",
             "observation.gripper_feedback_valid", "timestamp", "frame_index",
             "episode_index", "index", "task_index"]
QS = [("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99)]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def detect(g30, ts30):
    g, ts = g30[::2], ts30[::2]
    def first_run(mask, start=0):
        run = 0
        for i in range(start, len(mask)):
            run = run + 1 if mask[i] else 0
            if run >= CONFIRM:
                return i
        return None
    ic = first_run(g >= CLOSE_T)
    assert ic is not None
    io = first_run(g <= OPEN_T, start=ic + 1)
    assert io is not None
    return float(ts[ic]), float(ts[io])


def feat_stats(arr, count_override=None):
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    s = {"min": a.min(0), "max": a.max(0), "mean": a.mean(0),
         "std": a.std(0), "count": np.array([count_override or len(a)])}
    for name, q in QS:
        s[name] = np.quantile(a, q, axis=0)
    return s


def video_cut_and_stats(src_mp4: Path, dst_mp4: Path, n_frames: int):
    """Re-encode exactly n_frames, then decode the RESULT for image stats +
    integrity (decodes to the last frame or we fail loudly)."""
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src_mp4),
         "-frames:v", str(n_frames), "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30", "-an", str(dst_mp4)],
        check=True)
    cap = cv2.VideoCapture(str(dst_mp4))
    mins = np.full(3, np.inf); maxs = np.full(3, -np.inf)
    ssum = np.zeros(3); ssq = np.zeros(3); npx = 0
    samples = []
    got = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        got += 1
        small = cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA)
        rgb = small[:, :, ::-1].astype(np.float64) / 255.0   # BGR -> RGB, [0,1]
        flat = rgb.reshape(-1, 3)
        mins = np.minimum(mins, flat.min(0)); maxs = np.maximum(maxs, flat.max(0))
        ssum += flat.sum(0); ssq += (flat ** 2).sum(0); npx += len(flat)
        if got % 5 == 1:                       # subsample frames for quantiles
            samples.append(flat[::13])         # and pixels within the frame
    cap.release()
    if got != n_frames:
        raise RuntimeError(f"{dst_mp4.name}: decoded {got} != expected {n_frames}")
    mean = ssum / npx
    std = np.sqrt(np.maximum(0.0, ssq / npx - mean ** 2))
    samp = np.concatenate(samples, 0)
    s = {"min": mins, "max": maxs, "mean": mean, "std": std,
         "count": np.array([npx])}
    for name, q in QS:
        s[name] = np.quantile(samp, q, axis=0)
    return s


def review_image(dst_mp4: Path, t_close, t_open, t_end, out_jpg: Path):
    cap = cv2.VideoCapture(str(dst_mp4))
    tiles = []
    for label, t in [("close", t_close), ("open", t_open), ("end", t_end)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t * 30)) - 1))
        ok, img = cap.read()
        if not ok:
            img = np.zeros((720, 1280, 3), np.uint8)
        img = cv2.resize(img, (426, 240))
        cv2.putText(img, f"{label} t={t:.2f}s", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        tiles.append(img)
    cap.release()
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_jpg), np.concatenate(tiles, axis=1),
                [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    for sub in ["data/chunk-000", "meta/episodes/chunk-000",
                *(f"videos/{c}/chunk-000" for c in CAMS)]:
        (DST / sub).mkdir(parents=True, exist_ok=True)
    for sub in ["source_readonly_manifest", "validation_reports", "review"]:
        (RUN / sub).mkdir(parents=True, exist_ok=True)

    src_manifest, report_rows, ep_stats_all = [], [], []
    running_index = 0
    old_task_text = None

    for new_i, old_i in enumerate(range(50, 100)):
        dsrc = SRC / f"data/chunk-000/file-{old_i:03d}.parquet"
        d = pd.read_parquet(dsrc)
        act = np.stack(d["action"].to_numpy())
        st = np.stack(d["observation.state"].to_numpy())
        ts = d["timestamp"].to_numpy().astype(float)
        t_close, t_open = detect(act[:, 7], ts)
        t_end_target = t_open + TAIL_S
        k = int(np.searchsorted(ts, t_end_target + 1e-9) - 1)  # last row <= target
        t_end = float(ts[k])
        n = k + 1

        dd = d.iloc[:n].copy()
        dd["episode_index"] = new_i
        dd["index"] = np.arange(running_index, running_index + n, dtype=np.int64)
        dd["task_index"] = 0
        assert np.isfinite(np.stack(dd["action"].to_numpy())).all()
        assert np.isfinite(np.stack(dd["observation.state"].to_numpy())).all()
        dd.to_parquet(DST / f"data/chunk-000/file-{new_i:03d}.parquet", index=False)

        # per-episode numeric stats from the truncated rows
        stats = {}
        stats["observation.state"] = feat_stats(np.stack(dd["observation.state"].to_numpy()))
        stats["action"] = feat_stats(np.stack(dd["action"].to_numpy()))
        for col in ["observation.gripper_contact", "observation.gripper_feedback_valid",
                    "timestamp", "frame_index", "episode_index", "index", "task_index"]:
            stats[col] = feat_stats(dd[col].to_numpy().astype(np.float64))

        # videos: exact-frame cut + image stats from the cut result
        for cam in CAMS:
            vsrc = SRC / f"videos/{cam}/chunk-000/file-{old_i:03d}.mp4"
            vdst = DST / f"videos/{cam}/chunk-000/file-{new_i:03d}.mp4"
            stats[cam] = video_cut_and_stats(vsrc, vdst, n)

        review_image(DST / f"videos/{CAMS[0]}/chunk-000/file-{new_i:03d}.mp4",
                     t_close, t_open, t_end, RUN / f"review/ep{new_i:03d}_src{old_i}.jpg")

        # episodes meta: copy source row, rewrite bookkeeping + all stats columns
        e = pd.read_parquet(SRC / f"meta/episodes/chunk-000/file-{old_i:03d}.parquet").copy()
        if old_task_text is None:
            old_task_text = list(e["tasks"].iloc[0])[0]
        e["tasks"] = [[NEW_TASK]]
        e["episode_index"] = new_i
        e["length"] = n
        e["data/file_index"] = new_i
        e["dataset_from_index"] = running_index
        e["dataset_to_index"] = running_index + n
        for cam in CAMS:
            e[f"videos/{cam}/file_index"] = new_i
            e[f"videos/{cam}/from_timestamp"] = 0.0
            e[f"videos/{cam}/to_timestamp"] = round(n / 30.0, 6)
        if "meta/episodes/file_index" in e.columns:
            e["meta/episodes/file_index"] = new_i
        for feat, fs in stats.items():
            for key, val in fs.items():
                col = f"stats/{feat}/{key}"
                if col in e.columns:
                    e[col] = [np.asarray(val, dtype=np.float64)]
        e.to_parquet(DST / f"meta/episodes/chunk-000/file-{new_i:03d}.parquet", index=False)

        ep_stats_all.append(stats)
        report_rows.append({
            "old_index": old_i, "new_index": new_i,
            "t_close": round(t_close, 3), "t_open": round(t_open, 3),
            "t_end": round(t_end, 3), "orig_len": len(d), "new_len": n,
            "orig_seconds": round(float(ts[-1]), 2), "kept_seconds": round(t_end, 2)})
        src_manifest.append({
            "old_index": old_i,
            "data_parquet_sha256": sha256_file(dsrc),
            "chest_mp4_sha256": sha256_file(SRC / f"videos/{CAMS[0]}/chunk-000/file-{old_i:03d}.mp4"),
            "wrist_mp4_sha256": sha256_file(SRC / f"videos/{CAMS[1]}/chunk-000/file-{old_i:03d}.mp4")})
        running_index += n
        print(f"ep {old_i}->{new_i}: rows {len(d)}->{n}, t_end={t_end:.2f}s", flush=True)

    # tasks table (single new task, index 0)
    t_src = pd.read_parquet(SRC / "meta/tasks.parquet")
    if t_src.index.dtype == object:
        t_new = pd.DataFrame({"task_index": [0]},
                             index=pd.Index([NEW_TASK], name=t_src.index.name))
    else:
        t_new = pd.DataFrame({"task_index": [0], "task": [NEW_TASK]})
    t_new.to_parquet(DST / "meta/tasks.parquet")

    # info.json
    info = json.loads((SRC / "meta/info.json").read_text())
    info["total_episodes"] = 50
    info["total_frames"] = int(running_index)
    info["total_tasks"] = 1
    if "total_videos" in info:
        info["total_videos"] = 100
    if "splits" in info:
        info["splits"] = {"train": "0:50"}
    (DST / "meta/info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False))

    # fixed 45/5 split, seed 1000, drawn from sorted new indices
    rng = np.random.default_rng(1000)
    val = sorted(rng.choice(np.arange(50), size=5, replace=False).tolist())
    train = [i for i in range(50) if i not in val]
    split = {"seed": 1000, "rule": "rng.choice over sorted new indices 0..49, size 5, no replacement",
             "validation_new_indices": val, "train_new_indices": train,
             "task_text": NEW_TASK,
             "mapping": [{"old": r["old_index"], "new": r["new_index"],
                          "split": ("validation" if r["new_index"] in val else "train"),
                          "t_close": r["t_close"], "t_open": r["t_open"], "t_end": r["t_end"]}
                         for r in report_rows]}
    (RUN / "split_manifest.json").write_text(json.dumps(split, indent=2, ensure_ascii=False))

    # global stats from the 45 TRAIN episodes only (official aggregation)
    agg = None
    for modpath in ["lerobot.datasets.compute_stats", "lerobot.common.datasets.compute_stats"]:
        try:
            agg = __import__(modpath, fromlist=["aggregate_stats"]).aggregate_stats
            break
        except Exception:
            continue
    assert agg is not None

    def prep(stats):
        out = {}
        for feat, fs in stats.items():
            o = {}
            for k, v in fs.items():
                a = np.asarray(v, dtype=np.float64)
                if feat.startswith("observation.images.") and k != "count" and a.shape == (3,):
                    a = a.reshape(3, 1, 1)
                o[k] = a
            out[feat] = o
        return out

    train_stats = [prep(ep_stats_all[i]) for i in train]
    global_stats = agg(train_stats)

    def jsonable(o):
        if isinstance(o, dict):
            return {k: jsonable(v) for k, v in o.items()}
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    (DST / "meta/stats.json").write_text(json.dumps(jsonable(global_stats), indent=4))

    (RUN / "preprocessing_report.json").write_text(json.dumps({
        "run_id": RUN.name, "rule": "close>=0.85 & open<=0.40, 5 ticks @15Hz; t_end=t_open+7.0s",
        "frame_exact_cut": "row count k+1; video cut with ffmpeg -frames:v",
        "stats_scope": "global stats aggregated from the 45 TRAIN episodes only",
        "episodes": report_rows}, indent=2, ensure_ascii=False))
    (RUN / "source_readonly_manifest/source_files_sha256.json").write_text(json.dumps({
        "source_root": str(SRC), "original_task_text": old_task_text,
        "transfer_provenance": "404 files / 5905315566 bytes verified identical Orin vs A800 on 2026-08-25",
        "episodes": src_manifest}, indent=2, ensure_ascii=False))

    print("\nREBUILD_DONE")
    print("frames total:", running_index)
    print("validation episodes (new idx):", val)


if __name__ == "__main__":
    main()
