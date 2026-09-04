#!/usr/bin/env python3
"""Screwdriver-into-box cleaning, derived from the red/green parcel cleaners.

Three deliberate differences from those:

1. The cut is [0, t_open_last + 7 s], not a mid-recording window. This task
   starts from the operator's home pose, so the clock already begins at 0.

2. The tail anchor is the LAST confirmed release, not the first. Three episodes
   (215, 245, 249) drop the screwdriver and re-grab: ep215 holds 3.4 s then
   releases, ep245 releases twice before succeeding. Cutting at the first
   release would end those episodes at a drop and teach the policy that letting
   go early completes the task.

3. J5 is CHECKED but not shifted. These episodes were all recorded after the
   dataset's single branch switch, so all 50 already sit on the negative branch
   (per-episode mean spread 0.864 rad, largest gap 0.235 rad, nothing near
   2*pi). The script refuses to run if that stops being true, rather than
   silently applying a correction that is not needed here.

Reads the shared onearm_Tele dataset read-only and writes a derived dataset.
"""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SRC = Path("/home/nvidia/work/telop/onearm_Tele/lerobot_dataset")
RUN = Path("/home/nvidia/work/telop/screwdriver_clean_50_20260904")
CLEAN = RUN / "clean_lerobot_v3"
MAN = RUN / "manifest"
TASK = "把杯子里的螺丝刀放进纸盒里"
OLD_EPS = list(range(200, 250))          # 250 was deleted at the user's request
CAMS = ["observation.images.chest", "observation.images.wrist_right"]
OPEN_T, CLOSE_T, CONFIRM, TAIL_S = 0.15, 0.85, 5, 7.0
GRIP, J5, FPS = 7, 4, 30
TWO_PI = 2.0 * np.pi
QS = [("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99)]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def feat_stats(a, count_override=None):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    s = {"min": a.min(0), "max": a.max(0), "mean": a.mean(0), "std": a.std(0),
         "count": np.array([count_override if count_override else len(a)])}
    for n, q in QS:
        s[n] = np.quantile(a, q, axis=0)
    return s


def gripper_cycles(g15, ts15):
    """Every confirmed close->open pair on the 15 Hz axis."""
    out, run_c, run_o, state, t_close = [], 0, 0, "open", None
    for i, v in enumerate(g15):
        run_c = run_c + 1 if v >= CLOSE_T else 0
        run_o = run_o + 1 if v <= OPEN_T else 0
        if state == "open" and run_c >= CONFIRM:
            state, t_close = "closed", float(ts15[i])
        elif state == "closed" and run_o >= CONFIRM:
            out.append((t_close, float(ts15[i])))
            state, t_close = "open", None
    if state == "closed":
        out.append((t_close, None))
    return out


def cut_video(src, dst, n):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                    "-frames:v", str(n), "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "18", "-g", "2",
                    "-pix_fmt", "yuv420p", "-r", str(FPS), "-an", str(dst)], check=True)
    cap = cv2.VideoCapture(str(dst))
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    ssum = np.zeros(3)
    ssq = np.zeros(3)
    npx = 0
    samples = []
    got = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        got += 1
        small = cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA)
        rgb = small[:, :, ::-1].astype(np.float64) / 255.0
        flat = rgb.reshape(-1, 3)
        mins = np.minimum(mins, flat.min(0))
        maxs = np.maximum(maxs, flat.max(0))
        ssum += flat.sum(0)
        ssq += (flat ** 2).sum(0)
        npx += len(flat)
        if got % 5 == 1:
            samples.append(flat[::13])
    cap.release()
    if got != n:
        raise RuntimeError(dst.name + ": decoded " + str(got) + " != " + str(n))
    mean = ssum / npx
    std = np.sqrt(np.maximum(0.0, ssq / npx - mean ** 2))
    samp = np.concatenate(samples, 0)
    s = {"min": mins, "max": maxs, "mean": mean, "std": std, "count": np.array([npx])}
    for name, q in QS:
        s[name] = np.quantile(samp, q, axis=0)
    return s, got


def main():
    assert SRC.is_dir(), "source missing: " + str(SRC)
    if CLEAN.exists():
        shutil.rmtree(CLEAN)
    subs = ["data/chunk-000", "meta/episodes/chunk-000"]
    subs += ["videos/" + c + "/chunk-000" for c in CAMS]
    for sub in subs:
        (CLEAN / sub).mkdir(parents=True, exist_ok=True)
    MAN.mkdir(parents=True, exist_ok=True)

    # --- J5 gate: verify the branch problem is absent rather than assume it ---
    means = []
    for ep in OLD_EPS:
        d = pd.read_parquet(SRC / ("data/chunk-000/file-%03d.parquet" % ep),
                            columns=["observation.state"])
        means.append(float(np.stack(d["observation.state"].to_numpy())[:, J5].mean()))
    means = np.asarray(means)
    spread = float(means.max() - means.min())
    gap = float(np.diff(np.sort(means)).max())
    n_pos = int((means > 0).sum())
    print("J5 check: per-episode mean spread %.3f rad, largest gap %.3f rad, pos=%d neg=%d"
          % (spread, gap, n_pos, len(means) - n_pos))
    assert spread < 1.5 and gap < 1.0 and n_pos in (0, len(means)), (
        "J5 sits on two branches here; this task's cleaner does not shift it. "
        "spread=%.3f gap=%.3f pos=%d" % (spread, gap, n_pos))
    print("  single branch confirmed, no 2*pi alignment applied")

    decisions, kept, ep_stats = [], [], []
    running, new_i = 0, 0
    for ep in OLD_EPS:
        rec = {"old_index": ep}
        d = pd.read_parquet(SRC / ("data/chunk-000/file-%03d.parquet" % ep))
        st = np.stack(d["observation.state"].to_numpy())
        ac = np.stack(d["action"].to_numpy())
        ts = d["timestamp"].to_numpy().astype(float)
        if st.shape[1] != 8 or ac.shape[1] != 8:
            rec.update(decision="reject", reason="dims")
            decisions.append(rec)
            continue
        if not (np.isfinite(st).all() and np.isfinite(ac).all()):
            rec.update(decision="reject", reason="NaN/Inf")
            decisions.append(rec)
            continue
        if not (np.diff(ts) > 0).all():
            rec.update(decision="reject", reason="non-monotonic timestamps")
            decisions.append(rec)
            continue

        cy = gripper_cycles(ac[::2, GRIP], ts[::2])
        rec["gripper_cycles"] = len(cy)
        if not cy:
            rec.update(decision="reject", reason="no confirmed grasp")
            decisions.append(rec)
            continue
        if cy[-1][1] is None:
            rec.update(decision="reject", reason="never released")
            decisions.append(rec)
            continue
        t_close, t_open = cy[-1]
        rec.update(t_close_last=round(t_close, 3), t_open_last=round(t_open, 3),
                   retries=len(cy) - 1)

        t_target = t_open + TAIL_S
        k = int(np.searchsorted(ts, t_target + 1e-9) - 1)
        n = min(k + 1, len(ts))
        t_cut = float(ts[n - 1])
        rec.update(t_cut=round(t_cut, 3), orig_seconds=round(float(ts[-1]), 3),
                   clean_seconds=round(t_cut, 3), orig_actions=len(d), clean_actions=n)

        dd = d.iloc[:n].copy()
        dd["episode_index"] = new_i
        dd["index"] = np.arange(running, running + n, dtype=np.int64)
        dd["task_index"] = 0
        dd["frame_index"] = np.arange(n, dtype=np.int64)
        dd.to_parquet(CLEAN / ("data/chunk-000/file-%03d.parquet" % new_i), index=False)

        stats = {"observation.state": feat_stats(np.stack(dd["observation.state"].to_numpy())),
                 "action": feat_stats(np.stack(dd["action"].to_numpy()))}
        for col in ["observation.gripper_contact", "observation.gripper_feedback_valid",
                    "timestamp", "frame_index", "episode_index", "index", "task_index"]:
            if col in dd.columns:
                stats[col] = feat_stats(dd[col].to_numpy().astype(np.float64))

        vframes = {}
        for cam in CAMS:
            s, got = cut_video(SRC / ("videos/" + cam + "/chunk-000/file-%03d.mp4" % ep),
                               CLEAN / ("videos/" + cam + "/chunk-000/file-%03d.mp4" % new_i), n)
            stats[cam] = s
            vframes[cam] = got
        rec["video_frames"] = vframes
        rec["video_fps"] = FPS

        e = pd.read_parquet(SRC / ("meta/episodes/chunk-000/file-%03d.parquet" % ep)).copy()
        e["tasks"] = [[TASK]]
        e["episode_index"] = new_i
        e["length"] = n
        e["data/file_index"] = new_i
        e["dataset_from_index"] = running
        e["dataset_to_index"] = running + n
        for cam in CAMS:
            e["videos/" + cam + "/file_index"] = new_i
            e["videos/" + cam + "/from_timestamp"] = 0.0
            e["videos/" + cam + "/to_timestamp"] = round(n / FPS, 6)
        if "meta/episodes/file_index" in e.columns:
            e["meta/episodes/file_index"] = new_i
        for feat, fs in stats.items():
            for key, val in fs.items():
                col = "stats/" + feat + "/" + key
                if col in e.columns:
                    e[col] = [np.asarray(val, dtype=np.float64)]
        e.to_parquet(CLEAN / ("meta/episodes/chunk-000/file-%03d.parquet" % new_i), index=False)

        rec.update(decision="keep", new_index=new_i,
                   sha256_parquet=sha256_file(CLEAN / ("data/chunk-000/file-%03d.parquet" % new_i)))
        decisions.append(rec)
        ep_stats.append(stats)
        kept.append(new_i)
        running += n
        new_i += 1
        if new_i % 10 == 1:
            print("  ep%d -> %d: rows %d->%d, cycles=%d, t_cut=%.1fs"
                  % (ep, new_i - 1, len(d), n, len(cy), t_cut), flush=True)

    n_keep = len(kept)
    print("\nkept %d / %d" % (n_keep, len(OLD_EPS)))

    t_src = pd.read_parquet(SRC / "meta/tasks.parquet")
    if t_src.index.dtype == object:
        pd.DataFrame({"task_index": [0]},
                     index=pd.Index([TASK], name=t_src.index.name)
                     ).to_parquet(CLEAN / "meta/tasks.parquet")
    else:
        pd.DataFrame({"task_index": [0], "task": [TASK]}).to_parquet(CLEAN / "meta/tasks.parquet")

    info = json.loads((SRC / "meta/info.json").read_text())
    info["total_episodes"] = n_keep
    info["total_frames"] = int(running)
    info["total_tasks"] = 1
    if "total_videos" in info:
        info["total_videos"] = 2 * n_keep
    if "splits" in info:
        info["splits"] = {"train": "0:%d" % n_keep}
    (CLEAN / "meta/info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False))

    rng = np.random.default_rng(1000)
    n_val = 5 if n_keep >= 50 else max(1, n_keep // 10)
    val = sorted(rng.choice(np.arange(n_keep), size=n_val, replace=False).tolist())
    train = [i for i in range(n_keep) if i not in val]
    (RUN / "split_manifest.json").write_text(json.dumps(
        {"seed": 1000, "kept_episodes": n_keep, "task_text": TASK,
         "val": val, "train": train,
         "validation_new_indices": val, "train_new_indices": train,
         "note": "episode-level split; stats/training use train only"},
        indent=1, ensure_ascii=False))

    # global stats from TRAIN only
    agg = None
    for mp in ["lerobot.datasets.compute_stats", "lerobot.common.datasets.compute_stats"]:
        try:
            agg = __import__(mp, fromlist=["aggregate_stats"]).aggregate_stats
            break
        except Exception:
            continue
    if agg is None:
        print("WARNING: aggregate_stats not importable here; meta/stats.json NOT written")
    else:
        def f64(v):
            a = np.asarray(v)
            if a.dtype == object:
                a = np.stack([np.asarray(x, dtype=np.float64).ravel() for x in a]).ravel()
            return a.astype(np.float64)

        def prep(s):
            out = {}
            for feat, fs in s.items():
                o = {}
                for k, v in fs.items():
                    a = f64(v)
                    if feat.startswith("observation.images.") and k != "count" and a.shape == (3,):
                        a = a.reshape(3, 1, 1)
                    o[k] = a
                out[feat] = o
            return out

        gstats = agg([prep(ep_stats[i]) for i in train])

        def js(o):
            if isinstance(o, dict):
                return {k: js(v) for k, v in o.items()}
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            return o
        (CLEAN / "meta/stats.json").write_text(json.dumps(js(gstats), indent=4))
        print("wrote meta/stats.json from %d train episodes" % len(train))

    pd.DataFrame(decisions).to_csv(MAN / "episode_decisions.csv", index=False)
    (MAN / "cleaning_report.json").write_text(json.dumps(
        {"run_id": "screwdriver_clean_50_20260904",
         "task_text": TASK,
         "source": str(SRC),
         "source_episodes": "200-249 (250 deleted at the user's request before cleaning)",
         "cut_rule": "[0, last confirmed release + 7.0 s]; the LAST release is used "
                     "because 3 episodes re-grasp after dropping and the first "
                     "release there is a drop, not the placement",
         "j5_policy": "verified single branch (spread %.3f rad, gap %.3f rad); "
                      "no 2*pi alignment applied" % (spread, gap),
         "kept": n_keep, "rejected": len(OLD_EPS) - n_keep,
         "total_frames": int(running),
         "episodes": decisions},
        indent=1, ensure_ascii=False, default=str))
    print("wrote manifest + split")


if __name__ == "__main__":
    main()
