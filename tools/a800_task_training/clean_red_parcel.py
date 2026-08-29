#!/usr/bin/env python3
"""Data cleaning for the red-parcel retrain, per handoff section 2.1
(2026-08-28 revision). Reads the RAW Orin export copy, writes a derived
clean dataset; the raw copy is never modified.

Per-episode pipeline (each step recorded in the manifest):
  1. 15 Hz policy axis from the action timestamps; video stays 30 FPS.
  2. state/action are 8-D, timestamps strictly increasing, finite.
  3. Gripper events from the RAW teach action: open<=0.15, close>=0.85,
     hold otherwise, 5 consecutive 15 Hz actions to confirm a switch.
  4. t_close = first confirmed close; t_open = first confirmed open after it.
     Missing / out-of-order / excessive pre-grasp flips -> reject.
  5. t_cut = t_open + 7.0 s (or episode end if shorter). Video is cut by real
     presentation timestamp, never by changing FPS.
  6. Rebuild parquet + both MP4s + all meta consistently.
  7. Report per-episode durations, frame counts, event times, flips, SHA256.

J5 branch policy (handoff appendix A):
  The raw capture stores J5 in two equivalent branches 2*pi apart
  (19 episodes near +4.6 rad, 31 near -1.9 rad) purely because of when they
  were recorded. Keeping both makes a single physical pose look like two
  different inputs. We therefore express every episode in the SAME branch as
  the majority (negative) by subtracting 2*pi where needed. This is a
  representation alignment of the recorded value, not a safety-envelope
  rewrite: the physical pose is unchanged and no canonicalisation to a
  guard range is applied. The decision and the affected episodes are
  recorded in the manifest.
"""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RAW = Path.home() / "red_parcel_raw/lerobot_dataset"          # read-only copy
RUN = Path.home() / "parcel_smolvla/20260828_red_parcel_clean_50"
CLEAN = RUN / "clean_lerobot_v3"
MAN = RUN / "manifest"
TASK = "把箱子里的红色包裹拿出来放到桌子上。"
OLD_EPS = list(range(50, 100))
CAMS = ["observation.images.chest", "observation.images.wrist_right"]

OPEN_T, CLOSE_T, CONFIRM, TAIL_S = 0.15, 0.85, 5, 7.0
TWO_PI = 2.0 * np.pi
J5 = 4
QS = [("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99)]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def gripper_events(g15, ts15):
    """Deployment state machine on the 15 Hz axis. Returns
    (t_close, t_open, flips, pre_grasp_flips)."""
    state, run_c, run_o = 0, 0, 0
    t_close = t_open = None
    flips = 0
    for i, v in enumerate(g15):
        run_c = run_c + 1 if v >= CLOSE_T else 0
        run_o = run_o + 1 if v <= OPEN_T else 0
        if state == 0 and run_c >= CONFIRM:
            state = 1
            flips += 1
            if t_close is None:
                t_close = float(ts15[i])
        elif state == 1 and run_o >= CONFIRM:
            state = 0
            flips += 1
            if t_close is not None and t_open is None:
                t_open = float(ts15[i])
    pre = 0
    if t_close is not None:
        pre = max(0, flips_before(g15, ts15, t_close) - 1)
    return t_close, t_open, flips, pre


def flips_before(g15, ts15, t):
    state, run_c, run_o, n = 0, 0, 0, 0
    for i, v in enumerate(g15):
        if ts15[i] > t:
            break
        run_c = run_c + 1 if v >= CLOSE_T else 0
        run_o = run_o + 1 if v <= OPEN_T else 0
        if state == 0 and run_c >= CONFIRM:
            state, n = 1, n + 1
        elif state == 1 and run_o >= CONFIRM:
            state, n = 0, n + 1
    return n


def feat_stats(a, count_override=None):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    s = {"min": a.min(0), "max": a.max(0), "mean": a.mean(0), "std": a.std(0),
         "count": np.array([count_override or len(a)])}
    for n, q in QS:
        s[n] = np.quantile(a, q, axis=0)
    return s


def cut_video(src: Path, dst: Path, n_frames: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                    "-frames:v", str(n_frames), "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "18", "-g", "2",
                    "-pix_fmt", "yuv420p", "-r", "30", "-an", str(dst)], check=True)
    cap = cv2.VideoCapture(str(dst))
    mins = np.full(3, np.inf); maxs = np.full(3, -np.inf)
    ssum = np.zeros(3); ssq = np.zeros(3); npx = 0
    samples = []; got = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        got += 1
        small = cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA)
        rgb = small[:, :, ::-1].astype(np.float64) / 255.0
        flat = rgb.reshape(-1, 3)
        mins = np.minimum(mins, flat.min(0)); maxs = np.maximum(maxs, flat.max(0))
        ssum += flat.sum(0); ssq += (flat ** 2).sum(0); npx += len(flat)
        if got % 5 == 1:
            samples.append(flat[::13])
    cap.release()
    if got != n_frames:
        raise RuntimeError(f"{dst.name}: decoded {got} != {n_frames}")
    mean = ssum / npx
    std = np.sqrt(np.maximum(0.0, ssq / npx - mean ** 2))
    samp = np.concatenate(samples, 0)
    s = {"min": mins, "max": maxs, "mean": mean, "std": std, "count": np.array([npx])}
    for n, q in QS:
        s[n] = np.quantile(samp, q, axis=0)
    return s, got


def main():
    assert RAW.is_dir(), f"raw copy missing: {RAW}"
    if CLEAN.exists():
        shutil.rmtree(CLEAN)
    for sub in ["data/chunk-000", "meta/episodes/chunk-000",
                *(f"videos/{c}/chunk-000" for c in CAMS)]:
        (CLEAN / sub).mkdir(parents=True, exist_ok=True)
    MAN.mkdir(parents=True, exist_ok=True)

    # ---- pass 1: decide the J5 branch per episode (majority wins)
    branch = {}
    for ep in OLD_EPS:
        d = pd.read_parquet(RAW / f"data/chunk-000/file-{ep:03d}.parquet",
                            columns=["observation.state"])
        j5 = np.stack(d["observation.state"].to_numpy())[:, J5]
        branch[ep] = "pos" if j5.mean() > 0 else "neg"
    n_pos = sum(1 for v in branch.values() if v == "pos")
    target = "neg" if n_pos <= len(OLD_EPS) / 2 else "pos"
    print(f"J5 branches: pos={n_pos} neg={len(OLD_EPS)-n_pos} -> aligning all to '{target}'")

    decisions, ep_stats, kept = [], [], []
    running = 0
    new_i = 0

    for ep in OLD_EPS:
        rec = {"old_index": ep, "j5_branch": branch[ep], "j5_shifted": False}
        d = pd.read_parquet(RAW / f"data/chunk-000/file-{ep:03d}.parquet")
        st = np.stack(d["observation.state"].to_numpy())
        ac = np.stack(d["action"].to_numpy())
        ts = d["timestamp"].to_numpy().astype(float)

        # step 2: structural checks
        if st.shape[1] != 8 or ac.shape[1] != 8:
            rec.update(decision="reject", reason=f"dims {st.shape}/{ac.shape}")
            decisions.append(rec); continue
        if not (np.isfinite(st).all() and np.isfinite(ac).all()):
            rec.update(decision="reject", reason="NaN/Inf")
            decisions.append(rec); continue
        if not (np.diff(ts) > 0).all():
            rec.update(decision="reject", reason="non-monotonic timestamps")
            decisions.append(rec); continue

        # steps 3-4: gripper events on the 15 Hz axis from the RAW teach action
        g15, ts15 = ac[::2, 7], ts[::2]
        t_close, t_open, flips, pre = gripper_events(g15, ts15)
        rec.update(t_close=t_close, t_open=t_open, gripper_flips=flips,
                   pre_grasp_flips=pre)
        if t_close is None:
            rec.update(decision="reject", reason="no confirmed close")
            decisions.append(rec); continue
        if t_open is None:
            rec.update(decision="reject", reason="no confirmed open after close")
            decisions.append(rec); continue
        if pre > 2:
            rec.update(decision="reject", reason=f"{pre} flips before grasp")
            decisions.append(rec); continue

        # step 5: cut at t_open + 7 s by real timestamp
        t_target = t_open + TAIL_S
        k = int(np.searchsorted(ts, t_target + 1e-9) - 1)
        n = min(k + 1, len(ts))
        t_cut = float(ts[n - 1])
        rec.update(t_cut=t_cut, orig_seconds=round(float(ts[-1]), 3),
                   clean_seconds=round(t_cut, 3),
                   orig_actions=len(d), clean_actions=n)

        dd = d.iloc[:n].copy()
        # J5 branch alignment (representation only, physical pose unchanged)
        if branch[ep] != target:
            shift = -TWO_PI if target == "neg" else TWO_PI
            s2 = np.stack(dd["observation.state"].to_numpy()); s2[:, J5] += shift
            a2 = np.stack(dd["action"].to_numpy()); a2[:, J5] += shift
            dd["observation.state"] = list(s2)
            dd["action"] = list(a2)
            rec["j5_shifted"] = True
            rec["j5_shift_rad"] = float(shift)

        dd["episode_index"] = new_i
        dd["index"] = np.arange(running, running + n, dtype=np.int64)
        dd["task_index"] = 0
        dd.to_parquet(CLEAN / f"data/chunk-000/file-{new_i:03d}.parquet", index=False)

        stats = {}
        stats["observation.state"] = feat_stats(np.stack(dd["observation.state"].to_numpy()))
        stats["action"] = feat_stats(np.stack(dd["action"].to_numpy()))
        for col in ["observation.gripper_contact", "observation.gripper_feedback_valid",
                    "timestamp", "frame_index", "episode_index", "index", "task_index"]:
            if col in dd.columns:
                stats[col] = feat_stats(dd[col].to_numpy().astype(np.float64))

        vframes = {}
        for cam in CAMS:
            s, got = cut_video(RAW / f"videos/{cam}/chunk-000/file-{ep:03d}.mp4",
                               CLEAN / f"videos/{cam}/chunk-000/file-{new_i:03d}.mp4", n)
            stats[cam] = s
            vframes[cam] = got
        rec["video_frames"] = vframes
        rec["video_fps"] = 30

        e = pd.read_parquet(RAW / f"meta/episodes/chunk-000/file-{ep:03d}.parquet").copy()
        e["tasks"] = [[TASK]]
        e["episode_index"] = new_i
        e["length"] = n
        e["data/file_index"] = new_i
        e["dataset_from_index"] = running
        e["dataset_to_index"] = running + n
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
        e.to_parquet(CLEAN / f"meta/episodes/chunk-000/file-{new_i:03d}.parquet", index=False)

        rec.update(decision="keep", new_index=new_i,
                   sha256_parquet=sha256_file(CLEAN / f"data/chunk-000/file-{new_i:03d}.parquet"))
        decisions.append(rec)
        ep_stats.append(stats)
        kept.append(new_i)
        running += n
        new_i += 1
        if new_i % 10 == 1:
            print(f"  ep{ep} -> {new_i-1}: {len(d)}->{n} rows, t_cut={t_cut:.2f}s", flush=True)

    n_keep = len(kept)
    print(f"\nkept {n_keep} / {len(OLD_EPS)}")

    # tasks / info
    t_src = pd.read_parquet(RAW / "meta/tasks.parquet")
    if t_src.index.dtype == object:
        pd.DataFrame({"task_index": [0]},
                     index=pd.Index([TASK], name=t_src.index.name)).to_parquet(CLEAN / "meta/tasks.parquet")
    else:
        pd.DataFrame({"task_index": [0], "task": [TASK]}).to_parquet(CLEAN / "meta/tasks.parquet")
    info = json.loads((RAW / "meta/info.json").read_text())
    info["total_episodes"] = n_keep
    info["total_frames"] = int(running)
    info["total_tasks"] = 1
    if "total_videos" in info:
        info["total_videos"] = 2 * n_keep
    if "splits" in info:
        info["splits"] = {"train": f"0:{n_keep}"}
    (CLEAN / "meta/info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False))

    # 45/5 split, seed 1000
    rng = np.random.default_rng(1000)
    n_val = 5 if n_keep >= 50 else max(1, n_keep // 10)
    val = sorted(rng.choice(np.arange(n_keep), size=n_val, replace=False).tolist())
    train = [i for i in range(n_keep) if i not in val]
    (RUN / "split_manifest.json").write_text(json.dumps(
        {"seed": 1000, "kept_episodes": n_keep, "task_text": TASK,
         "validation_new_indices": val, "train_new_indices": train,
         "note": "episode-level split; stats/training use train only"},
        indent=2, ensure_ascii=False))

    # global stats from TRAIN only
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

    (MAN / "cleaning_report.json").write_text(json.dumps({
        "run_id": RUN.name, "task_text": TASK, "raw_source": str(RAW),
        "rule": "open<=0.15 close>=0.85 5-tick @15Hz; t_cut=t_open+7.0s by timestamp",
        "j5_policy": f"aligned all episodes to the '{target}' branch by +/-2pi "
                     "(representation only; physical pose and safety envelope untouched)",
        "kept": n_keep, "rejected": len(OLD_EPS) - n_keep,
        "total_frames": int(running), "episodes": decisions}, indent=2, ensure_ascii=False))
    pd.DataFrame(decisions).to_csv(MAN / "episode_decisions.csv", index=False)

    print("CLEAN_DONE frames:", running, "val:", val)


if __name__ == "__main__":
    main()
