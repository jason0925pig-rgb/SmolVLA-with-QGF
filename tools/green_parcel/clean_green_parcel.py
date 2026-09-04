#!/usr/bin/env python3
"""Green-parcel segment extraction, derived from clean_red_parcel.py.

Difference from the red cleaner: the red segment ran from the episode start to a
cut point, so its timestamps already began at 0. The green segment is cut out of
the MIDDLE of the recording, so timestamp and frame_index must be rebased to 0
and the video trimmed by frame number rather than merely truncated.

Start time per episode comes from the adaptive plan (green_adaptive_plan.json):
each start was chosen inside [red_release, green_grasp - 6 s] to put the arm pose
as close as possible to a shared reference pose while keeping the green parcel
visible. End is the episode's own end, with no tail truncation.

J5 branch policy is identical to the red cleaner: the raw capture stores J5 in
two branches 2*pi apart purely because of when each episode was recorded, which
makes one physical pose look like two different inputs. Every episode is
expressed in the majority (negative) branch. This is representation alignment of
the recorded value, not a safety-envelope rewrite.

The raw copy is never modified.
"""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RAW = Path.home() / "red_parcel_raw/lerobot_dataset"
RUN = Path("/ssd/zwwl_user2/parcel_smolvla/20260903_green_parcel_clean_50")
CLEAN = RUN / "clean_lerobot_v3"
MAN = RUN / "manifest"
PLAN = Path("/tmp/green_adaptive_plan.json")
TASK = "把箱子里的绿色包裹拿出来放到桌子上。"
CAMS = ["observation.images.chest", "observation.images.wrist_right"]
FFMPEG = str(Path.home() / "parcel_env/lib/python3.10/site-packages/imageio_ffmpeg"
             "/binaries/ffmpeg-linux-x86_64-v7.0.2")
TWO_PI = 2.0 * np.pi
J5 = 4
FPS = 30
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


def cut_video(src, dst, i0, n):
    """Frames [i0, i0+n) of src, re-encoded with the red dataset's settings.

    Selection is by frame NUMBER, not timestamp: parquet rows and video frames
    are 1:1, so an off-by-one from keyframe seeking would silently desynchronise
    the images from the states.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Trim by frame number, then rebase the presentation clock to 0 so the video
    # starts where the rebased parquet timestamps do. ffmpeg 7 rejects -r together
    # with -vsync 0 ("contradictory"), and trim+setpts keeps the constant 30 fps
    # the red dataset uses, so this form is both correct and consistent with it.
    sel = "trim=start_frame=" + str(i0) + ",setpts=PTS-STARTPTS"
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(src),
                    "-vf", sel,
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
    assert RAW.is_dir(), "raw copy missing: " + str(RAW)
    assert PLAN.is_file(), "plan missing: " + str(PLAN)
    plan = json.loads(PLAN.read_text())
    starts = {int(e["ep"]): float(e["t0"]) for e in plan["episodes"]}
    old_eps = sorted(starts)
    print("plan: " + str(len(old_eps)) + " episodes, reference pose "
          + str([round(x, 3) for x in plan["reference_pose"]]))

    if CLEAN.exists():
        shutil.rmtree(CLEAN)
    subs = ["data/chunk-000", "meta/episodes/chunk-000"]
    subs += ["videos/" + c + "/chunk-000" for c in CAMS]
    for sub in subs:
        (CLEAN / sub).mkdir(parents=True, exist_ok=True)
    MAN.mkdir(parents=True, exist_ok=True)

    branch = {}
    for ep in old_eps:
        d = pd.read_parquet(RAW / ("data/chunk-000/file-%03d.parquet" % ep),
                            columns=["observation.state"])
        j5 = np.stack(d["observation.state"].to_numpy())[:, J5]
        branch[ep] = "pos" if j5.mean() > 0 else "neg"
    n_pos = sum(1 for v in branch.values() if v == "pos")
    target = "neg" if n_pos <= len(old_eps) / 2 else "pos"
    print("J5 branches: pos=%d neg=%d -> aligning all to '%s'"
          % (n_pos, len(old_eps) - n_pos, target))

    decisions = []
    kept = []
    running = 0
    new_i = 0
    for ep in old_eps:
        rec = {"old_index": ep, "j5_branch": branch[ep], "j5_shifted": False}
        d = pd.read_parquet(RAW / ("data/chunk-000/file-%03d.parquet" % ep))
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

        t0 = starts[ep]
        i0 = int(np.searchsorted(ts, t0 - 1e-9))
        n = len(ts) - i0
        if n < 30:
            rec.update(decision="reject", reason="only %d rows after t0" % n)
            decisions.append(rec)
            continue
        rec.update(t0=round(t0, 3), i0=i0, orig_actions=len(d), clean_actions=n,
                   orig_seconds=round(float(ts[-1]), 3),
                   clean_seconds=round(float(ts[-1] - ts[i0]), 3))

        dd = d.iloc[i0:].copy().reset_index(drop=True)
        if branch[ep] != target:
            shift = -TWO_PI if target == "neg" else TWO_PI
            s2 = np.stack(dd["observation.state"].to_numpy())
            s2[:, J5] += shift
            a2 = np.stack(dd["action"].to_numpy())
            a2[:, J5] += shift
            dd["observation.state"] = list(s2)
            dd["action"] = list(a2)
            rec["j5_shifted"] = True
            rec["j5_shift_rad"] = float(shift)

        # rebase the clock and the frame counter: this segment starts mid-recording
        base = float(ts[i0])
        dd["timestamp"] = (dd["timestamp"].to_numpy().astype(float) - base).astype(np.float32)
        dd["frame_index"] = np.arange(n, dtype=np.int64)
        dd["episode_index"] = new_i
        dd["index"] = np.arange(running, running + n, dtype=np.int64)
        dd["task_index"] = 0
        dd.to_parquet(CLEAN / ("data/chunk-000/file-%03d.parquet" % new_i), index=False)

        stats = {"observation.state": feat_stats(np.stack(dd["observation.state"].to_numpy())),
                 "action": feat_stats(np.stack(dd["action"].to_numpy()))}
        for col in ["observation.gripper_contact", "observation.gripper_feedback_valid",
                    "timestamp", "frame_index", "episode_index", "index", "task_index"]:
            if col in dd.columns:
                stats[col] = feat_stats(dd[col].to_numpy().astype(np.float64))

        vframes = {}
        for cam in CAMS:
            s, got = cut_video(RAW / ("videos/" + cam + "/chunk-000/file-%03d.mp4" % ep),
                               CLEAN / ("videos/" + cam + "/chunk-000/file-%03d.mp4" % new_i),
                               i0, n)
            stats[cam] = s
            vframes[cam] = got
        rec["video_frames"] = vframes
        rec["video_fps"] = FPS

        e = pd.read_parquet(RAW / ("meta/episodes/chunk-000/file-%03d.parquet" % ep)).copy()
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
        kept.append(new_i)
        running += n
        new_i += 1
        if new_i % 10 == 1:
            print("  ep%d -> %d: rows %d->%d, t0=%.2fs" % (ep, new_i - 1, len(d), n, t0), flush=True)

    n_keep = len(kept)
    print("\nkept %d / %d" % (n_keep, len(old_eps)))

    t_src = pd.read_parquet(RAW / "meta/tasks.parquet")
    if t_src.index.dtype == object:
        pd.DataFrame({"task_index": [0]},
                     index=pd.Index([TASK], name=t_src.index.name)
                     ).to_parquet(CLEAN / "meta/tasks.parquet")
    else:
        pd.DataFrame({"task_index": [0], "task": [TASK]}).to_parquet(CLEAN / "meta/tasks.parquet")

    info = json.loads((RAW / "meta/info.json").read_text())
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
         "note": "episode-level split; stats/training use train only"},
        indent=1, ensure_ascii=False))

    pd.DataFrame(decisions).to_csv(MAN / "episode_decisions.csv", index=False)
    (MAN / "cleaning_report.json").write_text(json.dumps(
        {"run_id": "20260903_green_parcel_clean_50",
         "task_text": TASK,
         "raw_source": str(RAW),
         "start_rule": "per-episode adaptive: pose closest to the shared reference "
                       "inside [red_release, green_grasp-6s] with the green parcel visible",
         "end_rule": "the episode's own end (no tail truncation)",
         "reference_pose": plan["reference_pose"],
         "min_approach_s": plan["min_approach_s"],
         "j5_policy": "aligned all episodes to the '" + target + "' branch by +/-2pi "
                      "(representation only; physical pose and safety envelope untouched)",
         "kept": n_keep, "rejected": len(old_eps) - n_keep,
         "total_frames": int(running),
         "episodes": decisions},
        indent=1, ensure_ascii=False, default=str))
    print("wrote manifest + split")


if __name__ == "__main__":
    main()
