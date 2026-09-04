#!/usr/bin/env python3
"""Audit the cleaned screwdriver dataset before it is transferred or trained on.

Runs against whichever copy it is pointed at, so the same script checks the Orin
copy and the a800new copy after transfer. Read-only. Exits 1 on any hard failure.

Task-specific checks beyond the usual structure pass:
  - the cut must land after the LAST gripper release, not the first, because
    three episodes re-grasp after dropping the screwdriver;
  - J5 must sit on a single branch (this data needed no 2*pi alignment, and a
    regression here would mean the wrong source episodes were pulled in).
"""
import glob
import hashlib
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

RUN = sys.argv[1] if len(sys.argv) > 1 else "/home/nvidia/work/telop/screwdriver_clean_50_20260904"
CLEAN = RUN + "/clean_lerobot_v3"
TASK = "把杯子里的螺丝刀放进纸盒里"
CAMS = ["observation.images.chest", "observation.images.wrist_right"]
OPEN_T, CLOSE_T, CONFIRM, GRIP, J5 = 0.15, 0.85, 5, 7, 4
TAU = 2 * np.pi

fails = []
warns = []


def check(ok, msg, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + msg + (("   " + detail) if detail else ""))
    if not ok:
        fails.append(msg)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def cycles(g, ts):
    out, rc, ro, st, tc = [], 0, 0, "open", None
    for i, v in enumerate(g):
        rc = rc + 1 if v >= CLOSE_T else 0
        ro = ro + 1 if v <= OPEN_T else 0
        if st == "open" and rc >= CONFIRM:
            st, tc = "closed", float(ts[i])
        elif st == "closed" and ro >= CONFIRM:
            out.append((tc, float(ts[i])))
            st, tc = "open", None
    if st == "closed":
        out.append((tc, None))
    return out


print("audit target:", RUN)
print()
print("=== 1. 结构 ===")
files = sorted(glob.glob(CLEAN + "/data/chunk-000/*.parquet"))
check(len(files) == 50, "50 个 parquet", "got %d" % len(files))
for cam in CAMS:
    v = sorted(glob.glob(CLEAN + "/videos/" + cam + "/chunk-000/*.mp4"))
    check(len(v) == 50, "50 个视频 " + cam.split(".")[-1], "got %d" % len(v))
meta = sorted(glob.glob(CLEAN + "/meta/episodes/chunk-000/*.parquet"))
check(len(meta) == 50, "50 个 episode meta", "got %d" % len(meta))
check(os.path.exists(CLEAN + "/meta/stats.json"), "meta/stats.json 存在(缺它训练会崩)")

info = json.load(open(CLEAN + "/meta/info.json"))
print()
print("=== 2. info.json ===")
check(info["total_episodes"] == 50, "total_episodes = 50", str(info["total_episodes"]))
check(info.get("fps") == 30, "fps = 30", str(info.get("fps")))

print()
print("=== 3. 逐 episode ===")
rows_total = 0
lens = []
j5_means = []
bad_ts = bad_fi = bad_idx = bad_vid = 0
cut_before_last_release = []
tail_s = []
for i, f in enumerate(files):
    d = pd.read_parquet(f)
    n = len(d)
    lens.append(n)
    ts = d["timestamp"].to_numpy().astype(float)
    fi = d["frame_index"].to_numpy()
    idx = d["index"].to_numpy()
    st = np.stack(d["observation.state"].to_numpy())
    ac = np.stack(d["action"].to_numpy())
    if abs(ts[0]) > 1e-4 or not (np.diff(ts) > 0).all():
        bad_ts += 1
    if fi[0] != 0 or not (np.diff(fi) == 1).all():
        bad_fi += 1
    if idx[0] != rows_total or not (np.diff(idx) == 1).all():
        bad_idx += 1
    if not (np.isfinite(st).all() and np.isfinite(ac).all()):
        fails.append("ep%d NaN/Inf" % i)
    if int(d["episode_index"].iloc[0]) != i:
        fails.append("ep%d episode_index mismatch" % i)
    for cam in CAMS:
        cap = cv2.VideoCapture(CLEAN + "/videos/" + cam + "/chunk-000/file-%03d.mp4" % i)
        cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if cnt != n:
            bad_vid += 1
            warns.append("ep%d %s frames %d != rows %d" % (i, cam.split(".")[-1], cnt, n))
    cy = cycles(ac[::2, GRIP], ts[::2])
    rel = [b for a, b in cy if b is not None]
    if rel:
        tail_s.append(float(ts[-1] - rel[-1]))
        # the kept segment must extend past the final release, else the episode
        # was cut at a drop rather than at the placement
        if ts[-1] < rel[-1]:
            cut_before_last_release.append(i)
    else:
        cut_before_last_release.append(i)
    j5_means.append(float(st[:, J5].mean()))
    rows_total += n

check(bad_ts == 0, "时间戳从 0 起算且严格递增", "违例 %d" % bad_ts)
check(bad_fi == 0, "frame_index 从 0 连续", "违例 %d" % bad_fi)
check(bad_idx == 0, "全局 index 跨 episode 连续", "违例 %d" % bad_idx)
check(bad_vid == 0, "视频帧数 == parquet 行数", "违例 %d" % bad_vid)
check(rows_total == info["total_frames"], "total_frames 与实际一致",
      "%d vs %d" % (info["total_frames"], rows_total))

print()
print("=== 4. 截断点(必须在最后一次释放之后)===")
check(not cut_before_last_release, "每条都保留到最后一次释放之后",
      "违例 %s" % cut_before_last_release if cut_before_last_release else "")
if tail_s:
    t = np.asarray(tail_s)
    print("      释放后保留时长: min %.2f  med %.2f  max %.2f s (目标 7.0s 或到结尾)"
          % (t.min(), np.median(t), t.max()))
    check(t.max() <= 7.5, "尾巴不超过 7s(+容差)", "max %.2f s" % t.max())

print()
print("=== 5. J5 单分支 ===")
m = np.asarray(j5_means)
spread = m.max() - m.min()
gap = np.diff(np.sort(m)).max()
print("      逐条均值范围 [%.3f, %.3f]" % (m.min(), m.max()))
check(spread < 1.5, "逐条均值极差远小于 2*pi", "%.3f rad = %.1f deg" % (spread, np.degrees(spread)))
check(gap < 1.0, "无双簇间隙", "最大间隙 %.3f rad" % gap)
check(len(set(np.sign(m))) == 1, "全部同一分支", "sign 集合 %s" % set(np.sign(m).astype(int)))

print()
print("=== 6. 各关节极差 ===")
S = np.concatenate([np.stack(pd.read_parquet(f)["observation.state"].to_numpy()) for f in files], axis=0)
for j in range(7):
    sp = S[:, j].max() - S[:, j].min()
    print("      J%d  %6.3f rad = %6.1f deg" % (j + 1, sp, np.degrees(sp)))

print()
print("=== 7. 任务串 ===")
t = pd.read_parquet(CLEAN + "/meta/tasks.parquet")
found = TASK in list(t.index.astype(str)) or (("task" in t.columns) and TASK in list(t["task"]))
check(found, "meta/tasks 含正确任务串")
mt = pd.read_parquet(meta[0])["tasks"].iloc[0]
mt = mt[0] if isinstance(mt, (list, np.ndarray)) else mt
check(str(mt) == TASK, "episode meta 任务串一致", repr(str(mt)))

print()
print("=== 8. 划分 ===")
sp = json.load(open(RUN + "/split_manifest.json"))
tr, va = sp["train"], sp["val"]
check(len(tr) == 45 and len(va) == 5, "45 训 / 5 留", "%d / %d" % (len(tr), len(va)))
check(not (set(tr) & set(va)), "train 与 val 不相交")
check(sorted(tr + va) == list(range(50)), "并集恰好 0..49")
check(sp.get("validation_new_indices") == va, "seq_eval 兼容键存在且一致")
check(sp["task_text"] == TASK, "split_manifest 任务串一致")

print()
print("=== 9. SHA256 复算 ===")
dec = pd.read_csv(RUN + "/manifest/episode_decisions.csv")
keep = dec[dec["decision"] == "keep"]
bad = sum(1 for _, r in keep.iterrows()
          if sha256_file(CLEAN + "/data/chunk-000/file-%03d.parquet" % int(r["new_index"]))
          != r["sha256_parquet"])
check(bad == 0, "50 条 parquet SHA256 与 manifest 一致", "不符 %d" % bad)

print()
tot = sum(os.path.getsize(p) for p in glob.glob(CLEAN + "/**/*", recursive=True) if os.path.isfile(p))
print("      数据集 %.2f GiB  总帧 %d  段长 min %d / med %d / max %d"
      % (tot / 2 ** 30, rows_total, min(lens), int(np.median(lens)), max(lens)))

print()
if warns:
    print("警告 %d 条:" % len(warns))
    for w in warns[:8]:
        print("   ", w)
if fails:
    print("失败 %d 项:" % len(fails))
    for f in fails[:12]:
        print("   ", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
