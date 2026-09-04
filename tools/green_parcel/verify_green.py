#!/usr/bin/env python3
"""Audit the cleaned green-parcel dataset before any training time is spent.

The headline check is J5: the raw capture had a 6.764 rad (about 2*pi) spread
because episodes sat on two branches. After alignment the spread must be the
physical range of the joint, nowhere near 2*pi. Everything else here guards the
mid-recording cut: rebased clocks, contiguous frame indices, and parquet rows
matching video frames one for one.

Read-only. Exit code 1 if any hard check fails.
"""
import glob
import hashlib
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

RUN = "/ssd/zwwl_user2/parcel_smolvla/20260903_green_parcel_clean_50"
CLEAN = RUN + "/clean_lerobot_v3"
TASK = "把箱子里的绿色包裹拿出来放到桌子上。"
CAMS = ["observation.images.chest", "observation.images.wrist_right"]
TAU = 2 * np.pi
J5 = 4

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


print("=== 1. 结构 ===")
files = sorted(glob.glob(CLEAN + "/data/chunk-000/*.parquet"))
check(len(files) == 50, "50 个 parquet", "got %d" % len(files))
for cam in CAMS:
    v = sorted(glob.glob(CLEAN + "/videos/" + cam + "/chunk-000/*.mp4"))
    check(len(v) == 50, "50 个视频 " + cam.split(".")[-1], "got %d" % len(v))
meta = sorted(glob.glob(CLEAN + "/meta/episodes/chunk-000/*.parquet"))
check(len(meta) == 50, "50 个 episode meta", "got %d" % len(meta))

info = json.load(open(CLEAN + "/meta/info.json"))
print()
print("=== 2. info.json ===")
check(info["total_episodes"] == 50, "total_episodes = 50", str(info["total_episodes"]))
check(info.get("fps") == 30, "fps = 30", str(info.get("fps")))

print()
print("=== 3. 逐 episode:时钟重基、帧号、行数=视频帧数 ===")
rows_total = 0
j5_all = []
state_all = []
bad_ts = bad_fi = bad_vid = bad_idx = 0
lens = []
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
    j5_all.append(st[:, J5])
    state_all.append(st)
    rows_total += n

check(bad_ts == 0, "每条时间戳从 0 起算且严格递增", "违例 %d" % bad_ts)
check(bad_fi == 0, "frame_index 从 0 连续", "违例 %d" % bad_fi)
check(bad_idx == 0, "全局 index 跨 episode 连续", "违例 %d" % bad_idx)
check(bad_vid == 0, "视频帧数 == parquet 行数", "违例 %d" % bad_vid)
check(rows_total == info["total_frames"], "total_frames 与实际一致",
      "%d vs %d" % (info["total_frames"], rows_total))

print()
print("=== 4. J5 分支对齐(本次清洗的核心目标)===")
j5_cat = np.concatenate(j5_all)
per_ep_mean = np.array([x.mean() for x in j5_all])
spread_mean = per_ep_mean.max() - per_ep_mean.min()
spread_all = j5_cat.max() - j5_cat.min()
print("      每条 episode 的 J5 均值范围: [%.3f, %.3f]" % (per_ep_mean.min(), per_ep_mean.max()))
print("      逐条均值极差 %.3f rad   全部样本极差 %.3f rad   (2*pi = %.3f)"
      % (spread_mean, spread_all, TAU))
check(spread_mean < 1.0, "逐条 J5 均值极差 << 2*pi(原始为 6.764)",
      "%.3f rad = %.1f deg" % (spread_mean, np.degrees(spread_mean)))
check(spread_all < TAU * 0.5, "全部 J5 样本极差远小于 2*pi",
      "%.3f rad" % spread_all)
gaps = np.sort(per_ep_mean)
biggest_gap = np.diff(gaps).max() if len(gaps) > 1 else 0.0
check(biggest_gap < 1.0, "逐条均值无双簇间隙", "最大间隙 %.3f rad" % biggest_gap)

print()
print("=== 5. 各关节极差(对照:原始 J5 为 6.764)===")
S = np.concatenate(state_all, axis=0)
for j in range(7):
    sp = S[:, j].max() - S[:, j].min()
    print("      J%d  极差 %6.3f rad = %6.1f deg" % (j + 1, sp, np.degrees(sp)))

print()
print("=== 6. 任务串 ===")
t = pd.read_parquet(CLEAN + "/meta/tasks.parquet")
found = TASK in list(t.index.astype(str)) or (("task" in t.columns) and TASK in list(t["task"]))
check(found, "meta/tasks 含正确的绿包裹任务串")
mt = pd.read_parquet(meta[0])
mt_task = mt["tasks"].iloc[0]
mt_task = mt_task[0] if isinstance(mt_task, (list, np.ndarray)) else mt_task
check(str(mt_task) == TASK, "episode meta 的 tasks 一致", repr(str(mt_task)))

print()
print("=== 7. 划分 ===")
sp = json.load(open(RUN + "/split_manifest.json"))
tr, va = sp["train"], sp["val"]
check(len(tr) == 45 and len(va) == 5, "45 训 / 5 留", "%d / %d" % (len(tr), len(va)))
check(not (set(tr) & set(va)), "train 与 val 不相交")
check(sorted(tr + va) == list(range(50)), "并集恰好 0..49")
check(sp["task_text"] == TASK, "split_manifest 任务串一致")

print()
print("=== 8. SHA256 复算 vs 清洗时记录 ===")
dec = pd.read_csv(RUN + "/manifest/episode_decisions.csv")
keep = dec[dec["decision"] == "keep"]
bad = 0
for _, r in keep.iterrows():
    p = CLEAN + "/data/chunk-000/file-%03d.parquet" % int(r["new_index"])
    if sha256_file(p) != r["sha256_parquet"]:
        bad += 1
check(bad == 0, "50 条 parquet 的 SHA256 与 manifest 一致", "不符 %d" % bad)

print()
print("=== 9. 体积 ===")
tot = sum(os.path.getsize(p) for p in glob.glob(CLEAN + "/**/*", recursive=True) if os.path.isfile(p))
print("      数据集 %.2f GiB   总帧 %d   段长 min %d / med %d / max %d 帧"
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
