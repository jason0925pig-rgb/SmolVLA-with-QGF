#!/usr/bin/env python3
"""Re-cut the red-parcel videos from the ORIGINAL sources with -g 2
(keyframe every other frame, matching the recorder's random-access-friendly
encoding). Single lossy generation. Frame counts from preprocessing_report.
Verifies decoded frame count == parquet rows. Runs on the a800 (has ffmpeg).
4-way parallel."""
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

SRC = Path.home() / "parcel_smolvla/20260825_parcel_50/source_full/lerobot_dataset"
RUN = Path.home() / "parcel_smolvla/20260827_red_parcel_out_table_50_trunc7s_45train_5val"
OUT = RUN / "videos_g2"
CAMS = ["observation.images.chest", "observation.images.wrist_right"]

report = json.loads((RUN / "preprocessing_report.json").read_text())
jobs = []
for r in report["episodes"]:
    for cam in CAMS:
        jobs.append((cam, r["old_index"], r["new_index"], r["new_len"]))


def one(job):
    cam, old_i, new_i, n = job
    src = SRC / f"videos/{cam}/chunk-000/file-{old_i:03d}.mp4"
    dst = OUT / cam / "chunk-000" / f"file-{new_i:03d}.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                    "-frames:v", str(n), "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "18", "-g", "2", "-pix_fmt", "yuv420p", "-r", "30",
                    "-an", str(dst)], check=True)
    cap = cv2.VideoCapture(str(dst))
    got = 0
    while cap.read()[0]:
        got += 1
    cap.release()
    if got != n:
        raise RuntimeError(f"{dst.name} decoded {got} != {n}")
    return f"{cam.split('.')[-1]}/{new_i:03d} ok ({n}f)"


with ThreadPoolExecutor(max_workers=4) as ex:
    for i, res in enumerate(ex.map(one, jobs)):
        if i % 10 == 0:
            print(i, res, flush=True)
print("RECUT_G2_DONE", len(jobs), "videos")
