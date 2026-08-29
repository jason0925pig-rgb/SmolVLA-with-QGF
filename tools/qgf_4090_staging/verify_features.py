"""Handoff section 10 acceptance for the visual feature cache.

Per episode: shapes [N,128,960] / [N,50,8] / [N,8], finite, size, SHA256.
Also totals and the skip accounting from the manifest summary.

Usage: verify_features.py <run_dir> [report_json_out]
"""
import hashlib
import io
import json
import sys
from pathlib import Path

import torch

RUN = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else RUN / "manifest/feature_cache_report.json"
FEAT = RUN / "features"

fail = []


def check(cond, msg):
    if not cond:
        fail.append(msg)
        print("  FAIL  " + msg)


files = sorted(FEAT.glob("episode_*.pt"))
print(f"feature caches: {len(files)}")
check(len(files) == 50, f"expected 50 caches, got {len(files)}")

rows = []
total = 0
for p in files:
    d = torch.load(p, map_location="cpu", weights_only=False)
    vf = d["visual_features"]
    ac = d["action_chunk"]
    st = d["state"]
    n = int(vf.shape[0])
    total += n
    h = hashlib.sha256()
    with io.open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    row = {
        "file": p.name,
        "samples": n,
        "visual_shape": list(vf.shape),
        "action_shape": list(ac.shape),
        "state_shape": list(st.shape),
        "visual_dtype": str(vf.dtype),
        "bytes": p.stat().st_size,
        "sha256": h.hexdigest(),
        "finite": bool(
            torch.isfinite(vf.float()).all()
            and torch.isfinite(ac.float()).all()
            and torch.isfinite(st.float()).all()
        ),
    }
    rows.append(row)
    if list(vf.shape[1:]) != [128, 960]:
        check(False, f"{p.name}: visual shape {list(vf.shape)} != [N,128,960]")
    if list(ac.shape[1:]) != [50, 8]:
        check(False, f"{p.name}: action shape {list(ac.shape)} != [N,50,8]")
    if list(st.shape[1:]) != [8]:
        check(False, f"{p.name}: state shape {list(st.shape)} != [N,8]")
    if not row["finite"]:
        check(False, f"{p.name}: contains NaN/Inf")
    if n != int(ac.shape[0]) or n != int(st.shape[0]):
        check(False, f"{p.name}: N mismatch vf={n} ac={ac.shape[0]} st={st.shape[0]}")

print(f"total samples across caches: {total}")

ms = RUN / "manifest/manifest_summary.json"
summary_chunks = None
if ms.is_file():
    m = json.load(io.open(ms, encoding="utf-8"))
    summary_chunks = m.get("aligned_chunk_count")
    print(f"manifest aligned_chunk_count: {summary_chunks}")
    print(f"manifest skip accounting     : {m.get('alignment', {}).get('skipped')}")
    check(total == summary_chunks,
          f"feature sample total {total} != manifest aligned_chunk_count {summary_chunks}")

sizes = [r["bytes"] for r in rows]
print(f"cache size: total {sum(sizes) / 2**30:.2f} GiB, "
      f"min {min(sizes) / 2**20:.1f} MiB, max {max(sizes) / 2**20:.1f} MiB")
print(f"visual dtype(s): {sorted({r['visual_dtype'] for r in rows})}")

report = {
    "feature_dir": str(FEAT),
    "cache_count": len(files),
    "total_samples": total,
    "manifest_aligned_chunk_count": summary_chunks,
    "total_bytes": sum(sizes),
    "episodes": rows,
}
OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"report -> {OUT}")

print()
print("per-episode (first 8):")
for r in rows[:8]:
    print(f"  {r['file']:22s} N={r['samples']:4d} {r['visual_shape']}  "
          f"{r['bytes'] / 2**20:6.1f} MiB  {r['sha256'][:16]}")

print()
if fail:
    print(f"FAILED {len(fail)} check(s)")
    sys.exit(1)
print("ALL FEATURE CHECKS PASSED")
