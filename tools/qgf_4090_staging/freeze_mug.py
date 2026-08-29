"""Freeze/inspect mug baseline rollouts per RTX4090 handoff section 5.
Read-only. Writes a manifest JSON to stdout-adjacent path.
Usage: python3 freeze_mug.py <episodes_root> <out_json>
"""
import json, os, subprocess, sys, time
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
REQUIRED = ["episode_metadata.json", "transitions.parquet",
            "normalized_policy_chunks.parquet", "policy_observations.parquet",
            "chest.mp4", "wrist_right.mp4"]
QUIET_S = 120  # a dir touched within this window may still be recording

now = time.time()
eps, problems = [], []
for d in sorted(ROOT.glob("episode_*")):
    rec = {"dir": d.name, "ok": True, "reasons": []}
    age = now - d.stat().st_mtime
    rec["age_s"] = round(age, 1)
    if age < QUIET_S:
        rec["ok"] = False
        rec["reasons"].append(f"still hot ({age:.0f}s < {QUIET_S}s) - may be recording")
    for f in REQUIRED:
        p = d / f
        if not p.is_file() or p.stat().st_size == 0:
            rec["ok"] = False
            rec["reasons"].append(f"missing/empty {f}")
    if not (d / "episode_metadata.json").is_file():
        eps.append(rec); continue
    m = json.load(open(d / "episode_metadata.json"))
    rec["episode_index"] = m.get("episode_index")
    rec["outcome"] = m.get("outcome")
    rec["task_prompt"] = m.get("task_prompt")
    rec["notes"] = m.get("notes", "")
    rec["num_samples"] = m.get("num_samples")
    rec["num_transitions"] = m.get("num_transitions")
    rec["termination_source"] = m.get("termination_source")
    rec["finalized_at_utc"] = m.get("finalized_at_utc")
    rec["camera_features"] = sorted((m.get("camera_features") or {}).keys())
    if rec["outcome"] not in ("success", "failure"):
        rec["ok"] = False; rec["reasons"].append(f"bad outcome {rec['outcome']!r}")
    # parquet structure
    try:
        t = pq.read_table(d / "normalized_policy_chunks.parquet")
        rec["chunk_rows"] = t.num_rows
        col = "action_chunk_normalized"
        if col in t.column_names and t.num_rows:
            v = t.column(col)[0].as_py()
            import numpy as np
            a = np.asarray(v, dtype=object)
            try:
                a = np.array(v, dtype=float)
                rec["chunk_shape"] = list(a.shape)
            except Exception:
                rec["chunk_shape"] = ["flat", len(v)]
            if rec["chunk_shape"] not in ([50, 8], ["flat", 400]):
                rec["ok"] = False; rec["reasons"].append(f"chunk shape {rec['chunk_shape']}")
        else:
            rec["ok"] = False; rec["reasons"].append(f"no {col}")
    except Exception as e:
        rec["ok"] = False; rec["reasons"].append(f"chunk parquet: {e}")
    try:
        t = pq.read_table(d / "transitions.parquet")
        rec["transition_rows"] = t.num_rows
        if t.num_rows == 0:
            rec["ok"] = False; rec["reasons"].append("transitions empty")
        for tc in ("timestamp", "observation_timestamp", "t", "time_s"):
            if tc in t.column_names:
                import numpy as np
                ts = np.array(t.column(tc).to_pylist(), dtype=float)
                rec["ts_col"] = tc
                rec["ts_monotonic"] = bool((np.diff(ts) > 0).all())
                rec["duration_s"] = round(float(ts[-1] - ts[0]), 2)
                if not rec["ts_monotonic"]:
                    rec["ok"] = False; rec["reasons"].append("timestamps not strictly increasing")
                break
    except Exception as e:
        rec["ok"] = False; rec["reasons"].append(f"transitions parquet: {e}")
    try:
        t = pq.read_table(d / "policy_observations.parquet")
        rec["obs_rows"] = t.num_rows
        for c in ("state", "observation_state", "policy_state"):
            if c in t.column_names and t.num_rows:
                rec["state_dim"] = len(t.column(c)[0].as_py())
                if rec["state_dim"] != 8:
                    rec["ok"] = False; rec["reasons"].append(f"state dim {rec['state_dim']}")
                break
    except Exception as e:
        rec["ok"] = False; rec["reasons"].append(f"obs parquet: {e}")
    # video decodability (header + stream probe only)
    for cam in ("chest.mp4", "wrist_right.mp4"):
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_frames,avg_frame_rate,duration",
                 "-of", "default=nw=1", str(d / cam)],
                capture_output=True, text=True, timeout=60)
            rec[cam.replace(".mp4", "_probe")] = out.stdout.strip().replace("\n", " ")
            if out.returncode != 0:
                rec["ok"] = False; rec["reasons"].append(f"{cam} probe rc={out.returncode}")
        except FileNotFoundError:
            rec[cam.replace(".mp4", "_probe")] = "ffprobe-missing"
        except Exception as e:
            rec["ok"] = False; rec["reasons"].append(f"{cam}: {e}")
    rec["bytes_training"] = sum((d / f).stat().st_size for f in REQUIRED if (d / f).is_file())
    eps.append(rec)
    if not rec["ok"]:
        problems.append(rec["dir"])

prompts = sorted({e.get("task_prompt") for e in eps if e.get("task_prompt")})
cams = sorted({tuple(e.get("camera_features") or []) for e in eps})
notes_profiles = sorted({e.get("notes", "") for e in eps})
oc = {}
for e in eps:
    oc[e.get("outcome")] = oc.get(e.get("outcome"), 0) + 1

summary = {
    "episodes_root": str(ROOT),
    "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "n_dirs": len(eps),
    "n_ok": sum(1 for e in eps if e["ok"]),
    "problems": problems,
    "outcome_counts": oc,
    "distinct_prompts": prompts,
    "distinct_camera_keysets": [list(c) for c in cams],
    "distinct_notes": notes_profiles,
    "training_bytes_ok_only": sum(e["bytes_training"] for e in eps if e["ok"] and "bytes_training" in e),
    "episodes": eps,
}
OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

print(f"dirs={summary['n_dirs']}  ok={summary['n_ok']}  problems={problems}")
print(f"outcome: {oc}")
print(f"prompts({len(prompts)}): {prompts}")
print(f"camera keysets: {summary['distinct_camera_keysets']}")
print("notes profiles:")
for n in notes_profiles:
    print("   ", n)
print(f"training bytes (ok only): {summary['training_bytes_ok_only']/2**30:.2f} GiB")
print(f"manifest -> {OUT}")
