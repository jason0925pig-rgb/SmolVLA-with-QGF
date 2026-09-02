"""Freeze/inspect ONE task's baseline rollouts per RTX4090 handoff section 5.

Task-parameterized replacement for freeze_mug.py.  Read-only against the Orin
source: it opens files, it never writes, renames or deletes anything under the
episodes root.  It writes a manifest JSON to the requested output path only.

    python3 freeze_task.py [episodes_root] [out_json]

Both positional arguments are optional; when omitted they come from the
environment.  There is NO default that silently picks a task -- an unset
variable is a fatal error.

Required environment
    QGF_TASK_KEY       short task slug, e.g. red_parcel
    QGF_DATASET_ID     e.g. red_parcel_baseline50_20260902
    QGF_ORIN_EPISODES  episodes root on the Orin (used when argv[1] omitted)
    QGF_ORIN_BUNDLE    the SmolVLA bundle that PRODUCED these rollouts
    QGF_BUNDLE_NAME    destination bundle leaf name on the 4090
    QGF_EPISODE_FIRST  first cohort episode index (inclusive)
    QGF_EPISODE_LAST   last  cohort episode index (inclusive)

Optional environment
    QGF_RUN_ID              recorded for provenance when set
    QGF_EXPECTED_EPISODES   override the cohort size; defaults to
                            LAST - FIRST + 1.  Set it ONLY for a deliberately
                            non-contiguous frozen cohort (handoff section 5),
                            and say so in the run log.
    QGF_FREEZE_SHA256=1     also SHA256 every required file here.  Default 0
                            because stage_task_to_4090.sh already produces the
                            authoritative source_SHA256SUMS and hashing ~19 GiB
                            twice on a live Orin is wasteful.  The pipeline
                            still records a SHA256 per file either way.

Cohort membership is decided by DIRECTORY NAME index, i.e. episode_%06d in
[QGF_EPISODE_FIRST, QGF_EPISODE_LAST].  That is exactly how
qgf/scripts/build_real_robot_visual_iql_manifest.py later selects episodes
(--episode-first/--episode-last), so the freeze verdict and the trainer agree
by construction.  Directories outside the range are still inspected and
reported, but they are pilots/tests and never gate the verdict.

Exit status: 0 only when the cohort verdict is PASS.  The manifest JSON is
always written first, so a failing verdict is still fully diagnosable.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

REQUIRED = ["episode_metadata.json", "transitions.parquet",
            "normalized_policy_chunks.parquet", "policy_observations.parquet",
            "chest.mp4", "wrist_right.mp4"]
QUIET_S = 120  # a dir touched within this window may still be recording
EP_NAME_RE = re.compile(r"^episode_(\d+)$")


def req_env(name):
    v = os.environ.get(name)
    if v is None or not v.strip():
        sys.exit(
            "FATAL: required environment variable %s is unset or empty.\n"
            "  freeze_task.py is task-parameterized and refuses to guess a task.\n"
            "  Export the full QGF_* contract before running." % name)
    return v.strip()


def int_env(name):
    raw = req_env(name)
    try:
        return int(raw)
    except ValueError:
        sys.exit("FATAL: %s must be an integer, got %r" % (name, raw))


TASK_KEY = req_env("QGF_TASK_KEY")
DATASET_ID = req_env("QGF_DATASET_ID")
ORIN_EPISODES = req_env("QGF_ORIN_EPISODES")
ORIN_BUNDLE = req_env("QGF_ORIN_BUNDLE")
BUNDLE_NAME = req_env("QGF_BUNDLE_NAME")
EP_FIRST = int_env("QGF_EPISODE_FIRST")
EP_LAST = int_env("QGF_EPISODE_LAST")
RUN_ID = (os.environ.get("QGF_RUN_ID") or "").strip() or None
WANT_SHA256 = (os.environ.get("QGF_FREEZE_SHA256") or "0").strip() in ("1", "true", "yes")

if EP_LAST < EP_FIRST:
    sys.exit("FATAL: QGF_EPISODE_LAST (%d) < QGF_EPISODE_FIRST (%d)" % (EP_LAST, EP_FIRST))
RANGE_COUNT = EP_LAST - EP_FIRST + 1
_exp = (os.environ.get("QGF_EXPECTED_EPISODES") or "").strip()
if _exp:
    try:
        EXPECTED = int(_exp)
    except ValueError:
        sys.exit("FATAL: QGF_EXPECTED_EPISODES must be an integer, got %r" % _exp)
    EXPECTED_SOURCE = "QGF_EXPECTED_EPISODES override"
else:
    EXPECTED = RANGE_COUNT
    EXPECTED_SOURCE = "QGF_EPISODE_LAST - QGF_EPISODE_FIRST + 1"
if EXPECTED <= 0:
    sys.exit("FATAL: expected cohort size must be > 0 (got %d)" % EXPECTED)

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(ORIN_EPISODES)
if len(sys.argv) > 1 and ROOT != Path(ORIN_EPISODES):
    sys.exit(
        "FATAL: argv episodes_root %r disagrees with QGF_ORIN_EPISODES %r.\n"
        "  Refusing to freeze a root the rest of the pipeline will not use."
        % (str(ROOT), ORIN_EPISODES))
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/%s_freeze_manifest.json" % TASK_KEY)
if not ROOT.is_dir():
    sys.exit("FATAL: episodes root is not a directory: %s" % ROOT)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


now = time.time()
eps, problems = [], []
for d in sorted(ROOT.glob("episode_*")):
    rec = {"dir": d.name, "ok": True, "reasons": []}
    m_name = EP_NAME_RE.match(d.name)
    rec["dir_index"] = int(m_name.group(1)) if m_name else None
    rec["in_cohort"] = bool(
        d.is_dir() and rec["dir_index"] is not None
        and EP_FIRST <= rec["dir_index"] <= EP_LAST)
    if not d.is_dir():
        rec["ok"] = False
        rec["reasons"].append("not a directory")
        eps.append(rec)
        problems.append(rec["dir"])
        continue
    if rec["dir_index"] is None:
        rec["ok"] = False
        rec["reasons"].append("directory name does not match episode_<digits>")
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
        eps.append(rec)
        problems.append(rec["dir"])
        continue
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
    # the trainer indexes by directory name but reads metadata["episode_index"];
    # a disagreement between the two silently mislabels a whole episode.
    if rec["dir_index"] is not None:
        try:
            mi = int(rec["episode_index"])
        except (TypeError, ValueError):
            rec["ok"] = False
            rec["reasons"].append(f"episode_index not an int: {rec['episode_index']!r}")
        else:
            if mi != rec["dir_index"]:
                rec["ok"] = False
                rec["reasons"].append(
                    f"metadata episode_index {mi} != directory index {rec['dir_index']}")
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
    rec["file_bytes"] = {f: (d / f).stat().st_size for f in REQUIRED if (d / f).is_file()}
    rec["bytes_training"] = sum((d / f).stat().st_size for f in REQUIRED if (d / f).is_file())
    if WANT_SHA256:
        rec["file_sha256"] = {}
        for f in REQUIRED:
            p = d / f
            if p.is_file():
                try:
                    rec["file_sha256"][f] = sha256_file(p)
                except Exception as e:
                    rec["ok"] = False; rec["reasons"].append(f"sha256 {f}: {e}")
    eps.append(rec)
    if not rec["ok"]:
        problems.append(rec["dir"])

cohort = [e for e in eps if e.get("in_cohort")]
outside = [e for e in eps if not e.get("in_cohort")]

prompts = sorted({e.get("task_prompt") for e in eps if e.get("task_prompt")})
cams = sorted({tuple(e.get("camera_features") or []) for e in eps})
notes_profiles = sorted({e.get("notes", "") for e in eps})
oc = {}
for e in eps:
    oc[e.get("outcome")] = oc.get(e.get("outcome"), 0) + 1

cohort_prompts = sorted({e.get("task_prompt") for e in cohort if e.get("task_prompt")})
cohort_cams = sorted({tuple(e.get("camera_features") or []) for e in cohort})
cohort_oc = {}
for e in cohort:
    cohort_oc[e.get("outcome")] = cohort_oc.get(e.get("outcome"), 0) + 1
cohort_problems = [e["dir"] for e in cohort if not e["ok"]]

# ---------- handoff section 5 cohort verdict ----------
failures = []
if len(cohort) != EXPECTED:
    failures.append(
        "cohort size %d != expected %d (%s); refusing to take 'whatever N "
        "directories are there'" % (len(cohort), EXPECTED, EXPECTED_SOURCE))
if cohort_problems:
    failures.append("cohort episodes failed per-episode checks: %s" % cohort_problems)
missing_idx = sorted(set(range(EP_FIRST, EP_LAST + 1))
                     - {e["dir_index"] for e in cohort})
if not _exp and missing_idx:
    failures.append("episode indices missing from [%d,%d]: %s"
                    % (EP_FIRST, EP_LAST, missing_idx))
dup = sorted({i for i in [e["dir_index"] for e in cohort]
              if [e["dir_index"] for e in cohort].count(i) > 1})
if dup:
    failures.append("duplicate cohort directory indices: %s" % dup)
if len(cohort_prompts) != 1:
    failures.append("cohort task_prompt is not unique: %s" % cohort_prompts)
if len(cohort_cams) != 1:
    failures.append("cohort camera keyset is not unique: %s"
                    % [list(c) for c in cohort_cams])
bad_oc = sorted({str(k) for k in cohort_oc if k not in ("success", "failure")})
if bad_oc:
    failures.append("cohort outcomes outside {success,failure}: %s" % bad_oc)
if cohort_oc.get("success", 0) == 0 or cohort_oc.get("failure", 0) == 0:
    failures.append("cohort must contain BOTH success and failure episodes, got %s"
                    % cohort_oc)
cohort_bytes = sum(e.get("bytes_training", 0) for e in cohort)
if cohort_bytes <= 0:
    failures.append("cohort training bytes is zero")

verdict = "PASS" if not failures else "FAIL"

summary = {
    "episodes_root": str(ROOT),
    "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "task": {
        "task_key": TASK_KEY,
        "dataset_id": DATASET_ID,
        "run_id": RUN_ID,
        "orin_episodes": ORIN_EPISODES,
        "policy_bundle_that_produced_these_rollouts": ORIN_BUNDLE,
        "bundle_name": BUNDLE_NAME,
        "episode_first": EP_FIRST,
        "episode_last": EP_LAST,
        "expected_episodes": EXPECTED,
        "expected_episodes_source": EXPECTED_SOURCE,
        "sha256_computed_here": WANT_SHA256,
    },
    "required_files": REQUIRED,
    "n_dirs": len(eps),
    "n_ok": sum(1 for e in eps if e["ok"]),
    "problems": problems,
    "outcome_counts": oc,
    "distinct_prompts": prompts,
    "distinct_camera_keysets": [list(c) for c in cams],
    "distinct_notes": notes_profiles,
    "training_bytes_ok_only": sum(e["bytes_training"] for e in eps if e["ok"] and "bytes_training" in e),
    "cohort": {
        "n": len(cohort),
        "dirs": [e["dir"] for e in cohort],
        "problems": cohort_problems,
        "outcome_counts": cohort_oc,
        "distinct_prompts": cohort_prompts,
        "distinct_camera_keysets": [list(c) for c in cohort_cams],
        "training_bytes": cohort_bytes,
        "missing_indices": missing_idx,
    },
    "outside_cohort_dirs": [e["dir"] for e in outside],
    "verdict": verdict,
    "verdict_failures": failures,
    "episodes": eps,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

print(f"task={TASK_KEY}  dataset={DATASET_ID}  range=[{EP_FIRST},{EP_LAST}]  expected={EXPECTED}")
print(f"dirs={summary['n_dirs']}  ok={summary['n_ok']}  problems={problems}")
print(f"outcome: {oc}")
print(f"prompts({len(prompts)}): {prompts}")
print(f"camera keysets: {summary['distinct_camera_keysets']}")
print("notes profiles:")
for n in notes_profiles:
    print("   ", n)
print(f"training bytes (ok only): {summary['training_bytes_ok_only']/2**30:.2f} GiB")
print()
print(f"cohort n={len(cohort)}  outcome={cohort_oc}  problems={cohort_problems}")
print(f"cohort prompts({len(cohort_prompts)}): {cohort_prompts}")
print(f"cohort camera keysets: {[list(c) for c in cohort_cams]}")
print(f"cohort training bytes: {cohort_bytes/2**30:.2f} GiB")
if outside:
    print(f"WARNING outside-cohort dirs ignored by the verdict: {[e['dir'] for e in outside]}")
print(f"manifest -> {OUT}")
print()
print(f"=== COHORT VERDICT: {verdict} ===")
for f in failures:
    print("  FAIL:", f)
if failures:
    print("Handoff section 5: stop here.  Do not transfer, do not renumber the "
          "Orin originals, do not substitute deleted episodes.")
    sys.exit(1)
