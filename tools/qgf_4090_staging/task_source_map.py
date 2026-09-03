"""Emit source_episode_map.json for ONE task's baseline cohort (handoff section 5).

Task-parameterized replacement for mug_source_map.py.  Read-only against the
Orin source; the JSON goes to stdout so the caller decides where it lands.

    python3 task_source_map.py <episode_list_file> [episodes_root]

episodes_root defaults to $QGF_ORIN_EPISODES.  There is NO default that
silently picks a task -- an unset variable is a fatal error.

Required environment
    QGF_TASK_KEY       short task slug, e.g. red_parcel
    QGF_DATASET_ID     e.g. red_parcel_baseline50_20260902
    QGF_ORIN_EPISODES  episodes root on the Orin
    QGF_ORIN_BUNDLE    the SmolVLA bundle that PRODUCED these rollouts
    QGF_BUNDLE_NAME    destination bundle leaf name on the 4090
    QGF_EPISODE_FIRST  first cohort episode index (inclusive)
    QGF_EPISODE_LAST   last  cohort episode index (inclusive)

Optional environment
    QGF_SSD_ROOT            recorded as the destination root when set
    QGF_RUN_ID              recorded for provenance when set
    QGF_EXPECTED_EPISODES   override the cohort size; defaults to
                            LAST - FIRST + 1.  Set it ONLY for a deliberately
                            non-contiguous frozen cohort.

Handoff section 5 requires this file to record, per episode, the source episode
number, the outcome, the file sizes and (via source_SHA256SUMS, written by
stage_task_to_4090.sh) the SHA256.  It also requires that the Orin originals are
never modified: nothing here opens a file for writing.
"""
import json
import os
import platform
import sys
import time

REQUIRED = ["episode_metadata.json", "transitions.parquet",
            "normalized_policy_chunks.parquet", "policy_observations.parquet",
            "chest.mp4", "wrist_right.mp4"]


def req_env(name):
    v = os.environ.get(name)
    if v is None or not v.strip():
        sys.exit(
            "FATAL: required environment variable %s is unset or empty.\n"
            "  task_source_map.py is task-parameterized and refuses to guess a task."
            % name)
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
SSD_ROOT = (os.environ.get("QGF_SSD_ROOT") or "").strip() or None
RUN_ID = (os.environ.get("QGF_RUN_ID") or "").strip() or None

if EP_LAST < EP_FIRST:
    sys.exit("FATAL: QGF_EPISODE_LAST (%d) < QGF_EPISODE_FIRST (%d)" % (EP_LAST, EP_FIRST))
_exp = (os.environ.get("QGF_EXPECTED_EPISODES") or "").strip()
if _exp:
    try:
        EXPECTED = int(_exp)
    except ValueError:
        sys.exit("FATAL: QGF_EXPECTED_EPISODES must be an integer, got %r" % _exp)
    EXPECTED_SOURCE = "QGF_EXPECTED_EPISODES override"
else:
    EXPECTED = EP_LAST - EP_FIRST + 1
    EXPECTED_SOURCE = "QGF_EPISODE_LAST - QGF_EPISODE_FIRST + 1"

if len(sys.argv) < 2:
    sys.exit("usage: python3 task_source_map.py <episode_list_file> [episodes_root]")
lst = sys.argv[1]
src = sys.argv[2] if len(sys.argv) > 2 else ORIN_EPISODES
if len(sys.argv) > 2 and os.path.normpath(src) != os.path.normpath(ORIN_EPISODES):
    sys.exit(
        "FATAL: argv episodes_root %r disagrees with QGF_ORIN_EPISODES %r.\n"
        "  Refusing to map a root the rest of the pipeline will not use."
        % (src, ORIN_EPISODES))

eps = sorted(
    l.strip() for l in open(lst)
    if l.strip() and not l.lstrip().startswith("#")
)

if len(eps) != len(set(eps)):
    sys.exit("FATAL: %s contains duplicate episode names" % lst)
if len(eps) != EXPECTED:
    sys.exit("FATAL: frozen list has %d episodes, expected %d (%s)"
             % (len(eps), EXPECTED, EXPECTED_SOURCE))

NOT_COPIED = [
    "chest.mjpeg", "wrist_right.mjpeg", "samples.jsonl",
    "normalized_policy_chunks.jsonl", "policy_observations.jsonl",
    "capture_summary.json", "monitor.log", "recorder.log",
    "episode_metadata.json.bak_before_lighting",
]

out = {
    "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "source_host": os.uname().nodename if hasattr(os, "uname") else platform.node(),
    "source_root": src,
    "task": TASK_KEY,
    "dataset_id": DATASET_ID,
    "run_id": RUN_ID,
    "episode_range": [EP_FIRST, EP_LAST],
    "expected_episodes": EXPECTED,
    "expected_episodes_source": EXPECTED_SOURCE,
    "required_training_files": REQUIRED,
    "policy_bundle_that_produced_these_rollouts": ORIN_BUNDLE,
    "policy_bundle_dest": (SSD_ROOT + "/policy_bundles/" + BUNDLE_NAME) if SSD_ROOT else BUNDLE_NAME,
    "dataset_dest": (SSD_ROOT + "/datasets/" + DATASET_ID) if SSD_ROOT else None,
    "renumbered": False,
    "not_copied_non_training_files": NOT_COPIED,
    "episodes": [],
}

total_bytes = 0
for e in eps:
    d = os.path.join(src, e)
    if not os.path.isdir(d):
        sys.exit("FATAL: not a directory: %s" % d)
    sizes = {}
    for f in REQUIRED:
        p = os.path.join(d, f)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            sys.exit("FATAL missing/empty required source file: %s" % p)
        sizes[f] = os.path.getsize(p)
    total_bytes += sum(sizes.values())
    m = json.load(open(os.path.join(src, e, "episode_metadata.json")))
    out["episodes"].append({
        "source_dir": e,
        "dest_dir": e,
        "source_episode_index": m.get("episode_index"),
        "dest_episode_index": m.get("episode_index"),
        "outcome": m.get("outcome"),
        "success": m.get("success"),
        "task_prompt": m.get("task_prompt"),
        "notes": m.get("notes"),
        "num_samples": m.get("num_samples"),
        "num_transitions": m.get("num_transitions"),
        "termination_source": m.get("termination_source"),
        "finalized_at_utc": m.get("finalized_at_utc"),
        "file_bytes": sizes,
        "bytes_training": sum(sizes.values()),
    })

idx = [e["source_episode_index"] for e in out["episodes"]]
bad = [e["source_dir"] for e, i in zip(out["episodes"], idx) if not isinstance(i, int)]
if bad:
    sys.exit("FATAL: non-integer episode_index in metadata for: %s" % bad)
out["renumbered"] = idx != list(range(min(idx), min(idx) + len(idx)))
if out["renumbered"]:
    out["renumber_note"] = (
        "source indices are NOT contiguous; dest keeps original names so the "
        "mapping stays traceable. If the trainer requires contiguous indices, "
        "renumber the COPY on the 4090 and update its metadata there."
    )

# handoff section 5 cohort assertions, mirrored here so the provenance record
# can never disagree with the data it describes.
outside = [e["source_dir"] for e, i in zip(out["episodes"], idx)
           if not (EP_FIRST <= i <= EP_LAST)]
if outside:
    sys.exit("FATAL: episodes outside [%d,%d]: %s" % (EP_FIRST, EP_LAST, outside))
prompts = sorted({e["task_prompt"] for e in out["episodes"] if e["task_prompt"]})
if len(prompts) != 1:
    sys.exit("FATAL: task_prompt is not unique across the cohort: %s" % prompts)
oc = {}
for e in out["episodes"]:
    oc[e["outcome"]] = oc.get(e["outcome"], 0) + 1
badoc = sorted({str(k) for k in oc if k not in ("success", "failure")})
if badoc:
    sys.exit("FATAL: outcomes outside {success,failure}: %s" % badoc)
if oc.get("success", 0) == 0 or oc.get("failure", 0) == 0:
    sys.exit("FATAL: cohort must contain BOTH success and failure episodes, got %s" % oc)

out["outcome_counts"] = oc
out["task_prompt"] = prompts[0]
out["total_training_bytes"] = total_bytes
out["total_training_files"] = len(eps) * len(REQUIRED)

json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
