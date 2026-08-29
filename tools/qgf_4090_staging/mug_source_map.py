"""Emit source_episode_map.json for the mug baseline cohort (handoff section 5).

Usage: python3 mug_source_map.py <episode_list_file> <episodes_root>
"""
import json
import os
import sys
import time

lst, src = sys.argv[1], sys.argv[2]
eps = sorted(
    l.strip() for l in open(lst)
    if l.strip() and not l.lstrip().startswith("#")
)

NOT_COPIED = [
    "chest.mjpeg", "wrist_right.mjpeg", "samples.jsonl",
    "normalized_policy_chunks.jsonl", "policy_observations.jsonl",
    "capture_summary.json", "monitor.log", "recorder.log",
    "episode_metadata.json.bak_before_lighting",
]

out = {
    "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "source_host": os.uname().nodename,
    "source_root": src,
    "task": "mug_purple_box",
    "policy_bundle_that_produced_these_rollouts":
        "/home/nvidia/work/telop/models/smolvla_20260827_mug_purple_box",
    "renumbered": False,
    "not_copied_non_training_files": NOT_COPIED,
    "episodes": [],
}

for e in eps:
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
    })

idx = [e["source_episode_index"] for e in out["episodes"]]
out["renumbered"] = idx != list(range(min(idx), min(idx) + len(idx)))
if out["renumbered"]:
    out["renumber_note"] = (
        "source indices are NOT contiguous; dest keeps original names so the "
        "mapping stays traceable. If the trainer requires contiguous indices, "
        "renumber the COPY on the 4090 and update its metadata there."
    )

json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
