#!/usr/bin/env python3
"""Print saved baseline/QGF success rates for one real-robot comparison cohort.

Only finalized episode metadata is considered. Staging directories and discarded
episodes are deliberately excluded, so the displayed counts match the data that
can later be used for analysis or critic training.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MODE_RE = re.compile(r"(?:^|[;\s])policy_mode=(baseline|qgf)(?:[;\s]|$)")


def collect(root: Path, tag: str) -> dict[str, dict[str, int]]:
    result = {
        "baseline": {"kept": 0, "success": 0, "failure": 0},
        "qgf": {"kept": 0, "success": 0, "failure": 0},
    }
    for metadata_path in sorted((root / "episodes").glob("episode_*/episode_metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"QGF_COMPARISON_STATS_WARNING unreadable={metadata_path}: {exc}")
            continue
        notes = str(metadata.get("notes", ""))
        if tag and tag not in notes:
            continue
        match = MODE_RE.search(notes)
        if not match:
            continue
        mode = match.group(1)
        outcome = metadata.get("outcome")
        if outcome not in {"success", "failure"}:
            continue
        result[mode]["kept"] += 1
        result[mode][outcome] += 1
    return result


def rate(success: int, kept: int) -> str:
    return "n/a" if kept == 0 else f"{100.0 * success / kept:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--tag",
        default="",
        help="only include metadata whose notes contain this exact cohort tag",
    )
    args = parser.parse_args()
    stats = collect(args.dataset_root, args.tag)
    print(f"QGF_COMPARISON_STATS cohort={args.tag or '<all labeled modes>'}")
    for mode in ("baseline", "qgf"):
        item = stats[mode]
        print(
            "QGF_COMPARISON_STATS "
            f"mode={mode} kept={item['kept']} success={item['success']} "
            f"failure={item['failure']} success_rate={rate(item['success'], item['kept'])}"
        )
    baseline = stats["baseline"]
    qgf = stats["qgf"]
    if baseline["kept"] and qgf["kept"]:
        delta = 100.0 * qgf["success"] / qgf["kept"] - 100.0 * baseline["success"] / baseline["kept"]
        print(f"QGF_COMPARISON_STATS qgf_minus_baseline_pp={delta:+.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
