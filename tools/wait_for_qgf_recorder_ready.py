#!/usr/bin/env python3
"""Wait until a current QGF staging recorder has both required policy artifacts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def nonempty_jsonl(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        return bool(json.loads(path.read_text(encoding="utf-8").splitlines()[-1]))
    except (IndexError, json.JSONDecodeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout
    chunks = args.staging / "normalized_policy_chunks.jsonl"
    observations = args.staging / "policy_observations.jsonl"
    while time.monotonic() < deadline:
        if nonempty_jsonl(chunks) and nonempty_jsonl(observations):
            print(f"QGF_RECORDER_READY staging={args.staging}")
            return 0
        time.sleep(0.10)
    raise SystemExit(
        "QGF recorder did not receive a current normalized policy chunk and observation "
        f"within {args.timeout:.1f}s: staging={args.staging}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
