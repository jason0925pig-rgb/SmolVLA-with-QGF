#!/usr/bin/env python
"""Download model snapshots needed by this project."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="lerobot/smolvla_base")
    parser.add_argument("--local-dir", default="checkpoints/smolvla_base")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: pip install huggingface_hub") from exc

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    print(path)


if __name__ == "__main__":
    main()
