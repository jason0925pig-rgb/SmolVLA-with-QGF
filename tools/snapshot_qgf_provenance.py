#!/usr/bin/env python3
"""Snapshot small policy/preprocessor configs needed to interpret rollout data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    destination = args.dataset_root / "provenance" / "policy_configs"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    weight_records = []
    for source in sorted(args.checkpoint.rglob("*")):
        if not source.is_file():
            continue
        if source.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}:
            weight_records.append(
                {
                    "path": str(source.relative_to(args.checkpoint)).replace("\\", "/"),
                    "size": source.stat().st_size,
                    "sha256": sha256(source),
                }
            )
            continue
        if source.suffix.lower() not in {".json", ".yaml", ".yml"}:
            continue
        if source.stat().st_size > 10 * 1024 * 1024:
            continue
        relative = source.relative_to(args.checkpoint)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(
            {
                "path": str(relative).replace("\\", "/"),
                "size": source.stat().st_size,
                "sha256": sha256(source),
            }
        )
    revision = subprocess.run(
        ["git", "-C", str(args.project_root), "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout.strip()
    manifest = {
        "schema_version": "qgf-provenance-1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_revision": revision or "unknown",
        "checkpoint_source_path": str(args.checkpoint),
        "config_files": records,
        "weight_files_not_copied": weight_records,
        "note": (
            "Weights are not duplicated here. Preserve the referenced checkpoint; "
            "these configs include preprocessing/normalization metadata needed to "
            "map physical actions back into the policy action space."
        ),
    }
    (args.dataset_root / "provenance" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"QGF_PROVENANCE_READY files={len(records)} revision={manifest['project_revision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
