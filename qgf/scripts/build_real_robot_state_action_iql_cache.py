#!/usr/bin/env python3
"""Build no-vision QGF IQL rows from recorded real-robot episodes.

This mirrors the temporal action-chunk alignment of the visual-Q manifest,
while deliberately requiring no camera files.  Each cached row is
``(state, normalized 50x8 action chunk, next_state, reward, done)``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-episodes-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-first", type=int, required=True)
    parser.add_argument("--episode-last", type=int, required=True)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--policy-hz", type=float, default=15.0)
    parser.add_argument("--max-transition-gap-seconds", type=float, default=0.1)
    return parser.parse_args()


def nearest_index(timestamps: list[int], timestamp: int) -> tuple[int, int]:
    import numpy as np
    values = np.asarray(timestamps, dtype=np.int64)
    position = int(np.searchsorted(values, timestamp, side="left"))
    candidates = [index for index in (position - 1, position) if 0 <= index < len(values)]
    index = min(candidates, key=lambda candidate: abs(int(values[candidate]) - timestamp))
    return index, abs(int(values[index]) - timestamp)


def choose_observation(rows: list[dict], chunk_timestamp: int) -> dict | None:
    if not rows:
        return None
    return min(rows, key=lambda row: abs(int(row["timestamp_ns"]) - chunk_timestamp))


def rows_by_timestep(rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["observation_timestep"]), []).append(row)
    return grouped


def build_episode(episode_dir: Path, args: argparse.Namespace) -> tuple[list[dict], Counter]:
    import pyarrow.parquet as pq

    chunks = pq.read_table(episode_dir / "normalized_policy_chunks.parquet").to_pylist()
    observations = pq.read_table(episode_dir / "policy_observations.parquet").to_pylist()
    transitions = pq.read_table(episode_dir / "transitions.parquet").to_pylist()
    metadata = json.loads((episode_dir / "episode_metadata.json").read_text(encoding="utf-8"))
    if not transitions:
        raise ValueError(f"{episode_dir}: empty transitions")
    observation_rows = rows_by_timestep(observations)
    timestamps = [int(row["timestamp_ns"]) for row in transitions]
    max_gap_ns = int(args.max_transition_gap_seconds * 1_000_000_000)
    horizon_ns = int(round(args.action_horizon / args.policy_hz * 1_000_000_000))
    samples, skipped = [], Counter()
    for chunk in chunks:
        action = chunk["action_chunk_normalized"]
        if len(action) != args.action_horizon or any(len(step) != 8 for step in action):
            skipped["invalid_action_chunk_shape"] += 1
            continue
        observation = choose_observation(
            observation_rows.get(int(chunk["observation_timestep"]), []), int(chunk["timestamp_ns"])
        )
        if observation is None:
            skipped["missing_aligned_policy_observation"] += 1
            continue
        start_timestamp = int(observation["timestamp_ns"])
        start_index, start_gap = nearest_index(timestamps, start_timestamp)
        if start_gap > max_gap_ns:
            skipped["start_transition_too_far"] += 1
            continue
        target_timestamp = start_timestamp + horizon_ns
        end_index, _ = nearest_index(timestamps, target_timestamp)
        if target_timestamp >= timestamps[-1]:
            end_index = len(transitions) - 1
        if end_index < start_index:
            skipped["invalid_transition_order"] += 1
            continue
        span = transitions[start_index : end_index + 1]
        terminal = next((row for row in span if bool(row["done"])), None)
        if terminal is not None:
            end_index = start_index + span.index(terminal)
            span = transitions[start_index : end_index + 1]
        end_row = transitions[end_index]
        samples.append({
            "state": observation["state"], "action_chunk": action, "next_state": end_row["next_state"],
            "reward": float(sum(float(row["reward"]) for row in span)),
            "success": bool(any(row["success"] for row in span)),
            "done": bool(any(row["done"] for row in span)),
            "terminated": bool(any(row["terminated"] for row in span)),
            "truncated": bool(any(row["truncated"] for row in span)),
        })
    return samples, skipped


def main() -> None:
    args = parse_args()
    if args.episode_first < 0 or args.episode_last < args.episode_first:
        raise ValueError("invalid episode range")
    import torch
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report, total_skipped = [], Counter()
    for index in range(args.episode_first, args.episode_last + 1):
        episode_dir = args.raw_episodes_root / f"episode_{index:06d}"
        required = [
            episode_dir / "episode_metadata.json", episode_dir / "transitions.parquet",
            episode_dir / "policy_observations.parquet", episode_dir / "normalized_policy_chunks.parquet",
        ]
        if any(not path.is_file() for path in required):
            raise FileNotFoundError(f"episode_{index:06d}: required non-video file missing")
        samples, skipped = build_episode(episode_dir, args)
        if not samples:
            raise RuntimeError(f"episode_{index:06d}: no aligned samples")
        metadata = json.loads((episode_dir / "episode_metadata.json").read_text(encoding="utf-8"))
        payload = {
            "format": "armstrong-qgf-state-action-iql-v1", "episode_index": index,
            "episode_outcome": str(metadata.get("outcome", "unknown")),
            "state": torch.tensor([row["state"] for row in samples], dtype=torch.float32),
            "action_chunk": torch.tensor([row["action_chunk"] for row in samples], dtype=torch.float32),
            "next_state": torch.tensor([row["next_state"] for row in samples], dtype=torch.float32),
            "reward": torch.tensor([row["reward"] for row in samples], dtype=torch.float32),
            "success": torch.tensor([row["success"] for row in samples], dtype=torch.bool),
            "done": torch.tensor([row["done"] for row in samples], dtype=torch.bool),
            "terminated": torch.tensor([row["terminated"] for row in samples], dtype=torch.bool),
            "truncated": torch.tensor([row["truncated"] for row in samples], dtype=torch.bool),
        }
        if tuple(payload["state"].shape[1:]) != (8,) or tuple(payload["action_chunk"].shape[1:]) != (50, 8):
            raise ValueError(f"episode_{index:06d}: unexpected state/action shape")
        output = args.output_dir / f"episode_{index:06d}.pt"
        torch.save(payload, output)
        total_skipped.update(skipped)
        row = {"episode_index": index, "outcome": payload["episode_outcome"], "samples": len(samples), "skipped": dict(skipped), "path": output.name}
        report.append(row); print(json.dumps(row), flush=True)
    summary = {
        "format": "armstrong-qgf-state-action-iql-v1", "ablation": "no_visual_Q_s_a",
        "raw_episode_root": str(args.raw_episodes_root), "episode_range": [args.episode_first, args.episode_last],
        "episodes": report, "episode_count": len(report), "sample_count": sum(row["samples"] for row in report),
        "alignment": {"action_horizon": args.action_horizon, "policy_hz": args.policy_hz, "max_transition_gap_seconds": args.max_transition_gap_seconds},
        "skipped": dict(total_skipped), "provenance": "Camera files were intentionally neither required nor read.",
    }
    (args.output_dir / "state_action_cache_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
