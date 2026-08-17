#!/usr/bin/env python3
"""Build aligned QGF action-chunk samples from real Armstrong rollout episodes.

The robot recorder stores three complementary streams per episode:

* ``normalized_policy_chunks.parquet``: exact 50-step actions produced by
  SmolVLA, in the sampler's normalized action space;
* ``policy_observations.parquet``: state and two-camera frame indices at the
  observation used to produce each chunk;
* ``transitions.parquet``: executed real-robot state / reward / terminal data.

This script joins those streams without touching the robot.  It deliberately
uses the normalized action chunks, rather than postprocessed physical commands:
QGF differentiates through the normalized action chunk at inference time, so
training must use that same action space.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AlignmentConfig:
    action_horizon: int
    policy_hz: float
    max_transition_gap_seconds: float


def _nearest_index(sorted_timestamps, timestamp: int) -> tuple[int, int]:
    """Return ``(index, absolute_gap_ns)`` for a sorted integer timestamp list."""

    import numpy as np

    values = np.asarray(sorted_timestamps, dtype=np.int64)
    pos = int(np.searchsorted(values, timestamp, side="left"))
    candidates = [index for index in (pos - 1, pos) if 0 <= index < len(values)]
    if not candidates:
        raise ValueError("Cannot align against an empty timestamp stream.")
    index = min(candidates, key=lambda candidate: abs(int(values[candidate]) - timestamp))
    return index, abs(int(values[index]) - timestamp)


def _valid_camera_pair(row: dict) -> bool:
    return int(row["chest_frame_index"]) >= 0 and int(row["wrist_frame_index"]) >= 0


def _choose_observation(rows: list[dict], chunk_timestamp_ns: int) -> dict | None:
    rows = [row for row in rows if _valid_camera_pair(row)]
    if not rows:
        return None
    return min(rows, key=lambda row: abs(int(row["timestamp_ns"]) - chunk_timestamp_ns))


def _rows_by_observation_timestep(rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["observation_timestep"]), []).append(row)
    return grouped


def _load_episode_samples(episode_dir: Path, config: AlignmentConfig) -> tuple[list[dict], Counter]:
    import pyarrow.parquet as pq

    chunks = pq.read_table(episode_dir / "normalized_policy_chunks.parquet").to_pylist()
    observations = pq.read_table(episode_dir / "policy_observations.parquet").to_pylist()
    transitions = pq.read_table(episode_dir / "transitions.parquet").to_pylist()
    if not transitions:
        raise ValueError(f"{episode_dir}: transitions.parquet is empty.")

    metadata = json.loads((episode_dir / "episode_metadata.json").read_text(encoding="utf-8"))
    episode_index = int(metadata["episode_index"])
    task = str(metadata.get("task_prompt") or metadata.get("task") or "")
    outcome = str(metadata.get("outcome", "unknown"))
    observation_rows = _rows_by_observation_timestep(observations)
    transition_timestamps = [int(row["timestamp_ns"]) for row in transitions]
    max_gap_ns = int(config.max_transition_gap_seconds * 1_000_000_000)
    horizon_ns = int(round(config.action_horizon / config.policy_hz * 1_000_000_000))
    samples: list[dict] = []
    skipped: Counter = Counter()

    for chunk in chunks:
        action_chunk = chunk["action_chunk_normalized"]
        if len(action_chunk) != config.action_horizon or any(len(action) != 8 for action in action_chunk):
            skipped["invalid_action_chunk_shape"] += 1
            continue

        chunk_timestamp = int(chunk["timestamp_ns"])
        observation = _choose_observation(
            observation_rows.get(int(chunk["observation_timestep"]), []),
            chunk_timestamp,
        )
        if observation is None:
            skipped["missing_aligned_policy_observation"] += 1
            continue

        start_timestamp = int(observation["timestamp_ns"])
        start_index, start_gap = _nearest_index(transition_timestamps, start_timestamp)
        if start_gap > max_gap_ns:
            skipped["start_transition_too_far"] += 1
            continue

        target_timestamp = start_timestamp + horizon_ns
        end_index, _ = _nearest_index(transition_timestamps, target_timestamp)
        if target_timestamp >= transition_timestamps[-1]:
            end_index = len(transitions) - 1
        if end_index < start_index:
            skipped["invalid_transition_order"] += 1
            continue

        span = transitions[start_index : end_index + 1]
        terminal_row = next((row for row in span if bool(row["done"])), None)
        if terminal_row is not None:
            end_index = start_index + span.index(terminal_row)
            span = transitions[start_index : end_index + 1]
        end_row = transitions[end_index]
        done = bool(any(row["done"] for row in span))
        success = bool(any(row["success"] for row in span))
        reward = float(sum(float(row["reward"]) for row in span))
        terminated = bool(any(row["terminated"] for row in span))
        truncated = bool(any(row["truncated"] for row in span))

        next_visual_valid = _valid_camera_pair(end_row)
        samples.append(
            {
                "episode_index": episode_index,
                "episode_name": episode_dir.name,
                "episode_outcome": outcome,
                "task": task,
                "chunk_index": int(chunk["chunk_index"]),
                "observation_timestep": int(chunk["observation_timestep"]),
                "chunk_timestamp_ns": chunk_timestamp,
                "state": observation["state"],
                "action_chunk_normalized": action_chunk,
                "next_state": end_row["next_state"],
                "reward": reward,
                "success": success,
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
                "chest_frame_index": int(observation["chest_frame_index"]),
                "wrist_frame_index": int(observation["wrist_frame_index"]),
                "next_chest_frame_index": int(end_row["chest_frame_index"]) if next_visual_valid else -1,
                "next_wrist_frame_index": int(end_row["wrist_frame_index"]) if next_visual_valid else -1,
                "next_visual_valid": next_visual_valid,
                "start_transition_gap_ns": start_gap,
            }
        )
    return samples, skipped


def _stratified_episode_split(outcomes: dict[int, str], *, val_count: int, seed: int) -> dict:
    import random

    if val_count < 1 or val_count >= len(outcomes):
        raise ValueError("val_count must be between 1 and the number of episodes minus one.")
    groups: dict[str, list[int]] = {}
    for episode_index, outcome in outcomes.items():
        groups.setdefault(outcome, []).append(episode_index)
    rng = random.Random(seed)
    for indices in groups.values():
        rng.shuffle(indices)

    total = len(outcomes)
    selected: list[int] = []
    quotas: dict[str, int] = {}
    for outcome, indices in groups.items():
        quota = int(round(val_count * len(indices) / total))
        quota = min(quota, max(0, len(indices) - 1))
        quotas[outcome] = quota
        selected.extend(indices[:quota])
    remaining = val_count - len(selected)
    if remaining > 0:
        candidates = [index for indices in groups.values() for index in indices if index not in selected]
        rng.shuffle(candidates)
        selected.extend(candidates[:remaining])
    selected = sorted(selected[:val_count])
    train = sorted(index for index in outcomes if index not in set(selected))
    return {
        "strategy": "episode-level stratified by recorded outcome",
        "seed": seed,
        "train_episode_indices": train,
        "val_episode_indices": selected,
        "outcome_counts": dict(Counter(outcomes.values())),
        "val_outcome_counts": dict(Counter(outcomes[index] for index in selected)),
        "train_outcome_counts": dict(Counter(outcomes[index] for index in train)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-episodes-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-first", type=int, default=17)
    parser.add_argument("--episode-last", type=int, default=116)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--policy-hz", type=float, default=15.0)
    parser.add_argument("--max-transition-gap-seconds", type=float, default=0.1)
    parser.add_argument("--val-count", type=int, default=10)
    parser.add_argument("--split-seed", type=int, default=20260814)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.episode_last < args.episode_first:
        raise ValueError("--episode-last must be >= --episode-first.")
    config = AlignmentConfig(
        action_horizon=args.action_horizon,
        policy_hz=args.policy_hz,
        max_transition_gap_seconds=args.max_transition_gap_seconds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_samples: list[dict] = []
    outcomes: dict[int, str] = {}
    issues: list[str] = []
    skipped = Counter()
    for episode_index in range(args.episode_first, args.episode_last + 1):
        episode_dir = args.raw_episodes_root / f"episode_{episode_index:06d}"
        required = [
            episode_dir / "episode_metadata.json",
            episode_dir / "transitions.parquet",
            episode_dir / "normalized_policy_chunks.parquet",
            episode_dir / "policy_observations.parquet",
            episode_dir / "chest.mp4",
            episode_dir / "wrist_right.mp4",
        ]
        if not all(path.is_file() for path in required):
            issues.append(f"episode_{episode_index:06d}: missing required raw file")
            continue
        metadata = json.loads((episode_dir / "episode_metadata.json").read_text(encoding="utf-8"))
        outcomes[episode_index] = str(metadata.get("outcome", "unknown"))
        samples, episode_skipped = _load_episode_samples(episode_dir, config)
        if not samples:
            issues.append(f"episode_{episode_index:06d}: produced no aligned chunks")
        all_samples.extend(samples)
        skipped.update(episode_skipped)

    expected_episodes = args.episode_last - args.episode_first + 1
    if len(outcomes) != expected_episodes:
        raise RuntimeError("Cannot make a 90/10 split: one or more requested episodes are incomplete.")
    if issues:
        raise RuntimeError("\n".join(issues))

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(all_samples)
    manifest_path = args.output_dir / "aligned_normalized_chunks.parquet"
    pq.write_table(table, manifest_path, compression="zstd")
    split = _stratified_episode_split(outcomes, val_count=args.val_count, seed=args.split_seed)
    split_path = args.output_dir / "episode_split_90_10.json"
    split_path.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "format": "armstrong-qgf-visual-chunks-v1",
        "raw_episode_range": [args.episode_first, args.episode_last],
        "raw_episode_count": len(outcomes),
        "aligned_chunk_count": len(all_samples),
        "action_chunk_shape": [args.action_horizon, 8],
        "state_dim": 8,
        "action_space": "exact normalized SmolVLA policy output",
        "visual_input": "two RGB frames indexed from chest.mp4 and wrist_right.mp4",
        "alignment": {
            "policy_hz": args.policy_hz,
            "max_transition_gap_seconds": args.max_transition_gap_seconds,
            "skipped": dict(skipped),
        },
        "split_file": split_path.name,
        "sample_manifest": manifest_path.name,
        "outcome_counts": dict(Counter(outcomes.values())),
    }
    (args.output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
