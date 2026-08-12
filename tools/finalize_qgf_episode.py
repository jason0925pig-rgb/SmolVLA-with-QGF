#!/usr/bin/env python3
"""Label, validate and atomically commit one staged real-robot QGF episode."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def exclusive_file_lock(stream):
    """Use the Linux dataset lock; permit offline Windows validation."""
    if os.name == "posix":
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    else:
        # Real collection/finalization runs on the Orin.  Windows only uses
        # this branch for local schema tests where no concurrent writer exists.
        yield


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
    return rows


def build_transitions(samples: list[dict[str, Any]], success: bool, task: str) -> list[dict[str, Any]]:
    """Convert samples at t and t+1 to paper-compatible transitions."""
    if len(samples) < 2:
        raise ValueError("an episode needs at least two synchronized samples")
    transitions: list[dict[str, Any]] = []
    last_index = len(samples) - 2
    for index, (current, following) in enumerate(zip(samples, samples[1:], strict=False)):
        terminal = index == last_index
        transitions.append(
            {
                "episode_step": index,
                "timestamp_ns": int(current["timestamp_ns"]),
                "next_timestamp_ns": int(following["timestamp_ns"]),
                "state": [float(x) for x in current["state"]],
                "state_timestamp_ns": int(current.get("state_timestamp_ns", 0)),
                "action_policy": [float(x) for x in current["action_policy"]],
                "policy_action_timestep": int(
                    current.get("policy_action_timestep", -1)
                ),
                "action_policy_timestamp_ns": int(
                    current.get("action_policy_timestamp_ns", 0)
                ),
                "action_guarded": [float(x) for x in current["action_guarded"]],
                "action_guarded_timestamp_ns": int(
                    current.get("action_guarded_timestamp_ns", 0)
                ),
                "action_executed": [float(x) for x in current["action_executed"]],
                "action_executed_timestamp_ns": int(
                    current.get("action_executed_timestamp_ns", 0)
                ),
                "next_state": [float(x) for x in following["state"]],
                "reward": float(success and terminal),
                # Match the existing QGF dataset convention: success becomes
                # true at the successful terminal transition, not throughout
                # the entire episode.  The episode-level label is stored
                # separately as episode_success.
                "success": bool(success and terminal),
                "episode_success": bool(success),
                "terminated": bool(success and terminal),
                "truncated": bool((not success) and terminal),
                "done": terminal,
                "gripper_contact": bool(current.get("gripper_contact", False)),
                "chest_frame_index": int(current.get("chest_frame_index", -1)),
                "chest_timestamp_ns": int(current.get("chest_timestamp_ns", 0)),
                "wrist_frame_index": int(current.get("wrist_frame_index", -1)),
                "wrist_timestamp_ns": int(current.get("wrist_timestamp_ns", 0)),
                "task": task,
            }
        )
    return transitions


def existing_indices(episodes_dir: Path) -> set[int]:
    result: set[int] = set()
    for path in episodes_dir.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"):
        try:
            result.add(int(path.name.rsplit("_", 1)[1]))
        except ValueError:
            continue
    return result


def next_contiguous_index(episodes_dir: Path) -> int:
    used = existing_indices(episodes_dir)
    index = 0
    while index in used:
        index += 1
    return index


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required in the SmolVLA environment") from exc
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")


def validate_capture(staging: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples_path = staging / "samples.jsonl"
    summary_path = staging / "capture_summary.json"
    if not samples_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("capture did not produce samples.jsonl and capture_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    samples = read_jsonl(samples_path)
    chunks_path = staging / "normalized_policy_chunks.jsonl"
    if not chunks_path.is_file():
        raise FileNotFoundError("capture did not produce normalized_policy_chunks.jsonl")
    chunks = read_jsonl(chunks_path)
    policy_observations_path = staging / "policy_observations.jsonl"
    if not policy_observations_path.is_file():
        raise FileNotFoundError("capture did not produce policy_observations.jsonl")
    policy_observations = read_jsonl(policy_observations_path)
    if not summary.get("action_gate_was_enabled"):
        raise ValueError("SmolVLA action gate was never enabled")
    if len(samples) < 2:
        raise ValueError(
            f"only {len(samples)} synchronized samples were captured; "
            f"topic callback counts={summary.get('callback_counts', {})}"
        )
    if not chunks:
        raise ValueError("no normalized SmolVLA action chunk was captured")
    if not policy_observations:
        raise ValueError("no policy observation was captured")
    for name, video in summary.get("videos", {}).items():
        received = int(video.get("frames_received", 0))
        encoded = int(video.get("frames_encoded", 0))
        dropped = int(video.get("frames_dropped", 0))
        fps = float(video.get("source_effective_fps", 0.0))
        if received < 2 or encoded != received or dropped != 0:
            raise ValueError(
                f"{name} video is incomplete: received={received} encoded={encoded} dropped={dropped}"
            )
        if not 27.0 <= fps <= 33.0:
            raise ValueError(f"{name} source FPS {fps:.3f} is outside the required 30 FPS range")
    chunk_timesteps = {int(row["observation_timestep"]) for row in chunks}
    usable_observation_timesteps = {
        int(row["observation_timestep"])
        for row in policy_observations
        if int(row.get("chest_frame_index", -1)) >= 0
        and int(row.get("wrist_frame_index", -1)) >= 0
    }
    if not chunk_timesteps.intersection(usable_observation_timesteps):
        raise ValueError(
            "no normalized action chunk can be joined to a two-camera policy observation"
        )
    for video_name in (
        "chest.mp4",
        "wrist_right.mp4",
        "chest.mjpeg",
        "wrist_right.mjpeg",
    ):
        video = staging / video_name
        if not video.is_file() or video.stat().st_size == 0:
            raise ValueError(f"missing or empty video: {video_name}")
    summary["normalized_policy_chunks"] = len(chunks)
    summary["policy_observations"] = len(policy_observations)
    return samples, summary


def finalize(
    staging: Path,
    dataset_root: Path,
    outcome: str,
    task: str,
    notes: str,
    termination_source: str = "unknown",
) -> Path | None:
    staging = staging.resolve()
    dataset_root = dataset_root.resolve()
    staging_root = (dataset_root / ".staging").resolve()
    if staging_root not in staging.parents:
        raise ValueError(f"refusing staging directory outside {staging_root}")
    if outcome == "discard":
        shutil.rmtree(staging, ignore_errors=True)
        return None

    samples, capture_summary = validate_capture(staging)
    success = outcome == "success"
    transitions = build_transitions(samples, success=success, task=task)
    write_parquet(staging / "transitions.parquet", transitions)
    chunks = read_jsonl(staging / "normalized_policy_chunks.jsonl")
    write_parquet(staging / "normalized_policy_chunks.parquet", chunks)
    policy_observations = read_jsonl(staging / "policy_observations.jsonl")
    write_parquet(staging / "policy_observations.parquet", policy_observations)
    episodes_dir = dataset_root / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    lock_path = dataset_root / ".episode_index.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        with exclusive_file_lock(lock):
            index = next_contiguous_index(episodes_dir)
            episode_name = f"episode_{index:06d}"
            destination = episodes_dir / episode_name
            metadata = {
                "schema_version": "qgf-real-rollout-1.0",
                "episode_index": index,
                "episode_name": episode_name,
                "task": task,
                "outcome": outcome,
                "success": success,
                "reward_definition": (
                    "sparse terminal reward: 1 for labeled success, otherwise 0"
                ),
                "num_samples": len(samples),
                "num_transitions": len(transitions),
                "state_definition": (
                    "7 right-arm joint radians + gripper_closed in [0,1]"
                ),
                "action_policy_definition": (
                    "postprocessed physical SmolVLA target; 7 radians + gripper_closed"
                ),
                "normalized_chunk_definition": (
                    "direct output of SmolVLA predict_action_chunk before postprocessing; "
                    "stored in normalized_policy_chunks.parquet"
                ),
                "policy_observation_definition": (
                    "exact proprioceptive observation keyed by observation_timestep; "
                    "join with normalized_policy_chunks.parquet on observation_timestep"
                ),
                "action_guarded_definition": (
                    "target after task bounds and gripper temporal filter"
                ),
                "action_executed_definition": (
                    "accepted arm command + last executed gripper command"
                ),
                "camera_features": {
                    "chest": "chest.mp4",
                    "wrist_right": "wrist_right.mp4",
                    "chest_exact_jpeg_archive": "chest.mjpeg",
                    "wrist_right_exact_jpeg_archive": "wrist_right.mjpeg",
                },
                "task_prompt": task,
                "notes": notes,
                "termination_source": termination_source,
                "finalized_at_utc": utc_now(),
                "capture": capture_summary,
            }
            (staging / "episode_metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(staging, destination)
            manifest: list[dict[str, Any]] = []
            for episode_dir in sorted(episodes_dir.glob("episode_[0-9]*")):
                metadata_path = episode_dir / "episode_metadata.json"
                if not metadata_path.is_file():
                    continue
                item = json.loads(metadata_path.read_text(encoding="utf-8"))
                manifest.append(
                    {
                        key: item[key]
                        for key in (
                            "episode_index",
                            "episode_name",
                            "task",
                            "outcome",
                            "success",
                            "num_transitions",
                            "finalized_at_utc",
                        )
                    }
                )
            manifest_path = dataset_root / "episodes.jsonl"
            manifest_tmp = dataset_root / ".episodes.jsonl.tmp"
            manifest_tmp.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in manifest),
                encoding="utf-8",
            )
            os.replace(manifest_tmp, manifest_path)
            summary = {
                "schema_version": "qgf-real-rollout-1.0",
                "episodes": len(manifest),
                "successes": sum(bool(item["success"]) for item in manifest),
                "failures": sum(not bool(item["success"]) for item in manifest),
                "transitions": sum(int(item["num_transitions"]) for item in manifest),
                "updated_at_utc": utc_now(),
            }
            summary_tmp = dataset_root / ".dataset_summary.json.tmp"
            summary_tmp.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(summary_tmp, dataset_root / "dataset_summary.json")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--outcome", choices=("success", "failure", "discard"), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--termination-source", default="unknown")
    args = parser.parse_args()
    destination = finalize(
        args.staging,
        args.dataset_root,
        args.outcome,
        args.task,
        args.notes,
        args.termination_source,
    )
    if destination is None:
        print("QGF_EPISODE_DISCARDED")
    else:
        print(f"QGF_EPISODE_SAVED={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
