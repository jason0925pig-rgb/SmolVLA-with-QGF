#!/usr/bin/env python3
"""Offline validation for a trained one-arm SmolVLA checkpoint.

This script never imports ROS and never sends a robot command.  It loads a
checkpoint and deterministic frames from the recorded LeRobot dataset, checks
that preprocessing, flow-matching loss, action-chunk inference and
postprocessing are all finite, and writes a machine-readable report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies import SmolVLAConfig  # noqa: F401 - registers config type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/onearm_tele")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def finite_scalar(value: torch.Tensor) -> float:
    value = value.detach().float().cpu()
    if value.numel() != 1 or not torch.isfinite(value).all():
        raise RuntimeError(f"expected one finite scalar, got {value}")
    return float(value.item())


def deterministic_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        raise ValueError("dataset is empty")
    count = min(max(1, count), length)
    if count == 1:
        return [length // 2]
    return sorted({round(index * (length - 1) / (count - 1)) for index in range(count)})


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            output[key] = torch.stack(values)
        else:
            output[key] = values
    return output


def main() -> int:
    args = parse_args()
    started = time.time()
    checkpoint = args.checkpoint.resolve()
    dataset_root = args.dataset_root.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(dataset_root / "meta" / "info.json")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA validation requested but torch.cuda.is_available() is false")

    config = PreTrainedConfig.from_pretrained(checkpoint)
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(checkpoint, config=config)
    policy.to(args.device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    metadata = LeRobotDatasetMetadata(args.repo_id, root=dataset_root)
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=dataset_root,
        delta_timestamps=resolve_delta_timestamps(config, metadata),
        video_backend="torchcodec",
    )
    indices = deterministic_indices(len(dataset), args.samples)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    losses: list[float] = []
    action_min: list[float] | None = None
    action_max: list[float] | None = None
    chunk_shapes: set[tuple[int, ...]] = set()
    joint_absolute_errors: list[float] = []
    gripper_correct = 0
    gripper_total = 0
    batches = 0
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    with torch.inference_mode():
        for raw_batch in loader:
            demonstrated_actions = raw_batch["action"].detach().float().cpu()
            batch = preprocessor(raw_batch)
            loss, _ = policy.forward(batch)
            losses.append(finite_scalar(loss))

            observation = {key: value for key, value in batch.items() if key != "action"}
            chunk = policy.predict_action_chunk(observation)
            if chunk.ndim != 3:
                raise RuntimeError(f"expected BxTxA action chunk, got {tuple(chunk.shape)}")
            if not torch.isfinite(chunk).all():
                raise RuntimeError("normalized action chunk contains NaN or infinity")
            chunk_shapes.add(tuple(int(value) for value in chunk.shape))

            processed = []
            for step in range(chunk.shape[1]):
                processed.append(postprocessor(chunk[:, step, :]))
            actions = torch.stack(processed, dim=1).detach().float().cpu()
            if not torch.isfinite(actions).all():
                raise RuntimeError("postprocessed action chunk contains NaN or infinity")
            if demonstrated_actions.ndim == 2:
                demonstrated_actions = demonstrated_actions.unsqueeze(1)
            common_steps = min(actions.shape[1], demonstrated_actions.shape[1])
            if actions.shape[0] != demonstrated_actions.shape[0] or common_steps <= 0:
                raise RuntimeError(
                    "predicted and demonstrated action batches cannot be compared: "
                    f"{tuple(actions.shape)} vs {tuple(demonstrated_actions.shape)}"
                )
            predicted_common = actions[:, :common_steps, :]
            demonstrated_common = demonstrated_actions[:, :common_steps, :]
            joint_absolute_errors.extend(
                (predicted_common[..., :7] - demonstrated_common[..., :7])
                .abs()
                .reshape(-1)
                .tolist()
            )
            gripper_correct += int(
                (
                    (predicted_common[..., 7] >= 0.5)
                    == (demonstrated_common[..., 7] >= 0.5)
                ).sum()
            )
            gripper_total += int(predicted_common[..., 7].numel())
            current_min = actions.amin(dim=(0, 1)).tolist()
            current_max = actions.amax(dim=(0, 1)).tolist()
            if action_min is None:
                action_min = current_min
                action_max = current_max
            else:
                action_min = [min(a, b) for a, b in zip(action_min, current_min, strict=True)]
                action_max = [max(a, b) for a, b in zip(action_max, current_max, strict=True)]
            batches += 1

    if not losses or action_min is None or action_max is None:
        raise RuntimeError("validation produced no batches")
    if any(not math.isfinite(value) for value in losses + action_min + action_max):
        raise RuntimeError("validation report would contain a non-finite number")

    dataset_stats = json.loads(
        (dataset_root / "meta" / "stats.json").read_text(encoding="utf-8")
    )
    demonstrated_min = [float(value) for value in dataset_stats["action"]["min"]]
    demonstrated_max = [float(value) for value in dataset_stats["action"]["max"]]
    lower_overshoot = [
        max(0.0, lower - predicted)
        for predicted, lower in zip(action_min, demonstrated_min, strict=True)
    ]
    upper_overshoot = [
        max(0.0, predicted - upper)
        for predicted, upper in zip(action_max, demonstrated_max, strict=True)
    ]
    maximum_overshoot = max(lower_overshoot + upper_overshoot)

    report = {
        "status": "ok",
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "dataset_frames": len(dataset),
        "sample_indices": indices,
        "batch_count": batches,
        "loss": {
            "mean": statistics.fmean(losses),
            "min": min(losses),
            "max": max(losses),
            "values": losses,
        },
        "action_chunk_shapes": [list(shape) for shape in sorted(chunk_shapes)],
        "postprocessed_action_min": action_min,
        "postprocessed_action_max": action_max,
        "demonstrated_action_min": demonstrated_min,
        "demonstrated_action_max": demonstrated_max,
        "lower_overshoot": lower_overshoot,
        "upper_overshoot": upper_overshoot,
        "maximum_demonstrated_envelope_overshoot": maximum_overshoot,
        "joint_action_mae_rad": statistics.fmean(joint_absolute_errors),
        "joint_action_max_abs_error_rad": max(joint_absolute_errors),
        "gripper_binary_accuracy": gripper_correct / gripper_total,
        "gripper_samples": gripper_total,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_device_name": (
            torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu"
        ),
        "elapsed_seconds": time.time() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
