#!/usr/bin/env python3
"""Freeze the deployed SmolVLA vision encoder and cache two-camera features.

Input is the chunk-level manifest made by
``build_real_robot_visual_iql_manifest.py``.  Each sample is one *exact*
normalized 50-step policy action chunk paired with the chest and right-wrist
RGB frame that was available when the policy made that prediction.

The output is one compact ``episode_XXXXXX.pt`` per source episode.  It keeps
the raw normalized action chunk, real state transition and terminal labels, but
replaces RGB pixels with frozen SmolVLA visual tokens.  The online QGF critic
can reproduce those tokens with the same deployed policy vision encoder.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-episodes-root", type=Path, required=True)
    parser.add_argument("--aligned-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--episode-first", type=int)
    parser.add_argument("--episode-last", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _decode_selected_rgb_frames(path: Path, indices: set[int]):
    """Decode selected MP4 frame numbers once, retaining RGB tensors on CPU."""

    import av
    import torch

    if not indices:
        return {}
    decoded = {}
    with av.open(str(path)) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index not in indices:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            decoded[frame_index] = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
            if len(decoded) == len(indices):
                break
    missing = sorted(indices - set(decoded))
    if missing:
        raise RuntimeError(f"{path}: requested frame indices were not decodable: {missing[:10]}")
    return decoded


def _encode_pairs(policy, pairs, *, batch_size: int):
    """Encode ``[(chest_rgb, wrist_rgb), ...]`` into [N, 128, 960] BF16 tokens."""

    import torch

    if batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    device = next(policy.parameters()).device
    outputs = []
    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        chest = torch.stack([pair[0] for pair in batch_pairs]).to(device=device, dtype=torch.float32)
        wrist = torch.stack([pair[1] for pair in batch_pairs]).to(device=device, dtype=torch.float32)
        batch = {
            "observation.images.chest": chest / 255.0,
            "observation.images.wrist_right": wrist / 255.0,
        }
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            images, _ = policy.prepare_images(batch)
            chest_tokens = policy.model.vlm_with_expert.embed_image(images[0])
            wrist_tokens = policy.model.vlm_with_expert.embed_image(images[1])
            outputs.append(torch.cat((chest_tokens, wrist_tokens), dim=1).to("cpu", dtype=torch.bfloat16))
    return torch.cat(outputs, dim=0)


def _episode_rows(rows, episode_index: int):
    return [row for row in rows if int(row["episode_index"]) == episode_index]


def _extract_episode(policy, *, rows: list[dict], raw_episodes_root: Path, output_path: Path, batch_size: int):
    import torch

    episode_name = str(rows[0]["episode_name"])
    episode_dir = raw_episodes_root / episode_name
    current_chest = {int(row["chest_frame_index"]) for row in rows}
    current_wrist = {int(row["wrist_frame_index"]) for row in rows}
    next_chest = {int(row["next_chest_frame_index"]) for row in rows if bool(row["next_visual_valid"])}
    next_wrist = {int(row["next_wrist_frame_index"]) for row in rows if bool(row["next_visual_valid"])}
    chest_frames = _decode_selected_rgb_frames(episode_dir / "chest.mp4", current_chest | next_chest)
    wrist_frames = _decode_selected_rgb_frames(episode_dir / "wrist_right.mp4", current_wrist | next_wrist)

    current_pairs = [
        (chest_frames[int(row["chest_frame_index"])], wrist_frames[int(row["wrist_frame_index"])])
        for row in rows
    ]
    next_pairs = [
        (
            chest_frames[int(row["next_chest_frame_index"])],
            wrist_frames[int(row["next_wrist_frame_index"])],
        )
        if bool(row["next_visual_valid"])
        else current_pairs[index]
        for index, row in enumerate(rows)
    ]
    visual = _encode_pairs(policy, current_pairs, batch_size=batch_size)
    next_visual = _encode_pairs(policy, next_pairs, batch_size=batch_size)
    next_valid = torch.tensor([bool(row["next_visual_valid"]) for row in rows], dtype=torch.bool)
    next_visual[~next_valid] = 0
    payload = {
        "format": "armstrong-qgf-visual-iql-v1",
        "episode_index": int(rows[0]["episode_index"]),
        "episode_name": episode_name,
        "episode_outcome": str(rows[0]["episode_outcome"]),
        "task": str(rows[0]["task"]),
        "state": torch.tensor([row["state"] for row in rows], dtype=torch.float32),
        "action_chunk": torch.tensor(
            [row["action_chunk_normalized"] for row in rows], dtype=torch.float32
        ),
        "next_state": torch.tensor([row["next_state"] for row in rows], dtype=torch.float32),
        "reward": torch.tensor([row["reward"] for row in rows], dtype=torch.float32),
        "success": torch.tensor([row["success"] for row in rows], dtype=torch.bool),
        "done": torch.tensor([row["done"] for row in rows], dtype=torch.bool),
        "terminated": torch.tensor([row["terminated"] for row in rows], dtype=torch.bool),
        "truncated": torch.tensor([row["truncated"] for row in rows], dtype=torch.bool),
        "visual_features": visual,
        "next_visual_features": next_visual,
        "next_visual_valid": next_valid,
        "provenance": {
            "visual_encoder": "frozen deployed SmolVLA vision_model + connector",
            "views": ["chest", "wrist_right"],
            "per_view_tokens": int(visual.shape[1] // 2),
            "visual_feature_dim": int(visual.shape[-1]),
            "dtype": str(visual.dtype),
            "action_space": "normalized SmolVLA flow-sampler output",
            "current_frame_indices": [
                [int(row["chest_frame_index"]), int(row["wrist_frame_index"])] for row in rows
            ],
            "next_frame_indices": [
                [int(row["next_chest_frame_index"]), int(row["next_wrist_frame_index"])] for row in rows
            ],
        },
    }
    torch.save(payload, output_path)
    return {
        "episode_index": payload["episode_index"],
        "episode_name": episode_name,
        "sample_count": int(payload["state"].shape[0]),
        "visual_shape": list(payload["visual_features"].shape),
        "path": output_path.name,
    }


def main() -> None:
    args = _parse_args()
    if args.episode_first is not None and args.episode_last is not None:
        if args.episode_last < args.episode_first:
            raise ValueError("--episode-last must be >= --episode-first.")
    import pyarrow.parquet as pq
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    rows = pq.read_table(args.aligned_manifest).to_pylist()
    by_episode: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        episode_index = int(row["episode_index"])
        if args.episode_first is not None and episode_index < args.episode_first:
            continue
        if args.episode_last is not None and episode_index > args.episode_last:
            continue
        by_episode[episode_index].append(row)
    if not by_episode:
        raise RuntimeError("No samples selected from the aligned manifest.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint)
    policy.eval()
    policy.to(torch.device(args.device))
    report = []
    for episode_index in sorted(by_episode):
        output_path = args.output_dir / f"episode_{episode_index:06d}.pt"
        if output_path.exists() and not args.overwrite:
            report.append({"episode_index": episode_index, "path": output_path.name, "status": "existing"})
            continue
        report.append(
            _extract_episode(
                policy,
                rows=by_episode[episode_index],
                raw_episodes_root=args.raw_episodes_root,
                output_path=output_path,
                batch_size=args.batch_size,
            )
        )
        print(json.dumps(report[-1]), flush=True)
    summary = {
        "format": "armstrong-qgf-visual-iql-v1",
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "episodes": report,
        "episode_count": len(report),
        "sample_count": sum(int(item.get("sample_count", 0)) for item in report),
    }
    (args.output_dir / "visual_feature_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
