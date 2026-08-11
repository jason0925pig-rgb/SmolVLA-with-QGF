#!/usr/bin/env python
"""Precompute task-description features for collected rollout episodes."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _parse_task_ids(value: str) -> list[int] | None:
    if value.lower() in {"", "none", "all"}:
        return None
    parsed = json.loads(value)
    if isinstance(parsed, int):
        return [parsed]
    if not isinstance(parsed, list) or not all(isinstance(x, int) for x in parsed):
        raise ValueError("--task-ids must be a JSON integer list, e.g. '[0, 1]'.")
    return parsed


def _episode_files(input_dir: str | Path):
    input_path = Path(input_dir)
    episodes_dir = input_path / "episodes"
    if episodes_dir.exists():
        return sorted(episodes_dir.glob("episode_*.pt"))
    return sorted(input_path.rglob("episode_*.pt"))


def precompute_episode_features(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    feature_key: str,
    feature_fn,
    max_episodes: int | None = None,
) -> int:
    import torch

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_episodes_dir = output_dir / "episodes"
    output_episodes_dir.mkdir(parents=True, exist_ok=True)

    summary_path = input_dir / "summary.json"
    if summary_path.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary_path, output_dir / "summary.json")

    episode_files = _episode_files(input_dir)
    if max_episodes is not None:
        episode_files = episode_files[:max_episodes]
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.pt files found under {input_dir}.")

    for episode_file in episode_files:
        episode = torch.load(episode_file, map_location="cpu", weights_only=False)
        if "task_tokens" not in episode or "task_attention_mask" not in episode:
            raise ValueError(
                f"{episode_file} does not contain task_tokens and task_attention_mask."
            )
        feature = feature_fn(
            episode["task_tokens"].unsqueeze(0),
            episode["task_attention_mask"].unsqueeze(0),
        )
        episode[feature_key] = feature.squeeze(0).detach().to("cpu", dtype=torch.float32)
        torch.save(episode, output_episodes_dir / episode_file.name)

    return len(episode_files)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--env-type", default="libero")
    parser.add_argument("--task", default="libero_spatial")
    parser.add_argument("--task-ids", default="[0]")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-key", default="task_vlm_hidden")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument(
        "--rename-map",
        default='{"observation.images.image":"observation.images.camera1",'
        '"observation.images.image2":"observation.images.camera2"}',
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    import torch
    from lerobot.configs import PreTrainedConfig
    from lerobot.envs.factory import make_env_config
    from lerobot.policies import make_policy
    from lerobot.utils.import_utils import register_third_party_plugins
    from lerobot.utils.utils import init_logging

    from guided_action_flow.training.task_features import smolvla_text_hidden_features

    init_logging()
    register_third_party_plugins()

    task_ids = _parse_task_ids(args.task_ids)
    rename_map = json.loads(args.rename_map)
    env_cfg = make_env_config(
        args.env_type,
        task=args.task,
        task_ids=task_ids,
        max_parallel_tasks=1,
    )
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = Path(args.policy_path)
    policy_cfg.device = args.device
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=rename_map)
    policy.eval()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    def feature_fn(tokens, mask):
        return smolvla_text_hidden_features(
            policy.model,
            tokens.to(device=device),
            mask.to(device=device),
        )

    written = precompute_episode_features(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        feature_key=args.feature_key,
        feature_fn=feature_fn,
        max_episodes=args.max_episodes,
    )
    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "policy_path": str(args.policy_path),
        "env_type": args.env_type,
        "task": args.task,
        "task_ids": task_ids,
        "feature_key": args.feature_key,
        "feature_source": "vlm_hidden",
        "num_episodes": written,
    }
    with open(Path(args.output_dir) / "task_feature_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
