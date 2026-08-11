#!/usr/bin/env python
"""Evaluate SmolVLA baseline or in-loop QGF on LIBERO."""

from __future__ import annotations

import argparse
import json
import logging
import time
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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-type", default="libero")
    parser.add_argument("--task", default="libero_spatial")
    parser.add_argument("--task-ids", default="[0]")
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-async-envs", action="store_true")
    parser.add_argument("--max-videos", type=int, default=10)
    parser.add_argument("--critic-path")
    parser.add_argument("--critic-paths", nargs="+")
    parser.add_argument("--qgf-beta", type=float, default=10.0)
    parser.add_argument("--qgf-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--qgf-uncertainty-scale", type=float, default=0.0)
    parser.add_argument("--qgf-min-gate", type=float, default=0.0)
    parser.add_argument(
        "--rename-map",
        default='{"observation.images.image":"observation.images.camera1",'
        '"observation.images.image2":"observation.images.camera2"}',
    )
    return parser


def _resolve_critic_paths(args) -> list[str]:
    critic_paths = []
    if args.critic_path:
        critic_paths.append(args.critic_path)
    if args.critic_paths:
        critic_paths.extend(args.critic_paths)
    return critic_paths


def _parse_rename_map(value: str) -> dict[str, str]:
    aliases = {
        "hollytan_current": {
            "observation.images.image": "observation.images.camera1",
            "observation.images.wrist_image": "observation.images.camera2",
        },
        "hollytan_reversed": {
            "observation.images.wrist_image": "observation.images.camera1",
            "observation.images.image": "observation.images.camera2",
        },
    }
    if value in aliases:
        return aliases[value]
    return json.loads(value)


def _infer_task_feature_source(critics, critic_checkpoints) -> str:
    if not critics:
        return "none"

    task_feature_dim = critics[0].config.task_feature_dim
    if task_feature_dim == 0:
        return "none"

    sources = []
    for checkpoint in critic_checkpoints:
        # Backward compatibility: older task-conditioned checkpoints only used
        # hashed tokenizer features and did not store the source explicitly.
        sources.append(checkpoint.get("task_feature_source", "tokens"))
    if len(set(sources)) != 1:
        raise ValueError(f"All QGF critics must use the same task feature source: {sources}")

    source = sources[0]
    if source == "episode":
        raise ValueError(
            "task_feature_source='episode' is train-only; use a runtime source such as "
            "'tokens' or 'vlm_hidden' in critic checkpoints."
        )
    return source


def _summarize_qgf_diagnostics(processor) -> dict[str, float]:
    if processor is None or not processor.diagnostics:
        return {}
    keys = processor.diagnostics[0].keys()
    summary = {}
    for key in keys:
        values = [float(step[key].item()) for step in processor.diagnostics]
        summary[key] = sum(values) / len(values)
    summary["num_guided_denoise_steps"] = float(len(processor.diagnostics))
    return summary


def main() -> None:
    args = _build_arg_parser().parse_args()

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
    from lerobot.envs.utils import close_envs
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy_all
    from lerobot.utils.import_utils import register_third_party_plugins
    from lerobot.utils.random_utils import set_seed

    from guided_action_flow.critics import load_action_chunk_critic
    from guided_action_flow.guidance import QGuidanceConfig
    from guided_action_flow.policies import install_smolvla_qgf

    logging.basicConfig(level=logging.INFO)
    register_third_party_plugins()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_ids = _parse_task_ids(args.task_ids)
    rename_map = _parse_rename_map(args.rename_map)
    env_cfg = make_env_config(
        args.env_type,
        task=args.task,
        task_ids=task_ids,
        max_parallel_tasks=1,
    )
    envs = make_env(env_cfg, n_envs=args.batch_size, use_async_envs=args.use_async_envs)

    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = Path(args.policy_path)
    policy_cfg.device = args.device
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=rename_map)
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": rename_map},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=policy_cfg,
    )

    qgf_processor = None
    critic_paths = _resolve_critic_paths(args)
    if critic_paths:
        critics = []
        critic_checkpoints = []
        for critic_path in critic_paths:
            critic, critic_checkpoint = load_action_chunk_critic(
                critic_path,
                device=str(policy.config.device),
            )
            critics.append(critic)
            critic_checkpoints.append(critic_checkpoint)

        critic_action_dim = critics[0].config.action_dim
        if any(critic.config.action_dim != critic_action_dim for critic in critics):
            raise ValueError("All QGF critics must have the same action dimension.")
        critic_task_feature_dim = critics[0].config.task_feature_dim
        if any(critic.config.task_feature_dim != critic_task_feature_dim for critic in critics):
            raise ValueError("All QGF critics must have the same task feature dimension.")
        critic_task_feature_source = _infer_task_feature_source(critics, critic_checkpoints)
        qgf_processor = install_smolvla_qgf(
            policy,
            critic=critics[0] if len(critics) == 1 else critics,
            config=QGuidanceConfig(
                beta=args.qgf_beta,
                grad_clip_norm=args.qgf_grad_clip_norm,
                uncertainty_scale=args.qgf_uncertainty_scale,
                min_gate=args.qgf_min_gate,
            ),
            critic_action_dim=critic_action_dim,
            task_feature_dim=critic_task_feature_dim,
            task_feature_source=critic_task_feature_source,
        )
    else:
        critic_checkpoints = []

    videos_dir = output_dir / "videos" if args.max_videos > 0 else None
    start_time = time.time()
    try:
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=args.n_episodes,
            max_episodes_rendered=args.max_videos,
            videos_dir=videos_dir,
            return_episode_data=False,
            start_seed=args.seed,
            max_parallel_tasks=1,
        )
    finally:
        close_envs(envs)

    info["run_config"] = {
        "policy_path": str(args.policy_path),
        "env_type": args.env_type,
        "task": args.task,
        "task_ids": task_ids,
        "n_episodes": args.n_episodes,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "rename_map": rename_map,
        "eval_s_wall": time.time() - start_time,
    }
    if critic_paths:
        info["qgf"] = {
            "critic_paths": [str(path) for path in critic_paths],
            "ensemble_size": len(critic_paths),
            "beta": args.qgf_beta,
            "grad_clip_norm": args.qgf_grad_clip_norm,
            "uncertainty_scale": args.qgf_uncertainty_scale,
            "min_gate": args.qgf_min_gate,
            "critic_configs": [
                checkpoint["critic_config"] for checkpoint in critic_checkpoints
            ],
            "task_feature_source": critic_task_feature_source,
            "selected_epochs": [
                checkpoint.get("selected_epoch") for checkpoint in critic_checkpoints
            ],
            "diagnostics": _summarize_qgf_diagnostics(qgf_processor),
        }

    with open(output_dir / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info["overall"], indent=2))


if __name__ == "__main__":
    main()
