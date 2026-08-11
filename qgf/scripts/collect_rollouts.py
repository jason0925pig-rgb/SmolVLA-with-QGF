#!/usr/bin/env python
"""Collect real SmolVLA rollouts for critic training."""

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


def _task_descriptions(env) -> list[str]:
    try:
        return list(env.call("task_description"))
    except (AttributeError, NotImplementedError):
        try:
            return list(env.call("task"))
        except (AttributeError, NotImplementedError):
            return [""] * env.num_envs


def _extract_successes(info: dict, num_envs: int) -> list[bool]:
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict) and "is_success" in final_info:
            values = final_info["is_success"]
            return values.tolist() if hasattr(values, "tolist") else list(values)
    if "is_success" in info:
        values = info["is_success"]
        if hasattr(values, "tolist"):
            return values.tolist()
        return [bool(values)] * num_envs
    return [False] * num_envs


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
    parser.add_argument(
        "--rename-map",
        default='{"observation.images.image":"observation.images.camera1",'
        '"observation.images.image2":"observation.images.camera2"}',
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.batch_size != 1:
        raise ValueError("The first collector implementation only supports --batch-size=1.")

    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
    from lerobot.envs.utils import close_envs, preprocess_observation
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.constants import (
        ACTION,
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )
    from lerobot.utils.import_utils import register_third_party_plugins
    from lerobot.utils.random_utils import set_seed

    logging.basicConfig(level=logging.INFO)
    register_third_party_plugins()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

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

    per_episode = []
    episode_index = 0
    start_time = time.time()

    try:
        for task_group, group in envs.items():
            for task_id, env in group.items():
                max_steps = int(env.call("_max_episode_steps")[0])
                for local_episode in range(args.n_episodes):
                    seed = args.seed + episode_index
                    policy.reset()
                    observation, _ = env.reset(seed=[seed])
                    done = np.array([False] * env.num_envs)

                    states = []
                    policy_actions = []
                    env_actions = []
                    rewards = []
                    successes = []
                    dones = []
                    task_desc = _task_descriptions(env)[0]
                    task_tokens = None
                    task_attention_mask = None

                    for _step in range(max_steps):
                        observation = preprocess_observation(observation)
                        observation["task"] = _task_descriptions(env)
                        observation = env_preprocessor(observation)
                        policy_batch = preprocessor(observation)

                        state = policy_batch[OBS_STATE].detach().to("cpu")[0]
                        if task_tokens is None and OBS_LANGUAGE_TOKENS in policy_batch:
                            task_tokens = policy_batch[OBS_LANGUAGE_TOKENS].detach().to("cpu")[0]
                            task_attention_mask = (
                                policy_batch[OBS_LANGUAGE_ATTENTION_MASK]
                                .detach()
                                .to("cpu")[0]
                            )
                        with torch.inference_mode():
                            policy_action = policy.select_action(policy_batch)
                        action = postprocessor(policy_action)
                        action_transition = env_postprocessor({ACTION: action})
                        action = action_transition[ACTION]

                        action_numpy = action.to("cpu").numpy()
                        observation, reward, terminated, truncated, info = env.step(action_numpy)

                        step_successes = _extract_successes(info, env.num_envs)
                        done = terminated | truncated | done
                        if len(successes) == 0 and any(step_successes):
                            task_desc = _task_descriptions(env)[0]

                        states.append(state)
                        policy_actions.append(policy_action.detach().to("cpu")[0])
                        env_actions.append(action.detach().to("cpu")[0])
                        rewards.append(float(reward[0]))
                        successes.append(bool(step_successes[0]))
                        dones.append(bool(done[0]))

                        if bool(done[0]):
                            break

                    episode = {
                        "state": torch.stack(states),
                        "action_policy": torch.stack(policy_actions),
                        "action_env": torch.stack(env_actions),
                        "reward": torch.tensor(rewards, dtype=torch.float32),
                        "success": torch.tensor(successes, dtype=torch.bool),
                        "done": torch.tensor(dones, dtype=torch.bool),
                        "task": task_desc,
                        "task_group": task_group,
                        "task_id": int(task_id),
                        "seed": int(seed),
                        "policy_path": str(args.policy_path),
                        "obs_feature_kind": "policy_preprocessed_observation.state",
                        "action_policy_kind": "pre_postprocessor_smolvla_action",
                    }
                    if task_tokens is not None and task_attention_mask is not None:
                        episode["task_tokens"] = task_tokens
                        episode["task_attention_mask"] = task_attention_mask
                    episode_path = episodes_dir / f"episode_{episode_index:06d}.pt"
                    torch.save(episode, episode_path)

                    success = bool(torch.as_tensor(successes).any().item())
                    per_episode.append(
                        {
                            "episode_index": episode_index,
                            "task_group": task_group,
                            "task_id": int(task_id),
                            "local_episode": local_episode,
                            "seed": seed,
                            "num_steps": len(rewards),
                            "sum_reward": float(sum(rewards)),
                            "success": success,
                            "path": str(episode_path),
                        }
                    )
                    print(
                        f"episode={episode_index} task={task_group}/{task_id} "
                        f"steps={len(rewards)} success={success}"
                    )
                    episode_index += 1
    finally:
        close_envs(envs)

    summary = {
        "config": {
            "policy_path": str(args.policy_path),
            "env_type": args.env_type,
            "task": args.task,
            "task_ids": task_ids,
            "n_episodes": args.n_episodes,
            "seed": args.seed,
            "device": args.device,
            "rename_map": rename_map,
        },
        "per_episode": per_episode,
        "aggregated": {
            "n_episodes": len(per_episode),
            "pc_success": 100.0
            * sum(1 for episode in per_episode if episode["success"])
            / max(1, len(per_episode)),
            "avg_sum_reward": sum(episode["sum_reward"] for episode in per_episode)
            / max(1, len(per_episode)),
            "collect_s": time.time() - start_time,
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
