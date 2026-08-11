#!/usr/bin/env python
"""Small true-simulator counterfactual probe for LIBERO + SmolVLA.

The script intentionally keeps the first experiment small:

1. Roll a baseline policy to selected decision steps.
2. Save a full MuJoCo / robosuite snapshot.
3. Restore the same state and evaluate A/B/C branches:
   A = baseline prefix, B = QGF prefix, C = noisy baseline prefix.
4. Train a tiny pairwise ranking critic from the branch outcomes.

This is a proof-of-feasibility tool, not the final large-scale collector.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


def _parse_int_list(value: str) -> list[int]:
    parsed = json.loads(value)
    if isinstance(parsed, int):
        return [parsed]
    if not isinstance(parsed, list) or not all(isinstance(x, int) for x in parsed):
        raise ValueError("Expected a JSON int or int list.")
    return parsed


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
    parser.add_argument("--task-ids", default="[3]")
    parser.add_argument("--seed", type=int, default=3100)
    parser.add_argument("--n-episodes", type=int, default=2)
    parser.add_argument("--branch-steps", default="[20, 40, 60]")
    parser.add_argument("--prefix-steps", type=int, default=12)
    parser.add_argument("--tail-max-steps", type=int, default=220)
    parser.add_argument("--noise-std", type=float, default=0.45)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--critic-paths", nargs="*")
    parser.add_argument("--qgf-beta", type=float, default=3.0)
    parser.add_argument("--qgf-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--qgf-uncertainty-scale", type=float, default=0.0)
    parser.add_argument("--qgf-min-gate", type=float, default=0.0)
    parser.add_argument("--ranking-epochs", type=int, default=80)
    parser.add_argument("--ranking-lr", type=float, default=1.0e-3)
    parser.add_argument("--ranking-hidden-dim", type=int, default=256)
    parser.add_argument("--ranking-depth", type=int, default=2)
    parser.add_argument("--skip-ranking", action="store_true")
    parser.add_argument("--save-every-records", type=int, default=9)
    parser.add_argument("--score-step-bonus", type=float, default=0.05)
    parser.add_argument(
        "--rename-map",
        default='{"observation.images.image":"observation.images.camera1",'
        '"observation.images.image2":"observation.images.camera2"}',
    )
    return parser


def _simple_state(obj: Any) -> dict[str, tuple[str, Any]]:
    import numpy as np

    state: dict[str, tuple[str, Any]] = {}
    if not hasattr(obj, "__dict__"):
        return state
    for key, value in obj.__dict__.items():
        if isinstance(value, np.ndarray):
            state[key] = ("ndarray", value.copy())
        elif isinstance(value, (int, float, bool, str, type(None))):
            state[key] = ("scalar", copy.deepcopy(value))
        elif isinstance(value, (list, tuple)) and all(
            isinstance(item, (int, float, bool, str, type(None))) for item in value
        ):
            state[key] = ("sequence", copy.deepcopy(value))
    return state


def _restore_simple_state(obj: Any, state: dict[str, tuple[str, Any]]) -> None:
    import numpy as np

    for key, (kind, value) in state.items():
        try:
            if kind == "ndarray" and hasattr(obj, key) and isinstance(getattr(obj, key), np.ndarray):
                target = getattr(obj, key)
                if target.shape == value.shape:
                    target[...] = value
                else:
                    setattr(obj, key, value.copy())
            else:
                setattr(obj, key, copy.deepcopy(value))
        except Exception:
            continue


def _sim_flat(sim) -> Any:
    import numpy as np

    parts = [sim.get_state().flatten()]
    for name in [
        "ctrl",
        "qfrc_applied",
        "xfrc_applied",
        "mocap_pos",
        "mocap_quat",
        "qacc_warmstart",
        "act",
    ]:
        if hasattr(sim.data, name):
            parts.append(np.asarray(getattr(sim.data, name), dtype=np.float64).reshape(-1))
    return np.concatenate(parts)


def make_snapshot(inner_env) -> dict[str, Any]:
    import numpy as np

    env = inner_env._env
    sim = env.sim
    snapshot: dict[str, Any] = {
        "sim_state": copy.deepcopy(sim.get_state()),
        "sim_extra": {},
        "env_simple": _simple_state(env),
        "wrapped_env_simple": _simple_state(getattr(env, "env", None)),
        "inner_simple": _simple_state(inner_env),
        "robots": [],
        "flat": _sim_flat(sim),
    }
    for name in [
        "ctrl",
        "qfrc_applied",
        "xfrc_applied",
        "mocap_pos",
        "mocap_quat",
        "qacc_warmstart",
        "act",
    ]:
        if hasattr(sim.data, name):
            snapshot["sim_extra"][name] = np.asarray(getattr(sim.data, name)).copy()

    for robot in getattr(env, "robots", []):
        robot_state = {
            "robot_simple": _simple_state(robot),
            "controller_simple": {},
            "gripper_simple": {},
        }
        if getattr(robot, "controller", None) is not None:
            robot_state["controller_simple"] = _simple_state(robot.controller)
        if getattr(robot, "gripper", None) is not None:
            robot_state["gripper_simple"] = _simple_state(robot.gripper)
        snapshot["robots"].append(robot_state)
    return snapshot


def restore_snapshot(inner_env, snapshot: dict[str, Any]) -> float:
    import numpy as np

    env = inner_env._env
    sim = env.sim
    sim.set_state(snapshot["sim_state"])
    for name, value in snapshot["sim_extra"].items():
        if hasattr(sim.data, name):
            target = getattr(sim.data, name)
            if target.shape == value.shape:
                target[...] = value
    sim.forward()

    _restore_simple_state(env, snapshot["env_simple"])
    if getattr(env, "env", None) is not None:
        _restore_simple_state(env.env, snapshot.get("wrapped_env_simple", {}))
    _restore_simple_state(inner_env, snapshot["inner_simple"])
    for robot, robot_state in zip(getattr(env, "robots", []), snapshot["robots"]):
        _restore_simple_state(robot, robot_state["robot_simple"])
        if getattr(robot, "controller", None) is not None:
            _restore_simple_state(robot.controller, robot_state["controller_simple"])
        if getattr(robot, "gripper", None) is not None:
            _restore_simple_state(robot.gripper, robot_state["gripper_simple"])
    return float(np.max(np.abs(_sim_flat(sim) - snapshot["flat"])))


def _task_descriptions(env) -> list[str]:
    try:
        return list(env.call("task_description"))
    except (AttributeError, NotImplementedError):
        try:
            return list(env.call("task"))
        except (AttributeError, NotImplementedError):
            return [""] * env.num_envs


def _extract_success(info: dict) -> bool:
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict) and "is_success" in final_info:
            values = final_info["is_success"]
            values = values.tolist() if hasattr(values, "tolist") else list(values)
            return bool(values[0])
    if "is_success" in info:
        value = info["is_success"]
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return bool(value[0])
        return bool(value)
    return False


def _infer_task_feature_source(critics, critic_checkpoints) -> str:
    if not critics or critics[0].config.task_feature_dim == 0:
        return "none"
    sources = [checkpoint.get("task_feature_source", "tokens") for checkpoint in critic_checkpoints]
    if len(set(sources)) != 1:
        raise ValueError(f"All QGF critics must use the same task feature source: {sources}")
    source = sources[0]
    if source == "episode":
        raise ValueError("Runtime QGF cannot use task_feature_source='episode'.")
    return source


def _build_policy_batch(
    *,
    observation,
    task_description: str,
    env_preprocessor,
    preprocessor,
):
    from lerobot.envs.utils import preprocess_observation

    observation = preprocess_observation(copy.deepcopy(observation))
    observation["task"] = [task_description]
    observation = env_preprocessor(observation)
    return preprocessor(observation)


def _select_env_action(
    *,
    policy,
    observation,
    task_description: str,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
):
    import torch
    from lerobot.utils.constants import ACTION

    policy_batch = _build_policy_batch(
        observation=observation,
        task_description=task_description,
        env_preprocessor=env_preprocessor,
        preprocessor=preprocessor,
    )
    with torch.inference_mode():
        policy_action = policy.select_action(policy_batch)
    action = postprocessor(policy_action)
    action_transition = env_postprocessor({ACTION: action})
    env_action = action_transition[ACTION]
    return policy_batch, policy_action.detach().cpu()[0], env_action.detach().cpu().numpy()[0]


def _policy_action_to_env_action(*, policy_action, postprocessor, env_postprocessor):
    import torch
    from lerobot.utils.constants import ACTION

    if policy_action.ndim == 1:
        policy_action = policy_action.unsqueeze(0)
    with torch.no_grad():
        action = postprocessor(policy_action)
        action_transition = env_postprocessor({ACTION: action})
        env_action = action_transition[ACTION]
    return env_action.detach().cpu().numpy()[0]


def _score_branch(success: bool, total_steps: int, max_steps: int, step_bonus: float) -> float:
    if not success:
        return 0.0
    remaining_fraction = max(0.0, 1.0 - float(total_steps) / max(1, max_steps))
    return 1.0 + float(step_bonus) * remaining_fraction


def _evaluate_branch(
    *,
    vec_env,
    inner_env,
    snapshot,
    observation,
    task_description: str,
    branch_name: str,
    prefix_policy,
    tail_policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    prefix_steps: int,
    tail_max_steps: int,
    noise_std: float,
    rng,
) -> dict[str, Any]:
    import numpy as np
    import torch

    restore_diff = restore_snapshot(inner_env, snapshot)
    current_obs = copy.deepcopy(observation)
    prefix_policy.reset()
    prefix_actions = []
    prefix_policy_actions = []
    success = False
    terminated_any = False

    for _ in range(prefix_steps):
        policy_batch, policy_action, env_action = _select_env_action(
            policy=prefix_policy,
            observation=current_obs,
            task_description=task_description,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )
        if branch_name == "noisy":
            noisy_policy_action = policy_action + torch.tensor(
                rng.normal(0.0, noise_std, size=tuple(policy_action.shape)),
                dtype=policy_action.dtype,
            )
            noisy_policy_action = noisy_policy_action.clamp(-1.0, 1.0)
            env_action = _policy_action_to_env_action(
                policy_action=noisy_policy_action,
                postprocessor=postprocessor,
                env_postprocessor=env_postprocessor,
            )
            policy_action = noisy_policy_action
        try:
            current_obs, reward, terminated, truncated, info = vec_env.step(
                env_action[None, :].astype(np.float32)
            )
        except ValueError as exc:
            if "terminated episode" not in str(exc):
                raise
            terminated_any = True
            break
        prefix_actions.append(torch.tensor(env_action, dtype=torch.float32))
        prefix_policy_actions.append(policy_action.float())
        success = success or _extract_success(info)
        terminated_any = terminated_any or bool(terminated[0]) or bool(truncated[0])
        if terminated_any:
            break

    tail_steps = 0
    if not success and not terminated_any:
        tail_policy.reset()
        while tail_steps < tail_max_steps:
            _, _, env_action = _select_env_action(
                policy=tail_policy,
                observation=current_obs,
                task_description=task_description,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )
            try:
                current_obs, reward, terminated, truncated, info = vec_env.step(
                    env_action[None, :].astype(np.float32)
                )
            except ValueError as exc:
                if "terminated episode" not in str(exc):
                    raise
                break
            tail_steps += 1
            success = success or _extract_success(info)
            if success or bool(terminated[0]) or bool(truncated[0]):
                break

    while len(prefix_actions) < prefix_steps:
        prefix_actions.append(torch.zeros_like(prefix_actions[-1]) if prefix_actions else torch.zeros(7))
        prefix_policy_actions.append(
            torch.zeros_like(prefix_policy_actions[-1]) if prefix_policy_actions else torch.zeros(7)
        )

    total_steps = min(prefix_steps, len(prefix_actions)) + tail_steps
    return {
        "branch": branch_name,
        "restore_flat_max_abs_diff": restore_diff,
        "success": bool(success),
        "prefix_steps_executed": int(min(prefix_steps, len(prefix_actions))),
        "tail_steps": int(tail_steps),
        "total_steps": int(total_steps),
        "action_chunk_env": torch.stack(prefix_actions),
        "action_chunk_policy": torch.stack(prefix_policy_actions),
    }


def _train_ranking_critic(
    *,
    samples: list[dict[str, Any]],
    output_dir: Path,
    action_horizon: int,
    hidden_dim: int,
    depth: int,
    epochs: int,
    lr: float,
    device_name: str,
):
    import torch
    import torch.nn.functional as F

    from guided_action_flow.critics.action_chunk_critic import (
        ActionChunkCritic,
        ActionChunkCriticConfig,
    )

    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        groups.setdefault(sample["group_id"], []).append(index)

    pairs = []
    for group_id, indices in groups.items():
        for i in indices:
            for j in indices:
                if samples[i]["score"] > samples[j]["score"] + 1.0e-6:
                    pairs.append((i, j, group_id))

    metrics: dict[str, Any] = {
        "num_samples": len(samples),
        "num_groups": len(groups),
        "num_pairs": len(pairs),
    }
    if not pairs:
        metrics["status"] = "no_ranking_pairs"
        (output_dir / "ranking_metrics.json").write_text(json.dumps(metrics, indent=2))
        return metrics

    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    obs_features = torch.stack([sample["obs_feature"] for sample in samples]).float()
    action_chunks = torch.stack([sample["action_chunk_env"] for sample in samples]).float()
    critic_config = ActionChunkCriticConfig(
        obs_feature_dim=obs_features.shape[-1],
        action_dim=action_chunks.shape[-1],
        action_horizon=action_horizon,
        hidden_dim=hidden_dim,
        depth=depth,
    )
    critic = ActionChunkCritic(critic_config)
    critic.module.to(device)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=lr, weight_decay=1.0e-4)

    group_ids = sorted(groups)
    val_groups = set(group_ids[-max(1, len(group_ids) // 5) :]) if len(group_ids) >= 3 else set()
    train_pairs = [pair for pair in pairs if pair[2] not in val_groups]
    val_pairs = [pair for pair in pairs if pair[2] in val_groups]
    if not train_pairs:
        train_pairs = pairs
        val_pairs = []

    def pair_loss(pair_batch):
        good = torch.tensor([pair[0] for pair in pair_batch], dtype=torch.long)
        bad = torch.tensor([pair[1] for pair in pair_batch], dtype=torch.long)
        q_good = critic(
            obs_features=obs_features[good].to(device),
            action_chunk=action_chunks[good].to(device),
        )
        q_bad = critic(
            obs_features=obs_features[bad].to(device),
            action_chunk=action_chunks[bad].to(device),
        )
        loss = F.softplus(-(q_good - q_bad)).mean()
        acc = (q_good > q_bad).float().mean()
        return loss, acc

    history = []
    batch_size = min(32, max(1, len(train_pairs)))
    generator = torch.Generator().manual_seed(0)
    for epoch in range(epochs):
        order = torch.randperm(len(train_pairs), generator=generator).tolist()
        losses = []
        accuracies = []
        critic.module.train()
        for start in range(0, len(order), batch_size):
            batch = [train_pairs[idx] for idx in order[start : start + batch_size]]
            loss, acc = pair_loss(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            accuracies.append(float(acc.detach().cpu().item()))
        row = {
            "epoch": epoch,
            "train_loss": float(sum(losses) / max(1, len(losses))),
            "train_pair_acc": float(sum(accuracies) / max(1, len(accuracies))),
        }
        if val_pairs:
            critic.module.eval()
            with torch.no_grad():
                val_loss, val_acc = pair_loss(val_pairs)
            row["val_loss"] = float(val_loss.detach().cpu().item())
            row["val_pair_acc"] = float(val_acc.detach().cpu().item())
        history.append(row)

    checkpoint = {
        "critic_config": asdict(critic_config),
        "model_state_dict": {key: value.detach().cpu() for key, value in critic.state_dict().items()},
        "training_kind": "counterfactual_pairwise_ranking",
        "action_source": "env_action_prefix",
        "history": history,
        "num_samples": len(samples),
        "num_pairs": len(pairs),
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
    }
    torch.save(checkpoint, output_dir / "counterfactual_ranking_critic.pt")
    metrics.update(
        {
            "status": "trained",
            "num_train_pairs": len(train_pairs),
            "num_val_pairs": len(val_pairs),
            "final": history[-1],
        }
    )
    (output_dir / "ranking_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    args = _build_arg_parser().parse_args()

    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
    from lerobot.envs.utils import close_envs
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )
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
    task_ids = _parse_int_list(args.task_ids)
    branch_steps = set(_parse_int_list(args.branch_steps))
    rename_map = _parse_rename_map(args.rename_map)

    env_cfg = make_env_config(
        args.env_type,
        task=args.task,
        task_ids=task_ids,
        max_parallel_tasks=1,
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)

    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = Path(args.policy_path)
    policy_cfg.device = args.device
    qgf_policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    qgf_policy_cfg.pretrained_path = Path(args.policy_path)
    qgf_policy_cfg.device = args.device
    preprocessor_overrides = {
        "device_processor": {"device": str(policy_cfg.device)},
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

    baseline_policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=rename_map)
    baseline_policy.eval()
    qgf_policy = make_policy(cfg=qgf_policy_cfg, env_cfg=env_cfg, rename_map=rename_map)
    qgf_policy.eval()

    qgf_info: dict[str, Any] = {"enabled": False}
    if args.critic_paths:
        critics = []
        critic_checkpoints = []
        for critic_path in args.critic_paths:
            critic, checkpoint = load_action_chunk_critic(critic_path, device=str(policy_cfg.device))
            critics.append(critic)
            critic_checkpoints.append(checkpoint)
        critic_action_dim = critics[0].config.action_dim
        critic_task_feature_dim = critics[0].config.task_feature_dim
        critic_task_feature_source = _infer_task_feature_source(critics, critic_checkpoints)
        install_smolvla_qgf(
            qgf_policy,
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
        qgf_info = {
            "enabled": True,
            "critic_paths": [str(path) for path in args.critic_paths],
            "beta": args.qgf_beta,
            "grad_clip_norm": args.qgf_grad_clip_norm,
            "uncertainty_scale": args.qgf_uncertainty_scale,
            "min_gate": args.qgf_min_gate,
            "task_feature_source": critic_task_feature_source,
            "selected_epochs": [checkpoint.get("selected_epoch") for checkpoint in critic_checkpoints],
        }

    rng = np.random.default_rng(args.seed + 999)
    records: list[dict[str, Any]] = []
    torch_records: list[dict[str, Any]] = []
    start_time = time.time()

    def save_partial() -> None:
        torch.save({"samples": torch_records}, output_dir / "counterfactual_samples.pt")
        partial = {
            "num_records": len(records),
            "num_groups": len({record["group_id"] for record in records}),
            "records": records,
            "wall_time_s": time.time() - start_time,
            "status": "partial",
        }
        (output_dir / "counterfactual_partial.json").write_text(
            json.dumps(partial, indent=2, ensure_ascii=False)
        )

    try:
        for task_group, group in envs.items():
            for task_id, vec_env in group.items():
                inner_env = vec_env.envs[0]
                task_description = _task_descriptions(vec_env)[0]
                max_steps = int(vec_env.call("_max_episode_steps")[0])
                for episode_index in range(args.n_episodes):
                    seed = args.seed + episode_index
                    baseline_policy.reset()
                    observation, _ = vec_env.reset(seed=[seed])
                    for step in range(max_steps):
                        if step in branch_steps:
                            snapshot = make_snapshot(inner_env)
                            snapshot_batch = _build_policy_batch(
                                observation=observation,
                                task_description=task_description,
                                env_preprocessor=env_preprocessor,
                                preprocessor=preprocessor,
                            )
                            obs_feature = snapshot_batch[OBS_STATE].detach().cpu()[0].float()
                            task_tokens = (
                                snapshot_batch[OBS_LANGUAGE_TOKENS].detach().cpu()[0]
                                if OBS_LANGUAGE_TOKENS in snapshot_batch
                                else None
                            )
                            task_attention_mask = (
                                snapshot_batch[OBS_LANGUAGE_ATTENTION_MASK].detach().cpu()[0]
                                if OBS_LANGUAGE_ATTENTION_MASK in snapshot_batch
                                else None
                            )
                            group_id = f"{task_group}_{task_id}_ep{episode_index}_step{step}"
                            for branch_name, prefix_policy in [
                                ("baseline", baseline_policy),
                                ("qgf", qgf_policy),
                                ("noisy", baseline_policy),
                            ]:
                                branch = _evaluate_branch(
                                    vec_env=vec_env,
                                    inner_env=inner_env,
                                    snapshot=snapshot,
                                    observation=observation,
                                    task_description=task_description,
                                    branch_name=branch_name,
                                    prefix_policy=prefix_policy,
                                    tail_policy=baseline_policy,
                                    env_preprocessor=env_preprocessor,
                                    env_postprocessor=env_postprocessor,
                                    preprocessor=preprocessor,
                                    postprocessor=postprocessor,
                                    prefix_steps=args.prefix_steps,
                                    tail_max_steps=min(args.tail_max_steps, max_steps - step),
                                    noise_std=args.noise_std,
                                    rng=rng,
                                )
                                score = _score_branch(
                                    branch["success"],
                                    branch["total_steps"],
                                    min(args.tail_max_steps, max_steps - step),
                                    args.score_step_bonus,
                                )
                                public_record = {
                                    "group_id": group_id,
                                    "task_group": task_group,
                                    "task_id": int(task_id),
                                    "episode_index": episode_index,
                                    "seed": seed,
                                    "snapshot_step": step,
                                    "branch": branch_name,
                                    "success": branch["success"],
                                    "score": score,
                                    "prefix_steps_executed": branch["prefix_steps_executed"],
                                    "tail_steps": branch["tail_steps"],
                                    "total_steps": branch["total_steps"],
                                    "restore_flat_max_abs_diff": branch["restore_flat_max_abs_diff"],
                                }
                                records.append(public_record)
                                torch_record = {
                                    **public_record,
                                    "obs_feature": obs_feature,
                                    "action_chunk_env": branch["action_chunk_env"],
                                    "action_chunk_policy": branch["action_chunk_policy"],
                                }
                                if task_tokens is not None and task_attention_mask is not None:
                                    torch_record["task_tokens"] = task_tokens
                                    torch_record["task_attention_mask"] = task_attention_mask
                                torch_records.append(torch_record)
                                print(json.dumps(public_record, ensure_ascii=False))
                                if (
                                    args.save_every_records > 0
                                    and len(records) % args.save_every_records == 0
                                ):
                                    save_partial()

                            restore_snapshot(inner_env, snapshot)
                            baseline_policy.reset()

                        _, _, env_action = _select_env_action(
                            policy=baseline_policy,
                            observation=observation,
                            task_description=task_description,
                            env_preprocessor=env_preprocessor,
                            env_postprocessor=env_postprocessor,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                        )
                        observation, reward, terminated, truncated, info = vec_env.step(
                            env_action[None, :].astype(np.float32)
                        )
                        if _extract_success(info) or bool(terminated[0]) or bool(truncated[0]):
                            break
    finally:
        close_envs(envs)

    torch.save({"samples": torch_records}, output_dir / "counterfactual_samples.pt")

    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        group = grouped.setdefault(
            record["group_id"],
            {
                "group_id": record["group_id"],
                "task_group": record["task_group"],
                "task_id": record["task_id"],
                "episode_index": record["episode_index"],
                "seed": record["seed"],
                "snapshot_step": record["snapshot_step"],
                "branches": {},
            },
        )
        group["branches"][record["branch"]] = {
            "success": record["success"],
            "score": record["score"],
            "total_steps": record["total_steps"],
            "tail_steps": record["tail_steps"],
        }

    summary = {
        "config": {
            "policy_path": str(args.policy_path),
            "task": args.task,
            "task_ids": task_ids,
            "seed": args.seed,
            "n_episodes": args.n_episodes,
            "branch_steps": sorted(branch_steps),
            "prefix_steps": args.prefix_steps,
            "tail_max_steps": args.tail_max_steps,
            "noise_std": args.noise_std,
            "rename_map": rename_map,
            "qgf": qgf_info,
        },
        "num_records": len(records),
        "num_groups": len(grouped),
        "branch_success_rate": {
            branch: (
                sum(1 for record in records if record["branch"] == branch and record["success"])
                / max(1, sum(1 for record in records if record["branch"] == branch))
            )
            for branch in sorted({record["branch"] for record in records})
        },
        "groups": list(grouped.values()),
        "wall_time_s": time.time() - start_time,
    }
    (output_dir / "counterfactual_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    if args.skip_ranking:
        ranking_metrics = {"status": "skipped"}
    else:
        ranking_metrics = _train_ranking_critic(
            samples=torch_records,
            output_dir=output_dir,
            action_horizon=args.prefix_steps,
            hidden_dim=args.ranking_hidden_dim,
            depth=args.ranking_depth,
            epochs=args.ranking_epochs,
            lr=args.ranking_lr,
            device_name=args.device,
        )
    summary["ranking_metrics"] = ranking_metrics
    (output_dir / "counterfactual_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(json.dumps({"branch_success_rate": summary["branch_success_rate"], "ranking": ranking_metrics}, indent=2))


if __name__ == "__main__":
    main()
