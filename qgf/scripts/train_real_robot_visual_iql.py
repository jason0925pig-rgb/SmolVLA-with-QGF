#!/usr/bin/env python3
"""Train visual action-chunk critics with offline IQL.

Input is the feature cache written by ``extract_smolvla_visual_features.py``.
Each row is a real Armstrong transition containing the frozen two-view SmolVLA
tokens, 8-D proprioception, one exact normalized 50-step policy chunk, and its
recorded sparse terminal outcome.  Splits are fixed at episode level; no frame
from a validation episode is used in training.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    # The real-robot first pass deliberately trains one critic.  A critic
    # ensemble is only needed for disagreement-based uncertainty gating, which
    # is disabled for this experiment.
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount per committed 50-step chunk.")
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--polyak", type=float, default=0.005)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _clone_to_cpu(state_dict):
    return {name: value.detach().cpu().clone() for name, value in state_dict.items()}


def _expectile_loss(advantage, expectile: float):
    import torch

    weight = torch.where(advantage > 0, expectile, 1.0 - expectile)
    return (weight * advantage.square()).mean()


def _load_episode(data_dir: Path, episode_index: int) -> dict:
    import torch

    path = data_dir / f"episode_{episode_index:06d}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing visual feature cache: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "state", "action_chunk", "next_state", "reward", "success", "done",
        "terminated", "truncated", "visual_features", "next_visual_features", "next_visual_valid",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")
    count = int(payload["state"].shape[0])
    if count < 1 or tuple(payload["state"].shape[1:]) != (8,):
        raise ValueError(f"{path}: expected state [N,8], got {tuple(payload['state'].shape)}")
    if tuple(payload["action_chunk"].shape[1:]) != (50, 8):
        raise ValueError(f"{path}: expected action_chunk [N,50,8]")
    if tuple(payload["visual_features"].shape[1:]) != (128, 960):
        raise ValueError(f"{path}: expected visual_features [N,128,960]")
    if any(int(payload[key].shape[0]) != count for key in required if hasattr(payload[key], "shape")):
        raise ValueError(f"{path}: inconsistent sample count")
    return payload


def _join_episodes(data_dir: Path, episode_indices: list[int]) -> dict:
    import torch

    fields = [
        "state", "action_chunk", "next_state", "reward", "success", "done", "terminated", "truncated",
        "visual_features", "next_visual_features", "next_visual_valid",
    ]
    columns = {field: [] for field in fields}
    for episode_index in episode_indices:
        payload = _load_episode(data_dir, episode_index)
        for field in fields:
            columns[field].append(payload[field])
    out = {field: torch.cat(values, dim=0) for field, values in columns.items()}
    # A chunk which reaches recorded success is terminal even if a recorder did
    # not write its done flag until the following control cycle.
    out["done"] = out["done"].bool() | out["terminated"].bool() | out["truncated"].bool() | out["success"].bool()
    out["reward"] = torch.maximum(out["reward"].float(), out["success"].float())
    return out


class _TensorRows:
    def __init__(self, tensors: dict):
        self.tensors = tensors
        self.length = int(tensors["state"].shape[0])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return tuple(
            self.tensors[key][index]
            for key in ("state", "visual_features", "action_chunk", "next_state", "next_visual_features", "reward", "done")
        )


def _evaluate(q_model, value_model, data, *, batch_size: int, gamma: float, device):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    loader = DataLoader(_TensorRows(data), batch_size=batch_size, shuffle=False)
    q_model.eval()
    value_model.eval()
    losses, q_values, success_values = [], [], []
    with torch.inference_mode():
        for state, visual, action, next_state, next_visual, reward, done in loader:
            state = state.to(device); visual = visual.to(device); action = action.to(device)
            next_state = next_state.to(device); next_visual = next_visual.to(device)
            reward = reward.to(device); done = done.to(device).float()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = q_model(state, visual, action)
                target = reward + gamma * (1.0 - done) * value_model.forward_value(next_state, next_visual)
                losses.append(float(F.mse_loss(prediction, target).cpu()))
            q_values.append(prediction.float().cpu())
            success_values.append((reward > 0).cpu())
    q_values = torch.cat(q_values)
    success_values = torch.cat(success_values)
    success_q = q_values[success_values]
    failure_q = q_values[~success_values]
    return {
        "td_loss": float(sum(losses) / max(1, len(losses))),
        "q_mean": float(q_values.mean()),
        "q_success_mean": float(success_q.mean()) if success_q.numel() else None,
        "q_failure_mean": float(failure_q.mean()) if failure_q.numel() else None,
        "q_success_failure_gap": (
            float(success_q.mean() - failure_q.mean()) if success_q.numel() and failure_q.numel() else None
        ),
        "positive_reward_samples": int(success_values.sum()),
        "samples": int(q_values.numel()),
    }


def _train_member(*, member_index: int, args, train_data, val_data, config, device):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from guided_action_flow.critics.visual_transformer_critic import VisualTransformerCritic

    member_seed = args.seed + member_index * 1009
    torch.manual_seed(member_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(member_seed)
    q_model = VisualTransformerCritic(config).module.to(device)
    target_model = VisualTransformerCritic(config).module.to(device)
    target_model.load_state_dict(copy.deepcopy(q_model.state_dict()))
    for parameter in target_model.parameters():
        parameter.requires_grad_(False)
    value_model = VisualTransformerCritic(config).module.to(device)
    optimizer_q = torch.optim.AdamW(q_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_v = torch.optim.AdamW(value_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(member_seed)
    loader = DataLoader(_TensorRows(train_data), batch_size=args.batch_size, shuffle=True, generator=generator)

    best = {"val_td_loss": float("inf"), "epoch": None, "q_state_dict": None, "value_state_dict": None}
    history = []
    for epoch in range(1, args.epochs + 1):
        q_model.train(); value_model.train()
        q_losses, v_losses = [], []
        for state, visual, action, next_state, next_visual, reward, done in loader:
            state = state.to(device); visual = visual.to(device); action = action.to(device)
            next_state = next_state.to(device); next_visual = next_visual.to(device)
            reward = reward.to(device); done = done.to(device).float()

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                target_q = target_model(state, visual, action)
            value = value_model.forward_value(state, visual)
            value_loss = _expectile_loss(target_q.float() - value.float(), args.expectile)
            optimizer_v.zero_grad(set_to_none=True)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_model.parameters(), 10.0)
            optimizer_v.step()

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                q_target = reward + args.gamma * (1.0 - done) * value_model.forward_value(next_state, next_visual)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                q_prediction = q_model(state, visual, action)
                q_loss = F.mse_loss(q_prediction.float(), q_target.float())
            optimizer_q.zero_grad(set_to_none=True)
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(q_model.parameters(), 10.0)
            optimizer_q.step()

            with torch.no_grad():
                for target_parameter, parameter in zip(target_model.parameters(), q_model.parameters()):
                    target_parameter.mul_(1.0 - args.polyak).add_(parameter, alpha=args.polyak)
            q_losses.append(float(q_loss.detach().cpu()))
            v_losses.append(float(value_loss.detach().cpu()))

        val_metrics = _evaluate(q_model, value_model, val_data, batch_size=args.batch_size, gamma=args.gamma, device=device)
        entry = {
            "epoch": epoch,
            "train_q_loss": float(sum(q_losses) / max(1, len(q_losses))),
            "train_v_loss": float(sum(v_losses) / max(1, len(v_losses))),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(entry)
        print(json.dumps({"member": member_index, **entry}), flush=True)
        if val_metrics["td_loss"] < best["val_td_loss"]:
            best = {
                "val_td_loss": val_metrics["td_loss"],
                "epoch": epoch,
                "q_state_dict": _clone_to_cpu(q_model.state_dict()),
                "value_state_dict": _clone_to_cpu(value_model.state_dict()),
            }
    return best, history


def main() -> None:
    args = _parse_args()
    if args.ensemble_size < 1 or args.batch_size < 1 or args.epochs < 1:
        raise ValueError("--ensemble-size, --batch-size and --epochs must be positive.")
    import torch
    from guided_action_flow.critics.visual_transformer_critic import VisualTransformerCriticConfig

    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    train_episodes = [int(item) for item in split["train_episode_indices"]]
    val_episodes = [int(item) for item in split["val_episode_indices"]]
    if len(train_episodes) != 90 or len(val_episodes) != 10 or set(train_episodes) & set(val_episodes):
        raise ValueError("Expected a disjoint fixed 90/10 episode split.")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_data = _join_episodes(args.data_dir, train_episodes)
    val_data = _join_episodes(args.data_dir, val_episodes)
    config = VisualTransformerCriticConfig(
        state_dim=int(train_data["state"].shape[-1]),
        action_dim=int(train_data["action_chunk"].shape[-1]),
        action_horizon=int(train_data["action_chunk"].shape[-2]),
        visual_tokens=int(train_data["visual_features"].shape[-2]),
        visual_token_dim=int(train_data["visual_features"].shape[-1]),
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
    )
    metadata = {
        "format": "armstrong-qgf-visual-iql-v1",
        "device": str(device),
        "critic_config": asdict(config),
        "train_episode_indices": train_episodes,
        "val_episode_indices": val_episodes,
        "train_samples": int(train_data["state"].shape[0]),
        "val_samples": int(val_data["state"].shape[0]),
        "train_positive_rewards": int((train_data["reward"] > 0).sum()),
        "val_positive_rewards": int((val_data["reward"] > 0).sum()),
        "training_args": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
    }
    (args.output_dir / "training_input_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summaries = []
    for member_index in range(args.ensemble_size):
        best, history = _train_member(
            member_index=member_index,
            args=args,
            train_data=train_data,
            val_data=val_data,
            config=config,
            device=device,
        )
        checkpoint = {
            "format": "armstrong-qgf-visual-iql-v1",
            "critic_arch": "visual_transformer",
            "critic_config": asdict(config),
            "model_state_dict": best["q_state_dict"],
            "value_model_state_dict": best["value_state_dict"],
            "selected_epoch": best["epoch"],
            "selected_val_td_loss": best["val_td_loss"],
            "history": history,
            **metadata,
            "ensemble_member_index": member_index,
            "member_seed": args.seed + member_index * 1009,
        }
        output_path = args.output_dir / f"critic_member_{member_index:02d}.pt"
        torch.save(checkpoint, output_path)
        summaries.append({
            "member_index": member_index,
            "path": output_path.name,
            "selected_epoch": best["epoch"],
            "selected_val_td_loss": best["val_td_loss"],
        })
    (args.output_dir / "training_summary.json").write_text(
        json.dumps({**metadata, "members": summaries}, indent=2) + "\n"
    )
    print(json.dumps({"TRAINING_COMPLETE": True, "members": summaries}, indent=2))


if __name__ == "__main__":
    main()
