#!/usr/bin/env python
"""Train a QGF-compatible critic with MSE targets plus counterfactual ranking."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-data-dirs", nargs="+", required=True)
    parser.add_argument("--counterfactual-files", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--action-key", default="action_policy")
    parser.add_argument("--cf-action-key", default="action_chunk_policy")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pair-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--ranking-weight", type=float, default=0.2)
    parser.add_argument("--cf-regression-weight", type=float, default=0.0)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--score-margin", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser


def _clone_state_dict_to_cpu(state_dict):
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _select_checkpoint_metadata(history: list[dict]) -> dict:
    val_entries = [entry for entry in history if "val_total_loss" in entry]
    if not val_entries:
        return {
            "selected_epoch": history[-1]["epoch"] if history else None,
            "selected_metric": "final",
            "selected_val_loss": None,
        }
    best = min(val_entries, key=lambda entry: entry["val_total_loss"])
    return {
        "selected_epoch": best["epoch"],
        "selected_metric": "val_total_loss",
        "selected_val_loss": best["val_total_loss"],
    }


def _discover_counterfactual_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            files.extend(sorted(path.rglob("counterfactual_samples.pt")))
        else:
            files.append(path)
    return files


def _load_counterfactual_samples(files: list[Path], *, action_key: str, action_horizon: int):
    import torch

    samples = []
    skipped = 0
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for sample in payload.get("samples", []):
            if action_key not in sample:
                skipped += 1
                continue
            action_chunk = torch.as_tensor(sample[action_key], dtype=torch.float32)
            obs_feature = torch.as_tensor(sample["obs_feature"], dtype=torch.float32)
            if action_chunk.ndim != 2 or action_chunk.shape[0] != action_horizon:
                skipped += 1
                continue
            samples.append(
                {
                    "group_id": str(sample["group_id"]),
                    "branch": str(sample.get("branch", "")),
                    "score": float(sample.get("score", 0.0)),
                    "target": max(0.0, min(1.0, float(sample.get("score", 0.0)))),
                    "success": bool(sample.get("success", False)),
                    "obs_feature": obs_feature.flatten(),
                    "action_chunk": action_chunk,
                }
            )
    return samples, skipped


def _build_ranking_pairs(samples, *, margin: float):
    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        groups.setdefault(sample["group_id"], []).append(index)

    pairs = []
    for group_id, indices in groups.items():
        for good in indices:
            for bad in indices:
                if samples[good]["score"] > samples[bad]["score"] + margin:
                    pairs.append((good, bad, group_id))
    return groups, pairs


def main() -> None:
    args = _build_arg_parser().parse_args()

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    from guided_action_flow.critics.action_chunk_critic import (
        ActionChunkCritic,
        ActionChunkCriticConfig,
    )
    from guided_action_flow.training.critic_dataset import (
        build_action_chunk_dataset,
        discover_episode_files,
        load_episode_files,
        split_indices_by_episode,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_files = []
    for data_dir in args.rollout_data_dirs:
        episode_files.extend(discover_episode_files(data_dir))
    if not episode_files:
        raise FileNotFoundError(f"No rollout episode_*.pt files found under {args.rollout_data_dirs}.")

    episodes = load_episode_files(episode_files)
    mse_dataset = build_action_chunk_dataset(
        episodes,
        action_horizon=args.action_horizon,
        stride=args.stride,
        gamma=args.gamma,
        action_key=args.action_key,
        target_mode="success_to_go",
        task_feature_source="none",
    )

    cf_files = _discover_counterfactual_files(args.counterfactual_files)
    cf_samples, cf_skipped = _load_counterfactual_samples(
        cf_files,
        action_key=args.cf_action_key,
        action_horizon=args.action_horizon,
    )
    cf_groups, cf_pairs = _build_ranking_pairs(cf_samples, margin=args.score_margin)
    if not cf_pairs:
        raise ValueError("No counterfactual ranking pairs were produced.")

    generator = torch.Generator().manual_seed(args.seed)
    train_indices, val_indices = split_indices_by_episode(
        mse_dataset["episode_indices"],
        val_fraction=args.val_fraction,
        generator=generator,
    )

    mse_train_ds = TensorDataset(
        mse_dataset["obs_features"][train_indices],
        mse_dataset["action_chunks"][train_indices],
        mse_dataset["targets"][train_indices],
    )
    mse_train_loader = DataLoader(
        mse_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )

    group_ids = sorted(cf_groups)
    val_group_count = int(len(group_ids) * args.val_fraction)
    val_group_count = max(1, min(val_group_count, len(group_ids) - 1)) if len(group_ids) > 1 else 0
    shuffled_groups = [group_ids[index] for index in torch.randperm(len(group_ids), generator=generator).tolist()]
    val_groups = set(shuffled_groups[:val_group_count])
    train_pairs = [pair for pair in cf_pairs if pair[2] not in val_groups]
    val_pairs = [pair for pair in cf_pairs if pair[2] in val_groups]
    if not train_pairs:
        train_pairs = cf_pairs
        val_pairs = []
    train_cf_indices = [
        index for index, sample in enumerate(cf_samples) if sample["group_id"] not in val_groups
    ]
    val_cf_indices = [
        index for index, sample in enumerate(cf_samples) if sample["group_id"] in val_groups
    ]
    if not train_cf_indices:
        train_cf_indices = list(range(len(cf_samples)))
        val_cf_indices = []

    cf_obs = torch.stack([sample["obs_feature"] for sample in cf_samples]).float()
    cf_actions = torch.stack([sample["action_chunk"] for sample in cf_samples]).float()
    cf_targets = torch.tensor([sample["target"] for sample in cf_samples], dtype=torch.float32)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    critic_config = ActionChunkCriticConfig(
        obs_feature_dim=mse_dataset["obs_features"].shape[-1],
        action_dim=mse_dataset["action_chunks"].shape[-1],
        action_horizon=args.action_horizon,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    )
    if cf_obs.shape[-1] != critic_config.obs_feature_dim:
        raise ValueError(f"CF obs dim {cf_obs.shape[-1]} != critic obs dim {critic_config.obs_feature_dim}.")
    if cf_actions.shape[1:] != (
        critic_config.action_horizon,
        critic_config.action_dim,
    ):
        raise ValueError(
            f"CF action shape {tuple(cf_actions.shape[1:])} does not match critic "
            f"({critic_config.action_horizon}, {critic_config.action_dim})."
        )

    critic = ActionChunkCritic(critic_config)
    critic.module.to(device)
    optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    def ranking_loss(pair_batch):
        good = torch.tensor([pair[0] for pair in pair_batch], dtype=torch.long)
        bad = torch.tensor([pair[1] for pair in pair_batch], dtype=torch.long)
        q_good = critic(
            obs_features=cf_obs[good].to(device),
            action_chunk=cf_actions[good].to(device),
        )
        q_bad = critic(
            obs_features=cf_obs[bad].to(device),
            action_chunk=cf_actions[bad].to(device),
        )
        loss = F.softplus(-(q_good - q_bad)).mean()
        acc = (q_good > q_bad).float().mean()
        return loss, acc

    def eval_mse(indices):
        if indices.numel() == 0:
            return None
        critic.module.eval()
        losses = []
        with torch.no_grad():
            for start in range(0, int(indices.numel()), args.batch_size):
                batch_indices = indices[start : start + args.batch_size]
                pred = critic(
                    obs_features=mse_dataset["obs_features"][batch_indices].to(device),
                    action_chunk=mse_dataset["action_chunks"][batch_indices].to(device),
                )
                target = mse_dataset["targets"][batch_indices].to(device)
                losses.append(float(F.mse_loss(pred, target).detach().cpu().item()))
        return sum(losses) / max(1, len(losses))

    def eval_pairs(pairs):
        if not pairs:
            return None, None
        critic.module.eval()
        with torch.no_grad():
            loss, acc = ranking_loss(pairs)
        return float(loss.detach().cpu().item()), float(acc.detach().cpu().item())

    def cf_regression_loss(indices):
        if not indices:
            return None
        idx = torch.tensor(indices, dtype=torch.long)
        pred = critic(
            obs_features=cf_obs[idx].to(device),
            action_chunk=cf_actions[idx].to(device),
        )
        target = cf_targets[idx].to(device)
        return F.mse_loss(pred, target)

    def sample_cf_indices():
        count = min(args.pair_batch_size, len(train_cf_indices))
        order = torch.randint(len(train_cf_indices), (count,), generator=generator)
        return [train_cf_indices[int(index)] for index in order]

    history = []
    best_val_loss = None
    best_state_dict = None
    pair_order = list(range(len(train_pairs)))
    for epoch in range(args.epochs):
        critic.module.train()
        mse_losses = []
        cf_losses = []
        rank_losses = []
        rank_accs = []
        pair_perm = torch.randperm(len(train_pairs), generator=generator).tolist()
        pair_cursor = 0
        for obs_batch, action_batch, target_batch in mse_train_loader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            target_batch = target_batch.to(device)
            pred = critic(obs_features=obs_batch, action_chunk=action_batch)
            mse_loss = F.mse_loss(pred, target_batch)

            if pair_cursor >= len(pair_perm):
                pair_perm = torch.randperm(len(train_pairs), generator=generator).tolist()
                pair_cursor = 0
            pair_indices = pair_perm[pair_cursor : pair_cursor + args.pair_batch_size]
            pair_cursor += args.pair_batch_size
            pair_batch = [train_pairs[index] for index in pair_indices]
            rank_loss, rank_acc = ranking_loss(pair_batch)
            cf_loss = cf_regression_loss(sample_cf_indices())
            if cf_loss is None:
                cf_loss = mse_loss.detach().new_tensor(0.0)

            total_loss = (
                args.mse_weight * mse_loss
                + args.cf_regression_weight * cf_loss
                + args.ranking_weight * rank_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            mse_losses.append(float(mse_loss.detach().cpu().item()))
            cf_losses.append(float(cf_loss.detach().cpu().item()))
            rank_losses.append(float(rank_loss.detach().cpu().item()))
            rank_accs.append(float(rank_acc.detach().cpu().item()))

        train_mse = sum(mse_losses) / max(1, len(mse_losses))
        train_cf = sum(cf_losses) / max(1, len(cf_losses))
        train_rank = sum(rank_losses) / max(1, len(rank_losses))
        train_rank_acc = sum(rank_accs) / max(1, len(rank_accs))
        row = {
            "epoch": epoch,
            "train_mse_loss": train_mse,
            "train_cf_regression_loss": train_cf,
            "train_ranking_loss": train_rank,
            "train_ranking_acc": train_rank_acc,
            "train_total_loss": (
                args.mse_weight * train_mse
                + args.cf_regression_weight * train_cf
                + args.ranking_weight * train_rank
            ),
        }

        val_mse = eval_mse(val_indices)
        val_cf_loss = cf_regression_loss(val_cf_indices)
        val_rank_loss, val_rank_acc = eval_pairs(val_pairs)
        if val_mse is not None:
            row["val_mse_loss"] = val_mse
        if val_cf_loss is not None:
            row["val_cf_regression_loss"] = float(val_cf_loss.detach().cpu().item())
        if val_rank_loss is not None:
            row["val_ranking_loss"] = val_rank_loss
            row["val_ranking_acc"] = val_rank_acc
        if val_mse is not None or val_rank_loss is not None:
            row["val_total_loss"] = (
                args.mse_weight * (val_mse or 0.0)
                + args.cf_regression_weight
                * (float(val_cf_loss.detach().cpu().item()) if val_cf_loss is not None else 0.0)
                + args.ranking_weight * (val_rank_loss or 0.0)
            )
            if best_val_loss is None or row["val_total_loss"] < best_val_loss:
                best_val_loss = row["val_total_loss"]
                best_state_dict = _clone_state_dict_to_cpu(critic.state_dict())
        history.append(row)
        print(json.dumps(row))

    final_state_dict = _clone_state_dict_to_cpu(critic.state_dict())
    selection = _select_checkpoint_metadata(history)
    selected_state_dict = (
        best_state_dict
        if selection["selected_metric"] == "val_total_loss" and best_state_dict is not None
        else final_state_dict
    )
    checkpoint = {
        "critic_config": asdict(critic_config),
        "model_state_dict": selected_state_dict,
        "final_model_state_dict": final_state_dict,
        "training_kind": "mse_success_to_go_plus_counterfactual_ranking",
        "training_args": vars(args),
        "rollout_data_dirs": [str(path) for path in args.rollout_data_dirs],
        "counterfactual_files": [str(path) for path in cf_files],
        "num_mse_samples": int(mse_dataset["targets"].shape[0]),
        "num_rollout_episodes": len(episodes),
        "num_counterfactual_samples": len(cf_samples),
        "num_counterfactual_skipped": int(cf_skipped),
        "num_counterfactual_groups": len(cf_groups),
        "num_counterfactual_pairs": len(cf_pairs),
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "num_train_counterfactual_samples": len(train_cf_indices),
        "num_val_counterfactual_samples": len(val_cf_indices),
        "target_mean": float(mse_dataset["targets"].mean().item()),
        "target_success_fraction": float((mse_dataset["targets"] > 0).float().mean().item()),
        "task_feature_source": "none",
        "target_mode": "success_to_go",
        "history": history,
        **selection,
    }
    torch.save(checkpoint, output_dir / "critic.pt")

    metrics = {
        key: checkpoint[key]
        for key in [
            "training_kind",
            "num_mse_samples",
            "num_rollout_episodes",
            "num_counterfactual_samples",
            "num_counterfactual_skipped",
            "num_counterfactual_groups",
            "num_counterfactual_pairs",
            "num_train_pairs",
            "num_val_pairs",
            "target_mean",
            "target_success_fraction",
            "selected_epoch",
            "selected_metric",
            "selected_val_loss",
        ]
    }
    metrics["final"] = history[-1] if history else {}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
