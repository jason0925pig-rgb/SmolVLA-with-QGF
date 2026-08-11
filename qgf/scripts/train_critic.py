#!/usr/bin/env python
"""Train an action-chunk critic from collected rollout data."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def _select_checkpoint_metadata(history: list[dict]) -> dict:
    if not history:
        return {
            "selected_epoch": None,
            "selected_metric": "none",
            "selected_val_loss": None,
        }

    val_entries = [entry for entry in history if "val_loss" in entry]
    if not val_entries:
        return {
            "selected_epoch": history[-1]["epoch"],
            "selected_metric": "final",
            "selected_val_loss": None,
        }

    best = min(val_entries, key=lambda entry: entry["val_loss"])
    return {
        "selected_epoch": best["epoch"],
        "selected_metric": "val_loss",
        "selected_val_loss": best["val_loss"],
    }


def _clone_state_dict_to_cpu(state_dict):
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        help="One or more rollout directories. Overrides --data-dir when provided.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--target-mode",
        choices=["success_to_go", "progress_blend", "retrieval_progress_blend"],
        default="success_to_go",
    )
    parser.add_argument("--progress-key", default="progress_score")
    parser.add_argument("--success-weight", type=float, default=0.7)
    parser.add_argument("--progress-weight", type=float, default=0.3)
    parser.add_argument("--retrieval-k", type=int, default=5)
    parser.add_argument(
        "--task-feature-source",
        choices=["none", "tokens", "episode", "vlm_hidden"],
        default="none",
    )
    parser.add_argument("--task-feature-dim", type=int, default=128)
    parser.add_argument("--task-feature-key", default="task_features")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--critic-arch", choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--transformer-d-model", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument("--transformer-ff-multiplier", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    from guided_action_flow.critics.action_chunk_critic import (
        ActionChunkCritic,
        ActionChunkCriticConfig,
    )
    from guided_action_flow.critics.transformer_action_chunk_critic import (
        TransformerActionChunkCritic,
        TransformerActionChunkCriticConfig,
    )
    from guided_action_flow.training.critic_dataset import (
        build_action_chunk_dataset,
        discover_episode_files,
        load_episode_files,
        split_indices_by_episode,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dirs = args.data_dirs or ([args.data_dir] if args.data_dir else [])
    if not data_dirs:
        raise ValueError("Provide --data-dir or --data-dirs.")

    episode_files = []
    for data_dir in data_dirs:
        episode_files.extend(discover_episode_files(data_dir))
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.pt files found under {data_dirs}.")

    episodes = load_episode_files(episode_files)
    dataset = build_action_chunk_dataset(
        episodes,
        action_horizon=args.action_horizon,
        stride=args.stride,
        gamma=args.gamma,
        target_mode=args.target_mode,
        progress_key=args.progress_key,
        success_weight=args.success_weight,
        progress_weight=args.progress_weight,
        retrieval_k=args.retrieval_k,
        task_feature_source=args.task_feature_source,
        task_feature_dim=args.task_feature_dim,
        task_feature_key=args.task_feature_key,
    )

    obs_features = dataset["obs_features"]
    action_chunks = dataset["action_chunks"]
    targets = dataset["targets"]
    task_features = dataset.get("task_features")

    generator = torch.Generator().manual_seed(args.seed)
    num_samples = targets.shape[0]
    train_indices, val_indices = split_indices_by_episode(
        dataset["episode_indices"],
        val_fraction=args.val_fraction,
        generator=generator,
    )

    if task_features is None:
        train_ds = TensorDataset(
            obs_features[train_indices],
            action_chunks[train_indices],
            targets[train_indices],
        )
    else:
        train_ds = TensorDataset(
            obs_features[train_indices],
            action_chunks[train_indices],
            task_features[train_indices],
            targets[train_indices],
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    device_name = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    device = torch.device(device_name)
    common_config = {
        "obs_feature_dim": obs_features.shape[-1],
        "action_dim": action_chunks.shape[-1],
        "action_horizon": action_chunks.shape[1],
        "task_feature_dim": 0 if task_features is None else task_features.shape[-1],
    }
    if args.critic_arch == "mlp":
        critic_config = ActionChunkCriticConfig(
            **common_config,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
        )
        critic = ActionChunkCritic(critic_config)
    else:
        critic_config = TransformerActionChunkCriticConfig(
            **common_config,
            d_model=args.transformer_d_model,
            num_layers=args.transformer_layers,
            num_heads=args.transformer_heads,
            dropout=args.transformer_dropout,
            ff_multiplier=args.transformer_ff_multiplier,
        )
        critic = TransformerActionChunkCritic(critic_config)
    critic.module.to(device)
    optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_val_loss = None
    best_model_state_dict = None
    for epoch in range(args.epochs):
        critic.module.train()
        train_losses = []
        for batch in train_loader:
            if task_features is None:
                obs_batch, action_batch, target_batch = batch
                task_batch = None
            else:
                obs_batch, action_batch, task_batch, target_batch = batch
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            target_batch = target_batch.to(device)
            task_batch = task_batch.to(device) if task_batch is not None else None

            pred = critic(
                obs_features=obs_batch,
                action_chunk=action_batch,
                task_features=task_batch,
            )
            loss = F.mse_loss(pred, target_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        metrics = {
            "epoch": epoch,
            "train_loss": float(sum(train_losses) / max(1, len(train_losses))),
        }
        if val_indices.numel() > 0:
            critic.module.eval()
            with torch.no_grad():
                val_pred = critic(
                    obs_features=obs_features[val_indices].to(device),
                    action_chunk=action_chunks[val_indices].to(device),
                    task_features=(
                        task_features[val_indices].to(device)
                        if task_features is not None
                        else None
                    ),
                )
                val_loss = F.mse_loss(val_pred, targets[val_indices].to(device))
            metrics["val_loss"] = float(val_loss.detach().cpu().item())
            if best_val_loss is None or metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                best_model_state_dict = _clone_state_dict_to_cpu(critic.state_dict())
        history.append(metrics)
        print(json.dumps(metrics))

    final_model_state_dict = _clone_state_dict_to_cpu(critic.state_dict())
    selection = _select_checkpoint_metadata(history)
    selected_state_dict = (
        best_model_state_dict
        if selection["selected_metric"] == "val_loss" and best_model_state_dict is not None
        else final_model_state_dict
    )
    checkpoint = {
        "critic_config": asdict(critic_config),
        "critic_arch": args.critic_arch,
        "model_state_dict": selected_state_dict,
        "final_model_state_dict": final_model_state_dict,
        "training_args": vars(args),
        "data_dirs": [str(path) for path in data_dirs],
        "num_samples": int(num_samples),
        "num_episodes": len(episodes),
        "num_train_samples": int(train_indices.numel()),
        "num_val_samples": int(val_indices.numel()),
        "task_feature_source": args.task_feature_source,
        "task_feature_key": args.task_feature_key,
        "task_feature_dim": 0 if task_features is None else int(task_features.shape[-1]),
        "target_mode": args.target_mode,
        "progress_key": args.progress_key,
        "success_weight": args.success_weight,
        "progress_weight": args.progress_weight,
        "retrieval_k": args.retrieval_k,
        "train_episode_indices": sorted(
            set(dataset["episode_indices"][train_indices].tolist())
        ),
        "val_episode_indices": sorted(set(dataset["episode_indices"][val_indices].tolist())),
        "episode_files": [str(path) for path in episode_files],
        "target_mean": float(targets.mean().item()),
        "target_success_fraction": float((targets > 0).float().mean().item()),
        "critic_arch": args.critic_arch,
        "history": history,
        **selection,
    }
    torch.save(checkpoint, output_dir / "critic.pt")

    metrics = {
        "num_samples": int(num_samples),
        "num_episodes": len(episodes),
        "num_train_samples": checkpoint["num_train_samples"],
        "num_val_samples": checkpoint["num_val_samples"],
        "task_feature_source": checkpoint["task_feature_source"],
        "task_feature_key": checkpoint["task_feature_key"],
        "task_feature_dim": checkpoint["task_feature_dim"],
        "target_mode": checkpoint["target_mode"],
        "progress_key": checkpoint["progress_key"],
        "success_weight": checkpoint["success_weight"],
        "progress_weight": checkpoint["progress_weight"],
        "retrieval_k": checkpoint["retrieval_k"],
        "train_episode_indices": checkpoint["train_episode_indices"],
        "val_episode_indices": checkpoint["val_episode_indices"],
        "target_mean": checkpoint["target_mean"],
        "target_success_fraction": checkpoint["target_success_fraction"],
        "critic_arch": checkpoint["critic_arch"],
        "final": history[-1] if history else {},
        "selected": selection,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
