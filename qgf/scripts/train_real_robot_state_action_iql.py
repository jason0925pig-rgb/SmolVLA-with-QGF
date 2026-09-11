#!/usr/bin/env python3
"""Train the no-vision state/action IQL critic for the real-robot QGF ablation.

The input ``.pt`` episode files are the frozen visual-critic feature cache so
that sample alignment, terminal labels, action chunks, and episode split are
*identical* to the deployed full-Q training run.  Visual tensors are never
loaded into a model or passed through the loss: this script implements
Q(s, a_chunk), not Q(s, z, a_chunk).
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class StateActionTransformerCriticConfig:
    state_dim: int
    action_dim: int
    action_horizon: int
    d_model: int = 256
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    ff_multiplier: int = 4


def _build_module(config: StateActionTransformerCriticConfig):
    import torch
    import torch.nn as nn

    class StateActionTransformerCriticModule(nn.Module):
        def __init__(self, cfg: StateActionTransformerCriticConfig):
            super().__init__()
            if cfg.d_model % cfg.num_heads:
                raise ValueError("d_model must be divisible by num_heads")
            self.cfg = cfg
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            self.state_proj = nn.Sequential(nn.LayerNorm(cfg.state_dim), nn.Linear(cfg.state_dim, cfg.d_model))
            self.action_proj = nn.Sequential(nn.LayerNorm(cfg.action_dim), nn.Linear(cfg.action_dim, cfg.d_model))
            self.type_embedding = nn.Parameter(torch.zeros(1, 3, cfg.d_model))
            self.action_position = nn.Parameter(torch.zeros(1, cfg.action_horizon, cfg.d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model, nhead=cfg.num_heads,
                dim_feedforward=cfg.d_model * cfg.ff_multiplier,
                dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model),
                nn.SiLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, 1),
            )
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.type_embedding, std=0.02)
            nn.init.normal_(self.action_position, std=0.02)

        def _prefix(self, state):
            if state.ndim != 2 or state.shape[-1] != self.cfg.state_dim:
                raise ValueError(f"Expected state [B,{self.cfg.state_dim}], got {tuple(state.shape)}")
            batch = state.shape[0]
            cls = self.cls_token.expand(batch, -1, -1) + self.type_embedding[:, 0:1]
            state_token = self.state_proj(state.float()).unsqueeze(1) + self.type_embedding[:, 1:2]
            return [cls, state_token]

        def forward_value(self, state):
            encoded = self.encoder(torch.cat(self._prefix(state), dim=1))
            return self.head(encoded[:, 0]).squeeze(-1)

        def forward(self, state, action_chunk):
            if action_chunk.ndim != 3 or tuple(action_chunk.shape[1:]) != (
                self.cfg.action_horizon, self.cfg.action_dim
            ):
                raise ValueError(
                    f"Expected action chunk [B,{self.cfg.action_horizon},{self.cfg.action_dim}], "
                    f"got {tuple(action_chunk.shape)}"
                )
            action = self.action_proj(action_chunk.float())
            action = action + self.action_position + self.type_embedding[:, 2:3]
            encoded = self.encoder(torch.cat([*self._prefix(state), action], dim=1))
            return self.head(encoded[:, 0]).squeeze(-1)

    return StateActionTransformerCriticModule(config)


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--polyak", type=float, default=0.005)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-train-episodes", type=int, default=90)
    parser.add_argument("--expected-val-episodes", type=int, default=10)
    return parser.parse_args()


def _clone_to_cpu(state_dict):
    return {name: value.detach().cpu().clone() for name, value in state_dict.items()}


def _expectile_loss(advantage, expectile):
    import torch
    weight = torch.where(advantage > 0, expectile, 1.0 - expectile)
    return (weight * advantage.square()).mean()


def _load_episode(data_dir: Path, episode_index: int):
    import torch
    path = data_dir / f"episode_{episode_index:06d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"state", "action_chunk", "next_state", "reward", "success", "done", "terminated", "truncated"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    count = int(payload["state"].shape[0])
    if count < 1 or tuple(payload["state"].shape[1:]) != (8,) or tuple(payload["action_chunk"].shape[1:]) != (50, 8):
        raise ValueError(f"{path}: incompatible state/action shape")
    if any(int(payload[key].shape[0]) != count for key in required):
        raise ValueError(f"{path}: inconsistent row count")
    return payload


def _join(data_dir: Path, indices):
    import torch
    fields = ("state", "action_chunk", "next_state", "reward", "success", "done", "terminated", "truncated")
    columns = {field: [] for field in fields}
    for index in indices:
        payload = _load_episode(data_dir, index)
        for field in fields:
            columns[field].append(payload[field])
    data = {field: torch.cat(values, dim=0) for field, values in columns.items()}
    data["done"] = data["done"].bool() | data["terminated"].bool() | data["truncated"].bool() | data["success"].bool()
    data["reward"] = torch.maximum(data["reward"].float(), data["success"].float())
    return data


class _Rows:
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return int(self.data["state"].shape[0])
    def __getitem__(self, index):
        return tuple(self.data[key][index] for key in ("state", "action_chunk", "next_state", "reward", "done"))


def _evaluate(q_model, value_model, data, args, device):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    q_model.eval(); value_model.eval()
    losses, scores, positive = [], [], []
    with torch.inference_mode():
        for state, action, next_state, reward, done in DataLoader(_Rows(data), batch_size=args.batch_size):
            state, action, next_state = state.to(device), action.to(device), next_state.to(device)
            reward, done = reward.to(device), done.to(device).float()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = q_model(state, action)
                target = reward + args.gamma * (1.0 - done) * value_model.forward_value(next_state)
                losses.append(float(F.mse_loss(prediction.float(), target.float()).cpu()))
            scores.append(prediction.float().cpu()); positive.append((reward > 0).cpu())
    scores, positive = torch.cat(scores), torch.cat(positive)
    good, bad = scores[positive], scores[~positive]
    return {
        "td_loss": float(sum(losses) / len(losses)), "q_mean": float(scores.mean()),
        "q_success_mean": float(good.mean()) if good.numel() else None,
        "q_failure_mean": float(bad.mean()) if bad.numel() else None,
        "q_success_failure_gap": float(good.mean() - bad.mean()) if good.numel() and bad.numel() else None,
        "positive_reward_samples": int(positive.sum()), "samples": int(scores.numel()),
    }


def _train_member(member_index, args, train_data, val_data, config, device):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    seed = args.seed + member_index * 1009
    torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    q_model = _build_module(config).to(device)
    target_model = _build_module(config).to(device)
    target_model.load_state_dict(copy.deepcopy(q_model.state_dict()))
    for parameter in target_model.parameters(): parameter.requires_grad_(False)
    value_model = _build_module(config).to(device)
    opt_q = torch.optim.AdamW(q_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_v = torch.optim.AdamW(value_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(_Rows(train_data), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    best = {"val_td_loss": float("inf"), "epoch": None, "q_state_dict": None, "value_state_dict": None}
    history = []
    for epoch in range(1, args.epochs + 1):
        q_model.train(); value_model.train(); q_losses, v_losses = [], []
        for state, action, next_state, reward, done in loader:
            state, action, next_state = state.to(device), action.to(device), next_state.to(device)
            reward, done = reward.to(device), done.to(device).float()
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                target_q = target_model(state, action)
            value = value_model.forward_value(state)
            value_loss = _expectile_loss(target_q.float() - value.float(), args.expectile)
            opt_v.zero_grad(set_to_none=True); value_loss.backward(); torch.nn.utils.clip_grad_norm_(value_model.parameters(), 10.0); opt_v.step()
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                target = reward + args.gamma * (1.0 - done) * value_model.forward_value(next_state)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = q_model(state, action); q_loss = F.mse_loss(prediction.float(), target.float())
            opt_q.zero_grad(set_to_none=True); q_loss.backward(); torch.nn.utils.clip_grad_norm_(q_model.parameters(), 10.0); opt_q.step()
            with torch.no_grad():
                for target_parameter, parameter in zip(target_model.parameters(), q_model.parameters()):
                    target_parameter.mul_(1.0 - args.polyak).add_(parameter, alpha=args.polyak)
            q_losses.append(float(q_loss.detach().cpu())); v_losses.append(float(value_loss.detach().cpu()))
        metrics = _evaluate(q_model, value_model, val_data, args, device)
        row = {"epoch": epoch, "train_q_loss": float(sum(q_losses) / len(q_losses)), "train_v_loss": float(sum(v_losses) / len(v_losses)), **{f"val_{key}": value for key, value in metrics.items()}}
        history.append(row); print(json.dumps({"member": member_index, **row}), flush=True)
        if metrics["td_loss"] < best["val_td_loss"]:
            best = {"val_td_loss": metrics["td_loss"], "epoch": epoch, "q_state_dict": _clone_to_cpu(q_model.state_dict()), "value_state_dict": _clone_to_cpu(value_model.state_dict())}
    return best, history


def main():
    args = _args()
    import torch
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    train_ids, val_ids = [int(x) for x in split["train_episode_indices"]], [int(x) for x in split["val_episode_indices"]]
    if (
        len(train_ids) != args.expected_train_episodes
        or len(val_ids) != args.expected_val_episodes
        or set(train_ids) & set(val_ids)
    ):
        raise ValueError("Episode split does not match the declared fixed train/validation sizes")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, val = _join(args.data_dir, train_ids), _join(args.data_dir, val_ids)
    config = StateActionTransformerCriticConfig(state_dim=8, action_dim=8, action_horizon=50, d_model=args.d_model, num_layers=args.layers, num_heads=args.heads, dropout=args.dropout)
    metadata = {
        "format": "armstrong-qgf-state-action-iql-v1", "ablation": "no_visual_Q_s_a", "device": str(device),
        "critic_arch": "state_action_transformer", "critic_config": asdict(config),
        "train_episode_indices": train_ids, "val_episode_indices": val_ids,
        "train_samples": int(train["state"].shape[0]), "val_samples": int(val["state"].shape[0]),
        "train_positive_rewards": int((train["reward"] > 0).sum()), "val_positive_rewards": int((val["reward"] > 0).sum()),
        "training_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "provenance": "Uses exactly the full-Q frozen feature-cache rows and preserved episode split; visual tensors are not loaded into the critic.",
    }
    (args.output_dir / "training_input_summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    summaries = []
    for member in range(args.ensemble_size):
        best, history = _train_member(member, args, train, val, config, device)
        checkpoint = {**metadata, "model_state_dict": best["q_state_dict"], "value_model_state_dict": best["value_state_dict"], "selected_epoch": best["epoch"], "selected_val_td_loss": best["val_td_loss"], "history": history, "ensemble_member_index": member, "member_seed": args.seed + member * 1009}
        path = args.output_dir / f"critic_member_{member:02d}.pt"; torch.save(checkpoint, path)
        summaries.append({"member_index": member, "path": path.name, "selected_epoch": best["epoch"], "selected_val_td_loss": best["val_td_loss"]})
    (args.output_dir / "training_summary.json").write_text(json.dumps({**metadata, "members": summaries}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"TRAINING_COMPLETE": True, "members": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
