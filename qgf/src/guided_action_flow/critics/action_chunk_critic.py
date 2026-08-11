from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionChunkCriticConfig:
    obs_feature_dim: int
    action_dim: int
    action_horizon: int
    proprio_dim: int = 0
    task_feature_dim: int = 0
    hidden_dim: int = 512
    depth: int = 3


def build_mlp(input_dim: int, hidden_dim: int, depth: int):
    import torch.nn as nn

    if depth < 1:
        raise ValueError("depth must be >= 1")

    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.SiLU())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, 1))
    return nn.Sequential(*layers)


class ActionChunkCritic:
    """MLP critic over frozen observation features, proprio, and action chunks."""

    def __init__(self, config: ActionChunkCriticConfig):
        import torch.nn as nn

        self.config = config
        input_dim = (
            config.obs_feature_dim
            + config.proprio_dim
            + config.task_feature_dim
            + config.action_dim * config.action_horizon
        )
        self.module = build_mlp(input_dim, config.hidden_dim, config.depth)
        if not isinstance(self.module, nn.Module):
            raise TypeError("critic module must be a torch.nn.Module")

    def parameters(self):
        return self.module.parameters()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        return self.module.load_state_dict(state_dict)

    def __call__(self, obs_features, action_chunk, proprio=None, task_features=None):
        import torch

        batch_size = action_chunk.shape[0]
        flat_action = action_chunk.reshape(batch_size, -1)
        parts = [obs_features.reshape(batch_size, -1)]
        if proprio is not None:
            parts.append(proprio.reshape(batch_size, -1))
        if self.config.task_feature_dim > 0:
            if task_features is None:
                raise ValueError("task_features are required by this critic checkpoint.")
            task_features = task_features.reshape(batch_size, -1)
            if task_features.shape[-1] != self.config.task_feature_dim:
                raise ValueError(
                    "task_features last dimension does not match critic config: "
                    f"{task_features.shape[-1]} != {self.config.task_feature_dim}."
                )
            parts.append(task_features)
        x = torch.cat([*parts, flat_action], dim=-1)
        return self.module(x).squeeze(-1)
