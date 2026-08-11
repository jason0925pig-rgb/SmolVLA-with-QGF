from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerActionChunkCriticConfig:
    obs_feature_dim: int
    action_dim: int
    action_horizon: int
    proprio_dim: int = 0
    task_feature_dim: int = 0
    d_model: int = 256
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    ff_multiplier: int = 4


def _build_module(config: TransformerActionChunkCriticConfig):
    import torch
    import torch.nn as nn

    class TransformerActionChunkCriticModule(nn.Module):
        def __init__(self, cfg: TransformerActionChunkCriticConfig):
            super().__init__()
            if cfg.d_model % cfg.num_heads != 0:
                raise ValueError("d_model must be divisible by num_heads.")
            self.cfg = cfg
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            self.obs_proj = nn.Linear(cfg.obs_feature_dim, cfg.d_model)
            self.proprio_proj = (
                nn.Linear(cfg.proprio_dim, cfg.d_model) if cfg.proprio_dim > 0 else None
            )
            self.task_proj = (
                nn.Linear(cfg.task_feature_dim, cfg.d_model)
                if cfg.task_feature_dim > 0
                else None
            )
            self.action_proj = nn.Linear(cfg.action_dim, cfg.d_model)
            self.special_pos = nn.Parameter(torch.zeros(1, 4, cfg.d_model))
            self.action_pos = nn.Parameter(torch.zeros(1, cfg.action_horizon, cfg.d_model))

            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.num_heads,
                dim_feedforward=cfg.d_model * cfg.ff_multiplier,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(cfg.d_model),
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.SiLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model, 1),
            )
            self._reset_parameters()

        def _reset_parameters(self):
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.special_pos, std=0.02)
            nn.init.normal_(self.action_pos, std=0.02)

        def forward(self, obs_features, action_chunk, proprio=None, task_features=None):
            batch_size = action_chunk.shape[0]
            tokens = []
            cls = self.cls_token.expand(batch_size, -1, -1) + self.special_pos[:, 0:1, :]
            tokens.append(cls)
            obs = self.obs_proj(obs_features.reshape(batch_size, -1)).unsqueeze(1)
            tokens.append(obs + self.special_pos[:, 1:2, :])
            if self.proprio_proj is not None:
                if proprio is None:
                    raise ValueError("proprio is required by this critic checkpoint.")
                proprio_token = self.proprio_proj(proprio.reshape(batch_size, -1)).unsqueeze(1)
                tokens.append(proprio_token + self.special_pos[:, 2:3, :])
            if self.task_proj is not None:
                if task_features is None:
                    raise ValueError("task_features are required by this critic checkpoint.")
                task_features = task_features.reshape(batch_size, -1)
                if task_features.shape[-1] != self.cfg.task_feature_dim:
                    raise ValueError(
                        "task_features last dimension does not match critic config: "
                        f"{task_features.shape[-1]} != {self.cfg.task_feature_dim}."
                    )
                task_token = self.task_proj(task_features).unsqueeze(1)
                tokens.append(task_token + self.special_pos[:, 3:4, :])

            action_tokens = self.action_proj(action_chunk)
            action_tokens = action_tokens + self.action_pos[:, : action_tokens.shape[1], :]
            tokens.append(action_tokens)
            encoded = self.encoder(torch.cat(tokens, dim=1))
            return self.head(encoded[:, 0]).squeeze(-1)

    return TransformerActionChunkCriticModule(config)


class TransformerActionChunkCritic:
    """Transformer critic over state/task tokens and an ordered action chunk."""

    def __init__(self, config: TransformerActionChunkCriticConfig):
        self.config = config
        self.module = _build_module(config)

    def parameters(self):
        return self.module.parameters()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        return self.module.load_state_dict(state_dict)

    def __call__(self, obs_features, action_chunk, proprio=None, task_features=None):
        return self.module(
            obs_features=obs_features,
            action_chunk=action_chunk,
            proprio=proprio,
            task_features=task_features,
        )
