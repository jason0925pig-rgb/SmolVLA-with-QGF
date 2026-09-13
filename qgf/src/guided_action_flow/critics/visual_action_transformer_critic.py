"""Transformer critic for the no-state Q(visual-tokens, action-chunk) ablation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisualActionTransformerCriticConfig:
    action_dim: int
    action_horizon: int
    visual_token_dim: int = 960
    visual_tokens: int = 128
    d_model: int = 256
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    ff_multiplier: int = 4


def _build_module(config: VisualActionTransformerCriticConfig):
    import torch
    import torch.nn as nn

    class VisualActionTransformerCriticModule(nn.Module):
        def __init__(self, cfg: VisualActionTransformerCriticConfig):
            super().__init__()
            if cfg.d_model % cfg.num_heads:
                raise ValueError("d_model must be divisible by num_heads")
            self.cfg = cfg
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            self.visual_proj = nn.Sequential(nn.LayerNorm(cfg.visual_token_dim), nn.Linear(cfg.visual_token_dim, cfg.d_model))
            self.action_proj = nn.Sequential(nn.LayerNorm(cfg.action_dim), nn.Linear(cfg.action_dim, cfg.d_model))
            self.type_embedding = nn.Parameter(torch.zeros(1, 3, cfg.d_model))
            self.visual_position = nn.Parameter(torch.zeros(1, cfg.visual_tokens, cfg.d_model))
            self.action_position = nn.Parameter(torch.zeros(1, cfg.action_horizon, cfg.d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model, nhead=cfg.num_heads, dim_feedforward=cfg.d_model * cfg.ff_multiplier,
                dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
            self.head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model), nn.SiLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, 1))
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.type_embedding, std=0.02)
            nn.init.normal_(self.visual_position, std=0.02)
            nn.init.normal_(self.action_position, std=0.02)

        def _prefix(self, visual_tokens):
            if visual_tokens.ndim != 3 or tuple(visual_tokens.shape[1:]) != (self.cfg.visual_tokens, self.cfg.visual_token_dim):
                raise ValueError(f"Expected visual tokens [B,{self.cfg.visual_tokens},{self.cfg.visual_token_dim}], got {tuple(visual_tokens.shape)}")
            batch = visual_tokens.shape[0]
            cls = self.cls_token.expand(batch, -1, -1) + self.type_embedding[:, 0:1]
            visual = self.visual_proj(visual_tokens.float()) + self.visual_position + self.type_embedding[:, 1:2]
            return [cls, visual]

        def forward_value(self, visual_tokens):
            encoded = self.encoder(torch.cat(self._prefix(visual_tokens), dim=1))
            return self.head(encoded[:, 0]).squeeze(-1)

        def forward(self, visual_tokens, action_chunk):
            if action_chunk.ndim != 3 or tuple(action_chunk.shape[1:]) != (self.cfg.action_horizon, self.cfg.action_dim):
                raise ValueError(f"Expected action chunk [B,{self.cfg.action_horizon},{self.cfg.action_dim}], got {tuple(action_chunk.shape)}")
            action = self.action_proj(action_chunk.float()) + self.action_position + self.type_embedding[:, 2:3]
            encoded = self.encoder(torch.cat([*self._prefix(visual_tokens), action], dim=1))
            return self.head(encoded[:, 0]).squeeze(-1)

    return VisualActionTransformerCriticModule(config)


class VisualActionTransformerCritic:
    """Thin runtime wrapper matching the QGF checkpoint loader interface."""

    def __init__(self, config: VisualActionTransformerCriticConfig):
        self.config = config
        self.module = _build_module(config)

    def parameters(self):
        return self.module.parameters()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        return self.module.load_state_dict(state_dict)

    def __call__(self, visual_tokens, action_chunk):
        return self.module(visual_tokens, action_chunk)
