"""Transformer critic for a state, two-view visual tokens, and action chunk.

The visual input is deliberately token-level rather than a flattened image
feature.  It is produced by the *frozen* deployed SmolVLA image encoder and
connector, so the critic sees the same two views during offline IQL training
and eventual Q-guided policy inference.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisualTransformerCriticConfig:
    state_dim: int
    action_dim: int
    action_horizon: int
    visual_token_dim: int = 960
    visual_tokens: int = 128
    d_model: int = 256
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    ff_multiplier: int = 4


def _build_module(config: VisualTransformerCriticConfig):
    import torch
    import torch.nn as nn

    class VisualTransformerCriticModule(nn.Module):
        def __init__(self, cfg: VisualTransformerCriticConfig):
            super().__init__()
            if cfg.d_model % cfg.num_heads != 0:
                raise ValueError("d_model must be divisible by num_heads.")
            self.cfg = cfg
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            self.state_proj = nn.Sequential(nn.LayerNorm(cfg.state_dim), nn.Linear(cfg.state_dim, cfg.d_model))
            self.visual_proj = nn.Sequential(
                nn.LayerNorm(cfg.visual_token_dim),
                nn.Linear(cfg.visual_token_dim, cfg.d_model),
            )
            self.action_proj = nn.Sequential(nn.LayerNorm(cfg.action_dim), nn.Linear(cfg.action_dim, cfg.d_model))
            self.type_embedding = nn.Parameter(torch.zeros(1, 4, cfg.d_model))
            self.visual_position = nn.Parameter(torch.zeros(1, cfg.visual_tokens, cfg.d_model))
            self.action_position = nn.Parameter(torch.zeros(1, cfg.action_horizon, cfg.d_model))
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
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.type_embedding, std=0.02)
            nn.init.normal_(self.visual_position, std=0.02)
            nn.init.normal_(self.action_position, std=0.02)

        def _encode_base(self, state, visual_tokens):
            if state.ndim != 2 or state.shape[-1] != self.cfg.state_dim:
                raise ValueError(f"Expected state [B,{self.cfg.state_dim}], got {tuple(state.shape)}.")
            if visual_tokens.ndim != 3 or tuple(visual_tokens.shape[1:]) != (
                self.cfg.visual_tokens,
                self.cfg.visual_token_dim,
            ):
                raise ValueError(
                    "Expected visual tokens "
                    f"[B,{self.cfg.visual_tokens},{self.cfg.visual_token_dim}], got {tuple(visual_tokens.shape)}."
                )
            batch_size = state.shape[0]
            cls = self.cls_token.expand(batch_size, -1, -1) + self.type_embedding[:, 0:1]
            state_token = self.state_proj(state).unsqueeze(1) + self.type_embedding[:, 1:2]
            visual = self.visual_proj(visual_tokens.float())
            visual = visual + self.visual_position + self.type_embedding[:, 2:3]
            return [cls, state_token, visual]

        def forward_value(self, state, visual_tokens):
            encoded = self.encoder(torch.cat(self._encode_base(state, visual_tokens), dim=1))
            return self.head(encoded[:, 0]).squeeze(-1)

        def forward(self, state, visual_tokens, action_chunk):
            if action_chunk.ndim != 3 or tuple(action_chunk.shape[1:]) != (
                self.cfg.action_horizon,
                self.cfg.action_dim,
            ):
                raise ValueError(
                    "Expected action chunk "
                    f"[B,{self.cfg.action_horizon},{self.cfg.action_dim}], got {tuple(action_chunk.shape)}."
                )
            action = self.action_proj(action_chunk.float())
            action = action + self.action_position + self.type_embedding[:, 3:4]
            encoded = self.encoder(torch.cat([*self._encode_base(state, visual_tokens), action], dim=1))
            return self.head(encoded[:, 0]).squeeze(-1)

    return VisualTransformerCriticModule(config)


class VisualTransformerCritic:
    """Thin wrapper matching the other QGF critic interfaces."""

    def __init__(self, config: VisualTransformerCriticConfig):
        self.config = config
        self.module = _build_module(config)

    def parameters(self):
        return self.module.parameters()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        return self.module.load_state_dict(state_dict)

    def __call__(self, state, visual_tokens, action_chunk):
        return self.module(state, visual_tokens, action_chunk)
