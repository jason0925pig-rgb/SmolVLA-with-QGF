from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeMetrics:
    episode_index: int
    success: bool
    total_reward: float
    length: int

