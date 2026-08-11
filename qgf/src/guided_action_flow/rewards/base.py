from __future__ import annotations

from typing import Protocol

from guided_action_flow.types import RewardOutput, Transition


class RewardFunction(Protocol):
    """Reward interface used by rollout collection and critic training."""

    def __call__(self, transition: Transition) -> RewardOutput:
        """Compute reward and diagnostics for one transition."""

