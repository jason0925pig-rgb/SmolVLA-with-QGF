from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from guided_action_flow.types import RewardOutput, Transition


@dataclass(frozen=True)
class LiberoSparseSuccessReward:
    """Sparse success reward wrapper for LIBERO-style `info` dictionaries."""

    success_keys: tuple[str, ...] = ("success", "is_success")
    success_reward: float = 1.0
    failure_reward: float = 0.0
    terminal_only: bool = True

    def __init__(
        self,
        success_keys: Iterable[str] = ("success", "is_success"),
        success_reward: float = 1.0,
        failure_reward: float = 0.0,
        terminal_only: bool = True,
    ) -> None:
        object.__setattr__(self, "success_keys", tuple(success_keys))
        object.__setattr__(self, "success_reward", float(success_reward))
        object.__setattr__(self, "failure_reward", float(failure_reward))
        object.__setattr__(self, "terminal_only", bool(terminal_only))

    def __call__(self, transition: Transition) -> RewardOutput:
        success = self._extract_success(transition)
        active = (not self.terminal_only) or transition.done or success
        reward = self.success_reward if active and success else self.failure_reward
        return RewardOutput(
            reward=reward,
            diagnostics={
                "success": float(success),
                "terminal": float(transition.done),
                "reward/libero_sparse_success": reward,
            },
        )

    def _extract_success(self, transition: Transition) -> bool:
        for key in self.success_keys:
            if key in transition.info:
                return bool(transition.info[key])
        return False

