from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from guided_action_flow.rewards.base import RewardFunction
from guided_action_flow.types import RewardOutput, Transition


@dataclass(frozen=True)
class WeightedRewardTerm:
    name: str
    fn: RewardFunction
    weight: float = 1.0


class CompositeReward:
    """Weighted sum of reward terms with merged diagnostics."""

    def __init__(self, terms: Sequence[WeightedRewardTerm]):
        self.terms = tuple(terms)

    def __call__(self, transition: Transition) -> RewardOutput:
        total = 0.0
        diagnostics: dict[str, float] = {}
        for term in self.terms:
            output = term.fn(transition)
            weighted = term.weight * output.reward
            total += weighted
            diagnostics[f"reward/{term.name}"] = weighted
            diagnostics.update(output.diagnostics)
        diagnostics["reward/total"] = total
        return RewardOutput(reward=total, diagnostics=diagnostics)

