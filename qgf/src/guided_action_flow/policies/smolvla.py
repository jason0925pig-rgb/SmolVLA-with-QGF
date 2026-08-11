from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from guided_action_flow.types import PolicyInput, PolicyOutput


@dataclass(frozen=True)
class SmolVLAConfig:
    model_id: str = "lerobot/smolvla_base"
    device: str = "cuda"
    dtype: str = "bfloat16"


class SmolVLAAdapter:
    """Adapter for a loaded LeRobot SmolVLA policy object.

    This wrapper avoids hard-coding a LeRobot API before the pinned checkout is
    inspected. It supports dependency injection: once SmolVLA is loaded by the
    experiment script, pass the concrete object here and map its action method.
    """

    def __init__(self, config: SmolVLAConfig, policy: Any | None = None):
        self.config = config
        self.policy = policy

    @classmethod
    def from_pretrained(cls, config: SmolVLAConfig) -> "SmolVLAAdapter":
        raise NotImplementedError(
            "Inspect the pinned LeRobot checkout, then implement SmolVLA loading here."
        )

    def act_chunk(self, policy_input: PolicyInput) -> PolicyOutput:
        if self.policy is None:
            raise RuntimeError("SmolVLA policy is not loaded.")

        if hasattr(self.policy, "act_chunk"):
            action_chunk = self.policy.act_chunk(policy_input)
        elif hasattr(self.policy, "select_action"):
            action_chunk = self.policy.select_action(policy_input)
        else:
            raise AttributeError(
                "Loaded SmolVLA policy has no known action method. "
                "Inspect the pinned LeRobot API and update SmolVLAAdapter."
            )

        return PolicyOutput(action_chunk=action_chunk)

