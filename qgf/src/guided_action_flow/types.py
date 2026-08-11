from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class PolicyInput:
    """Normalized input expected by a chunk policy adapter."""

    observation: Mapping[str, Any]
    instruction: str
    proprio: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyOutput:
    """Action chunk and optional internals returned by a policy adapter."""

    action_chunk: Any
    obs_features: Any | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Transition:
    observation: Mapping[str, Any]
    action: Any
    reward: float
    next_observation: Mapping[str, Any]
    done: bool
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardOutput:
    reward: float
    diagnostics: Mapping[str, float] = field(default_factory=dict)

