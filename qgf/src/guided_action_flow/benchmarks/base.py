from __future__ import annotations

from typing import Any, Protocol


class BenchmarkEnvAdapter(Protocol):
    """Minimal interface shared by benchmark adapters."""

    @property
    def task_instruction(self) -> str:
        """Natural-language task instruction."""

    @property
    def action_dim(self) -> int:
        """Continuous action dimension expected by the environment."""

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Reset the environment and return a normalized observation."""

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Step the environment with one low-level action."""

    def close(self) -> None:
        """Release simulator resources."""

