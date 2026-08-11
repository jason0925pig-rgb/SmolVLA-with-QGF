from __future__ import annotations

from typing import Protocol

from guided_action_flow.types import PolicyInput, PolicyOutput


class ChunkPolicy(Protocol):
    """Policy interface for models that output action chunks."""

    def act_chunk(self, policy_input: PolicyInput) -> PolicyOutput:
        """Return an action chunk for the current observation and instruction."""

