from __future__ import annotations

from collections.abc import Sequence


def discounted_returns(rewards: Sequence[float], dones: Sequence[bool], gamma: float) -> list[float]:
    if len(rewards) != len(dones):
        raise ValueError("rewards and dones must have the same length.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")

    returns = [0.0 for _ in rewards]
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        if dones[idx]:
            running = 0.0
        running = float(rewards[idx]) + gamma * running
        returns[idx] = running
    return returns

