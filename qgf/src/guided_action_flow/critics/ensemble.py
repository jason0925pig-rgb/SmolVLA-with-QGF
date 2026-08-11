from __future__ import annotations


class CriticEnsemble:
    """Small ensemble wrapper for uncertainty-aware Q guidance."""

    def __init__(self, critics):
        if not critics:
            raise ValueError("CriticEnsemble requires at least one critic.")
        self.critics = tuple(critics)

    def values(self, *args, **kwargs):
        import torch

        return torch.stack([critic(*args, **kwargs) for critic in self.critics], dim=0)

    def mean(self, *args, **kwargs):
        return self.values(*args, **kwargs).mean(dim=0)

    def disagreement(self, *args, **kwargs):
        return self.values(*args, **kwargs).std(dim=0)

