from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LiberoConfig:
    suite: str | None = None
    task: str | None = None
    seed: int = 0
    horizon: int | None = None
    repo_root: str = "third_party/LIBERO"
    data_root: str = "data/libero"


class LiberoAdapter:
    """Thin adapter around the installed LIBERO environment.

    The exact construction path depends on the pinned LIBERO checkout. Keep
    LIBERO-specific imports lazy so the rest of the package stays importable on
    machines that only edit code.
    """

    def __init__(self, config: LiberoConfig):
        self.config = config
        self._env: Any | None = None
        self._task_instruction = ""
        self._action_dim: int | None = None

    @property
    def task_instruction(self) -> str:
        return self._task_instruction

    @property
    def action_dim(self) -> int:
        if self._action_dim is None:
            raise RuntimeError("LIBERO env is not built yet; call build() first.")
        return self._action_dim

    def build(self) -> None:
        """Construct the LIBERO env after inspecting the installed API.

        This method is intentionally explicit instead of guessing LIBERO import
        paths. The first implementation step on the experiment machine should
        inspect `third_party/LIBERO` and wire this method to the pinned version.
        """

        raise NotImplementedError(
            "Wire LiberoAdapter.build() to the pinned LIBERO checkout before running."
        )

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if self._env is None:
            self.build()
        if self._env is None:
            raise RuntimeError("LIBERO env did not initialize.")
        raw_obs = self._env.reset(seed=seed) if seed is not None else self._env.reset()
        return self.normalize_observation(raw_obs)

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self._env is None:
            raise RuntimeError("Call reset() before step().")
        next_obs, reward, done, info = self._env.step(action)
        return self.normalize_observation(next_obs), float(reward), bool(done), dict(info)

    def normalize_observation(self, raw_obs: Mapping[str, Any]) -> dict[str, Any]:
        return {"raw": raw_obs}

    def close(self) -> None:
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()

