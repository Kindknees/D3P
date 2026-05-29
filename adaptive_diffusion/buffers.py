"""Action-buffer primitives for adaptive execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ActionBuffer:
    """Stores the unexecuted actions from the current diffusion plan."""

    actions: np.ndarray
    horizon: int
    age: int = 0
    plan_id: int = 0
    denoise_steps: int = 0
    source: str = "unknown"

    def __post_init__(self) -> None:
        self.actions = np.asarray(self.actions, dtype=np.float32)
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.actions.ndim != 2:
            raise ValueError("actions must have shape [remaining, action_dim]")
        if len(self.actions) > self.horizon:
            raise ValueError("actions cannot exceed horizon")

    @classmethod
    def empty(cls, horizon: int, action_dim: int) -> "ActionBuffer":
        """Create an empty buffer with a known action dimension."""

        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        return cls(actions=np.zeros((0, action_dim), dtype=np.float32), horizon=horizon)

    @property
    def remaining(self) -> int:
        """Number of actions left to execute."""

        return int(self.actions.shape[0])

    @property
    def is_empty(self) -> bool:
        """Whether the buffer has no remaining actions."""

        return self.remaining == 0

    def pop_first(self) -> np.ndarray:
        """Remove and return the next action.

        Raises:
            IndexError: If the buffer is empty.
        """

        if self.is_empty:
            raise IndexError("cannot pop from an empty action buffer")
        action = self.actions[0].copy()
        self.actions = self.actions[1:].copy()
        self.age += 1
        return action

    def replace(self, actions: np.ndarray, denoise_steps: int, source: str) -> None:
        """Replace the current plan and reset plan-local metadata."""

        next_actions = np.asarray(actions, dtype=np.float32)
        if next_actions.ndim != 2:
            raise ValueError("replacement actions must have shape [horizon, action_dim]")
        if len(next_actions) == 0:
            raise ValueError("replacement actions cannot be empty")
        if len(next_actions) > self.horizon:
            raise ValueError("replacement actions cannot exceed horizon")
        if next_actions.shape[1] != self.actions.shape[1]:
            raise ValueError(
                f"replacement action_dim {next_actions.shape[1]} does not match "
                f"buffer action_dim {self.actions.shape[1]}"
            )
        if denoise_steps < 0:
            raise ValueError("denoise_steps must be non-negative")

        self.actions = next_actions.copy()
        self.age = 0
        self.plan_id += 1
        self.denoise_steps = int(denoise_steps)
        self.source = str(source)

    def summary(self, action_dim: int, max_horizon: int) -> np.ndarray:
        """Return flattened padded actions plus a validity mask."""

        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if max_horizon <= 0:
            raise ValueError("max_horizon must be positive")
        if self.actions.shape[1] != action_dim:
            raise ValueError(
                f"summary action_dim {action_dim} does not match buffer action_dim "
                f"{self.actions.shape[1]}"
            )

        padded = np.zeros((max_horizon, action_dim), dtype=np.float32)
        mask = np.zeros((max_horizon,), dtype=np.float32)
        n_valid = min(self.remaining, max_horizon)
        if n_valid:
            padded[:n_valid] = self.actions[:n_valid]
            mask[:n_valid] = 1.0
        return np.concatenate([padded.reshape(-1), mask], axis=0)
