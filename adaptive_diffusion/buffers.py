"""Action-buffer primitives for adaptive diffusion execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ActionBuffer:
    """Mutable FIFO buffer for one environment's remaining action chunk."""

    actions: np.ndarray

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions)
        if actions.ndim != 2:
            raise ValueError("ActionBuffer actions must have shape [T, action_dim].")
        self.actions = actions.copy()

    @classmethod
    def empty(cls, action_dim: int, dtype=np.float32) -> "ActionBuffer":
        return cls(np.empty((0, action_dim), dtype=dtype))

    @property
    def remaining_length(self) -> int:
        return int(self.actions.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    def is_empty(self) -> bool:
        return self.remaining_length == 0

    def first(self) -> np.ndarray:
        if self.is_empty():
            raise ValueError("Cannot read first action from an empty buffer.")
        return self.actions[0].copy()

    def consume_one(self) -> np.ndarray:
        action = self.first()
        self.actions = self.actions[1:].copy()
        return action

    def replace(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions)
        if actions.ndim != 2:
            raise ValueError("Replacement actions must have shape [T, action_dim].")
        if actions.shape[1] != self.action_dim:
            raise ValueError(
                f"Replacement action_dim {actions.shape[1]} does not match "
                f"buffer action_dim {self.action_dim}."
            )
        self.actions = actions.copy()

    def copy(self) -> "ActionBuffer":
        return ActionBuffer(self.actions.copy())


def pad_residual_buffer(residual_actions: np.ndarray, target_len: int) -> np.ndarray:
    """Pad a residual action buffer by repeating its final action."""

    residual_actions = np.asarray(residual_actions)
    if residual_actions.ndim != 2:
        raise ValueError("Residual actions must have shape [T, action_dim].")
    if target_len <= 0:
        raise ValueError("target_len must be positive.")
    if len(residual_actions) == 0:
        raise ValueError("Cannot repair an empty residual buffer.")
    if len(residual_actions) >= target_len:
        return residual_actions[:target_len].copy()

    last = residual_actions[-1:]
    pad = np.repeat(last, target_len - len(residual_actions), axis=0)
    return np.concatenate([residual_actions, pad], axis=0)


def first_action_jump(old_actions: np.ndarray, new_actions: np.ndarray) -> float:
    """L2 jump between first actions of two non-empty buffers."""

    old_actions = np.asarray(old_actions)
    new_actions = np.asarray(new_actions)
    if len(old_actions) == 0 or len(new_actions) == 0:
        raise ValueError("Cannot compute action jump for an empty buffer.")
    return float(np.linalg.norm(new_actions[0] - old_actions[0]))


def mean_overlap_jump(old_actions: np.ndarray, new_actions: np.ndarray) -> float:
    """Mean L2 jump over the overlapping prefix of two buffers."""

    old_actions = np.asarray(old_actions)
    new_actions = np.asarray(new_actions)
    overlap = min(len(old_actions), len(new_actions))
    if overlap == 0:
        raise ValueError("Cannot compute buffer jump without overlap.")
    jumps = np.linalg.norm(new_actions[:overlap] - old_actions[:overlap], axis=1)
    return float(np.mean(jumps))
