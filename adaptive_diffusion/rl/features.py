"""Feature helpers for the replan-only PPO controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from adaptive_diffusion.buffers import ActionBuffer


@dataclass(frozen=True)
class ReplanFeatureConfig:
    obs_dim: int
    action_dim: int
    horizon: int

    def __post_init__(self) -> None:
        if self.obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")

    @property
    def buffer_summary_dim(self) -> int:
        return self.horizon * self.action_dim + self.horizon

    @property
    def input_dim(self) -> int:
        return self.obs_dim + self.buffer_summary_dim + 3


def flatten_state_obs(obs: Mapping[str, Any], obs_dim: int) -> np.ndarray:
    if "state" not in obs:
        raise KeyError("obs must contain key 'state'")
    state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
    if state.shape != (obs_dim,):
        raise ValueError(f"state obs shape {state.shape} does not match ({obs_dim},)")
    return state


def uncertainty_value(uncertainty: Any) -> float:
    if uncertainty is None:
        return 0.0
    if hasattr(uncertainty, "uncertainty"):
        return float(uncertainty.uncertainty)
    if isinstance(uncertainty, Mapping):
        return float(uncertainty.get("uncertainty", 0.0))
    return float(uncertainty)


def build_replan_features(
    obs: Mapping[str, Any],
    buffer: ActionBuffer,
    config: ReplanFeatureConfig,
    uncertainty: Any = None,
) -> np.ndarray:
    """Build x_t = [state, buffer summary, remaining/H, age/H, uncertainty]."""

    state = flatten_state_obs(obs, obs_dim=config.obs_dim)
    buffer_summary = buffer.summary(
        action_dim=config.action_dim,
        max_horizon=config.horizon,
    )
    scalars = np.asarray(
        [
            buffer.remaining / config.horizon,
            buffer.age / config.horizon,
            uncertainty_value(uncertainty),
        ],
        dtype=np.float32,
    )
    features = np.concatenate([state, buffer_summary, scalars], axis=0).astype(np.float32)
    if features.shape != (config.input_dim,):
        raise ValueError(f"feature shape {features.shape} does not match ({config.input_dim},)")
    return features


def replan_action_mask(buffer: ActionBuffer) -> np.ndarray:
    """Return valid action mask for [continue, replan]."""

    if buffer.is_empty:
        return np.asarray([False, True], dtype=bool)
    return np.asarray([True, True], dtype=bool)
