"""Uncertainty signals used by adaptive replanning controllers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np


@dataclass(frozen=True)
class UncertaintyEstimate:
    """Decision-time uncertainty values for logging and controllers."""

    uncertainty: float = 0.0
    td_error: float = 0.0
    raw_abs_td_error: float = 0.0
    running_mean: float = 0.0
    running_std: float = 0.0
    count: int = 0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "uncertainty": float(self.uncertainty),
            "td_error": float(self.td_error),
            "raw_abs_td_error": float(self.raw_abs_td_error),
            "running_mean": float(self.running_mean),
            "running_std": float(self.running_std),
            "count": int(self.count),
        }


class UncertaintySignal(Protocol):
    """Interface for uncertainty signals consumed by the executor."""

    @property
    def value(self) -> float:
        """Most recent normalized uncertainty."""

    def reset(self) -> None:
        """Reset episode-local signal state."""

    def before_action(self, obs: Any, buffer: Any) -> UncertaintyEstimate:
        """Return the signal available before the controller acts."""

    def after_transition(
        self,
        obs: Any,
        action: np.ndarray,
        reward: float,
        next_obs: Any,
        done: bool,
        info: Any,
    ) -> UncertaintyEstimate:
        """Update the signal from the latest transition."""


class NullUncertaintySignal:
    """No-op uncertainty signal used by fixed baselines."""

    @property
    def value(self) -> float:
        return 0.0

    def reset(self) -> None:
        pass

    def before_action(self, obs: Any, buffer: Any) -> UncertaintyEstimate:
        return UncertaintyEstimate()

    def after_transition(
        self,
        obs: Any,
        action: np.ndarray,
        reward: float,
        next_obs: Any,
        done: bool,
        info: Any,
    ) -> UncertaintyEstimate:
        return UncertaintyEstimate()


class TDErrorSignal:
    """TD-error signal backed by a value function callable."""

    def __init__(
        self,
        value_fn: Callable[[Any], float],
        gamma: float = 0.99,
        eps: float = 1e-8,
    ) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.value_fn = value_fn
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.reset()

    @property
    def value(self) -> float:
        return self._estimate.uncertainty

    def reset(self) -> None:
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._estimate = UncertaintyEstimate()

    def before_action(self, obs: Any, buffer: Any) -> UncertaintyEstimate:
        return self._estimate

    def after_transition(
        self,
        obs: Any,
        action: np.ndarray,
        reward: float,
        next_obs: Any,
        done: bool,
        info: Any,
    ) -> UncertaintyEstimate:
        value = self._value(obs)
        next_value = 0.0 if done else self._value(next_obs)
        td_error = float(reward) + self.gamma * next_value * (1.0 - float(done)) - value
        abs_td_error = abs(td_error)
        self._update_stats(abs_td_error)
        running_std = self._std
        if running_std <= self.eps:
            normalized = 0.0
        else:
            normalized = (abs_td_error - self._mean) / (running_std + self.eps)
        self._estimate = UncertaintyEstimate(
            uncertainty=float(normalized),
            td_error=float(td_error),
            raw_abs_td_error=float(abs_td_error),
            running_mean=float(self._mean),
            running_std=float(running_std),
            count=int(self._count),
        )
        return self._estimate

    @property
    def _std(self) -> float:
        if self._count < 2:
            return 0.0
        return math.sqrt(self._m2 / (self._count - 1))

    def _update_stats(self, value: float) -> None:
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2

    def _value(self, obs: Any) -> float:
        return float(np.asarray(self.value_fn(obs), dtype=np.float32).reshape(-1)[0])
