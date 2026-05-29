"""Rollout storage for replan-only PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from adaptive_diffusion.rl.advantage import compute_gae


@dataclass
class RolloutBatch:
    obs_features: np.ndarray
    actions: np.ndarray
    logprobs: np.ndarray
    values: np.ndarray
    rewards_hl: np.ndarray
    rewards_env: np.ndarray
    dones: np.ndarray
    action_masks: np.ndarray
    replanned: np.ndarray
    denoise_steps: np.ndarray
    nfe: np.ndarray
    discarded_actions: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


class ReplanRolloutBuffer:
    """Fixed-size rollout buffer for binary high-level replanning steps."""

    def __init__(self, capacity: int, feature_dim: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.reset()

    def reset(self) -> None:
        self.ptr = 0
        self.obs_features = np.zeros((self.capacity, self.feature_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.logprobs = np.zeros((self.capacity,), dtype=np.float32)
        self.values = np.zeros((self.capacity,), dtype=np.float32)
        self.rewards_hl = np.zeros((self.capacity,), dtype=np.float32)
        self.rewards_env = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=bool)
        self.action_masks = np.zeros((self.capacity, 2), dtype=bool)
        self.replanned = np.zeros((self.capacity,), dtype=bool)
        self.denoise_steps = np.zeros((self.capacity,), dtype=np.int64)
        self.nfe = np.zeros((self.capacity,), dtype=np.int64)
        self.discarded_actions = np.zeros((self.capacity,), dtype=np.int64)
        self.advantages = np.zeros((self.capacity,), dtype=np.float32)
        self.returns = np.zeros((self.capacity,), dtype=np.float32)

    @property
    def size(self) -> int:
        return self.ptr

    @property
    def full(self) -> bool:
        return self.ptr >= self.capacity

    def add(
        self,
        obs_features: np.ndarray,
        action: int,
        logprob: float,
        value: float,
        reward_hl: float,
        reward_env: float,
        done: bool,
        action_mask: np.ndarray,
        replanned: bool,
        denoise_steps: int,
        nfe: int,
        discarded_actions: int,
    ) -> None:
        if self.full:
            raise IndexError("rollout buffer is full")
        features = np.asarray(obs_features, dtype=np.float32).reshape(-1)
        if features.shape != (self.feature_dim,):
            raise ValueError(
                f"obs_features shape {features.shape} does not match ({self.feature_dim},)"
            )
        mask = np.asarray(action_mask, dtype=bool).reshape(-1)
        if mask.shape != (2,):
            raise ValueError("action_mask must have shape [2]")
        i = self.ptr
        self.obs_features[i] = features
        self.actions[i] = int(action)
        self.logprobs[i] = float(logprob)
        self.values[i] = float(value)
        self.rewards_hl[i] = float(reward_hl)
        self.rewards_env[i] = float(reward_env)
        self.dones[i] = bool(done)
        self.action_masks[i] = mask
        self.replanned[i] = bool(replanned)
        self.denoise_steps[i] = int(denoise_steps)
        self.nfe[i] = int(nfe)
        self.discarded_actions[i] = int(discarded_actions)
        self.ptr += 1

    def compute_returns_and_advantages(
        self,
        next_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        advantages, returns = compute_gae(
            rewards=self.rewards_hl[: self.ptr],
            values=self.values[: self.ptr],
            dones=self.dones[: self.ptr],
            next_value=next_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        self.advantages[: self.ptr] = advantages
        self.returns[: self.ptr] = returns

    def as_batch(self, normalize_advantages: bool = False) -> RolloutBatch:
        advantages = self.advantages[: self.ptr].copy()
        if normalize_advantages and len(advantages) > 1:
            std = float(advantages.std())
            if std > 1e-8:
                advantages = (advantages - float(advantages.mean())) / (std + 1e-8)
        return RolloutBatch(
            obs_features=self.obs_features[: self.ptr].copy(),
            actions=self.actions[: self.ptr].copy(),
            logprobs=self.logprobs[: self.ptr].copy(),
            values=self.values[: self.ptr].copy(),
            rewards_hl=self.rewards_hl[: self.ptr].copy(),
            rewards_env=self.rewards_env[: self.ptr].copy(),
            dones=self.dones[: self.ptr].copy(),
            action_masks=self.action_masks[: self.ptr].copy(),
            replanned=self.replanned[: self.ptr].copy(),
            denoise_steps=self.denoise_steps[: self.ptr].copy(),
            nfe=self.nfe[: self.ptr].copy(),
            discarded_actions=self.discarded_actions[: self.ptr].copy(),
            advantages=advantages,
            returns=self.returns[: self.ptr].copy(),
        )
