"""Replan-only PPO controller wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from adaptive_diffusion.buffers import ActionBuffer
from adaptive_diffusion.controllers.fixed import ControllerDecision
from adaptive_diffusion.rl.actor_critic import ReplanActorCritic
from adaptive_diffusion.rl.features import (
    ReplanFeatureConfig,
    build_replan_features,
    replan_action_mask,
)


@dataclass(frozen=True)
class PPOReplanDecisionInfo:
    obs_features: np.ndarray
    action_mask: np.ndarray
    action: int
    logprob: float
    value: float
    entropy: float
    probs: np.ndarray


class PPOReplanController:
    """Binary learned controller that decides continue vs replan."""

    name = "ppo_replan_only"

    def __init__(
        self,
        model: ReplanActorCritic,
        feature_config: ReplanFeatureConfig,
        deterministic: bool = True,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.feature_config = feature_config
        self.deterministic = bool(deterministic)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.last_decision_info: PPOReplanDecisionInfo | None = None

    def decide(
        self,
        buffer: ActionBuffer,
        obs: Mapping[str, Any] | None = None,
        uncertainty: Any = None,
    ) -> ControllerDecision:
        if obs is None:
            raise ValueError("PPOReplanController requires obs features")
        features = build_replan_features(
            obs=obs,
            buffer=buffer,
            config=self.feature_config,
            uncertainty=uncertainty,
        )
        action_mask = replan_action_mask(buffer)
        feature_tensor = torch.from_numpy(features).float().to(self.device)
        mask_tensor = torch.from_numpy(action_mask).to(self.device)
        with torch.no_grad():
            output = self.model.act(
                feature_tensor,
                mask_tensor,
                deterministic=self.deterministic,
            )
        action = int(output["action"].detach().cpu().reshape(-1)[0].item())
        self.last_decision_info = PPOReplanDecisionInfo(
            obs_features=features.copy(),
            action_mask=action_mask.copy(),
            action=action,
            logprob=float(output["logprob"].detach().cpu().reshape(-1)[0].item()),
            value=float(output["value"].detach().cpu().reshape(-1)[0].item()),
            entropy=float(output["entropy"].detach().cpu().reshape(-1)[0].item()),
            probs=output["probs"].detach().cpu().numpy().reshape(-1).astype(np.float32),
        )
        return ControllerDecision(action=action, forced_replan=buffer.is_empty)
