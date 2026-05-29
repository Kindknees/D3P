"""Actor-critic network for binary replanning decisions."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from adaptive_diffusion.rl.masked_categorical import MaskedCategorical


def _activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    raise ValueError("activation must be 'tanh' or 'relu'")


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str = "tanh",
) -> nn.Sequential:
    dims = [int(input_dim), *[int(dim) for dim in hidden_dims], int(output_dim)]
    layers: list[nn.Module] = []
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        if i < len(dims) - 2:
            layers.append(_activation(activation))
    return nn.Sequential(*layers)


class ReplanActorCritic(nn.Module):
    """Binary actor-critic used by replan-only PPO."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.activation = activation
        self.actor = build_mlp(
            input_dim,
            self.hidden_dims,
            output_dim=2,
            activation=activation,
        )
        self.critic = build_mlp(
            input_dim,
            self.hidden_dims,
            output_dim=1,
            activation=activation,
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"feature dim {features.shape[-1]} does not match {self.input_dim}"
            )
        logits = self.actor(features)
        values = self.critic(features).squeeze(-1)
        return logits, values

    def distribution(
        self,
        features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[MaskedCategorical, torch.Tensor]:
        logits, values = self.forward(features)
        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)
        return MaskedCategorical(logits=logits, mask=action_mask), values

    def act(
        self,
        features: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        dist, values = self.distribution(features, action_mask)
        if deterministic:
            actions = torch.argmax(dist.probs, dim=-1)
        else:
            actions = dist.sample()
        return {
            "action": actions,
            "logprob": dist.log_prob(actions),
            "entropy": dist.entropy(),
            "value": values,
            "probs": dist.probs,
        }

    def evaluate_actions(
        self,
        features: torch.Tensor,
        action_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        dist, values = self.distribution(features, action_mask)
        return {
            "logprob": dist.log_prob(actions),
            "entropy": dist.entropy(),
            "value": values,
            "probs": dist.probs,
        }
