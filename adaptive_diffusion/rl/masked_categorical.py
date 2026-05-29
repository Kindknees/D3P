"""Masked categorical distribution helpers."""

from __future__ import annotations

import torch
from torch.distributions import Categorical


class MaskedCategorical:
    """Categorical distribution with invalid actions masked out."""

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor) -> None:
        if logits.shape != mask.shape:
            raise ValueError("logits and mask must have the same shape")
        mask = mask.to(dtype=torch.bool, device=logits.device)
        if not torch.all(mask.any(dim=-1)):
            raise ValueError("each distribution row must have at least one valid action")
        masked_logits = logits.masked_fill(~mask, -1.0e9)
        self.mask = mask
        self.distribution = Categorical(logits=masked_logits)

    @property
    def probs(self) -> torch.Tensor:
        return self.distribution.probs

    def sample(self) -> torch.Tensor:
        return self.distribution.sample()

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()
