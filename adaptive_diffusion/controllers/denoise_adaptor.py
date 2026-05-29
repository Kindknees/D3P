"""Rule-based adaptive denoising controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from adaptive_diffusion.buffers import ActionBuffer
from adaptive_diffusion.controllers.fixed import ControllerDecision


def _uncertainty_value(uncertainty: Any) -> float:
    if uncertainty is None:
        return 0.0
    if hasattr(uncertainty, "uncertainty"):
        return float(uncertainty.uncertainty)
    if isinstance(uncertainty, Mapping):
        return float(uncertainty.get("uncertainty", 0.0))
    return float(uncertainty)


@dataclass(frozen=True)
class UncertaintyRuleDenoiseAdaptor:
    """Map normalized uncertainty to one of several denoising budgets."""

    candidate_steps: Sequence[int] = (4, 8, 12, 20)
    uncertainty_thresholds: Sequence[float] = (0.5, 1.0, 1.5)

    def __post_init__(self) -> None:
        candidates = tuple(int(step) for step in self.candidate_steps)
        thresholds = tuple(float(threshold) for threshold in self.uncertainty_thresholds)
        if len(candidates) < 2:
            raise ValueError("candidate_steps must contain at least two values")
        if any(step <= 0 for step in candidates):
            raise ValueError("candidate_steps must be positive")
        if tuple(sorted(candidates)) != candidates:
            raise ValueError("candidate_steps must be sorted ascending")
        if len(thresholds) != len(candidates) - 1:
            raise ValueError("uncertainty_thresholds must have len(candidate_steps) - 1 values")
        if tuple(sorted(thresholds)) != thresholds:
            raise ValueError("uncertainty_thresholds must be sorted ascending")
        object.__setattr__(self, "candidate_steps", candidates)
        object.__setattr__(self, "uncertainty_thresholds", thresholds)

    @property
    def max_denoise_steps(self) -> int:
        return int(self.candidate_steps[-1])

    def select_denoise_steps(self, uncertainty: Any) -> int:
        """Return the first budget whose threshold exceeds the uncertainty."""

        value = _uncertainty_value(uncertainty)
        for index, threshold in enumerate(self.uncertainty_thresholds):
            if value < threshold:
                return int(self.candidate_steps[index])
        return int(self.candidate_steps[-1])


class DenoiseOnlyController:
    """Fixed-interval replanning controller with adaptive denoising on replans."""

    name = "denoise_only"

    def __init__(self, adaptor: UncertaintyRuleDenoiseAdaptor | None = None) -> None:
        self.adaptor = adaptor or UncertaintyRuleDenoiseAdaptor()

    def decide(
        self,
        buffer: ActionBuffer,
        obs: Any = None,
        uncertainty: Any = None,
    ) -> ControllerDecision:
        if buffer.is_empty:
            return ControllerDecision(
                action=1,
                forced_replan=True,
                denoise_steps=self.adaptor.select_denoise_steps(uncertainty),
            )
        return ControllerDecision(action=0, forced_replan=False)
