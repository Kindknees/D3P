"""TD-error heuristic replanning controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from adaptive_diffusion.buffers import ActionBuffer
from adaptive_diffusion.controllers.fixed import ControllerDecision


@dataclass
class TDErrorHeuristicController:
    """Replan when normalized TD-error uncertainty exceeds a threshold."""

    threshold_mode: str = "z_score"
    threshold: float = 1.0
    threshold_percentile: float = 90.0
    calibrated_threshold: Optional[float] = None

    name = "td_error_replan"

    def __post_init__(self) -> None:
        if self.threshold_mode not in {"z_score", "percentile"}:
            raise ValueError("threshold_mode must be 'z_score' or 'percentile'")
        if not 0.0 <= float(self.threshold_percentile) <= 100.0:
            raise ValueError("threshold_percentile must be in [0, 100]")

    def calibrate(self, values: Iterable[float]) -> float:
        """Set the percentile threshold from calibration uncertainty values."""

        array = np.asarray(list(values), dtype=np.float32)
        if array.size == 0:
            raise ValueError("calibration values cannot be empty")
        self.calibrated_threshold = float(
            np.percentile(array, float(self.threshold_percentile))
        )
        return self.calibrated_threshold

    def decide(
        self,
        buffer: ActionBuffer,
        obs: Any = None,
        uncertainty: Any = None,
    ) -> ControllerDecision:
        if buffer.is_empty:
            return ControllerDecision(action=1, forced_replan=True)

        if self._uncertainty_value(uncertainty) >= self._active_threshold():
            return ControllerDecision(action=1, forced_replan=False)
        return ControllerDecision(action=0, forced_replan=False)

    def _active_threshold(self) -> float:
        if self.threshold_mode == "percentile":
            if self.calibrated_threshold is None:
                raise RuntimeError("percentile threshold requires calibrate() before decide")
            return float(self.calibrated_threshold)
        return float(self.threshold)

    def _uncertainty_value(self, uncertainty: Any) -> float:
        if uncertainty is None:
            return 0.0
        if hasattr(uncertainty, "uncertainty"):
            return float(uncertainty.uncertainty)
        if isinstance(uncertainty, Mapping):
            return float(uncertainty.get("uncertainty", 0.0))
        return float(uncertainty)
