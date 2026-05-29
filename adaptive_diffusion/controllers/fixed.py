"""Fixed-chunk replanning controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adaptive_diffusion.buffers import ActionBuffer


@dataclass(frozen=True)
class ControllerDecision:
    """Binary controller output for one environment step."""

    action: int
    forced_replan: bool = False
    denoise_steps: int | None = None


class FixedChunkController:
    """Replan only when the action buffer is empty."""

    name = "fixed_chunk"

    def decide(
        self,
        buffer: ActionBuffer,
        obs: Any = None,
        uncertainty: Any = None,
    ) -> ControllerDecision:
        """Return 1 to replan when empty, otherwise 0 to continue."""

        if buffer.is_empty:
            return ControllerDecision(action=1, forced_replan=True)
        return ControllerDecision(action=0, forced_replan=False)
