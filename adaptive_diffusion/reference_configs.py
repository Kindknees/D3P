"""Reference baseline configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


FIXED_CONTROLLER_METHODS = {"fixed_chunk", "high_compute", "low_compute"}
ADAPTIVE_METHODS = {"td_error_replan", "denoise_only"}
SUPPORTED_METHODS = FIXED_CONTROLLER_METHODS | ADAPTIVE_METHODS


@dataclass(frozen=True)
class BaselineReferenceConfig:
    method: str
    chunk_horizon: int
    denoise_steps: int
    source: str


def resolve_baseline_reference_config(
    method: str,
    chunk_horizon: Optional[int] = None,
    denoise_steps: Optional[int] = None,
) -> BaselineReferenceConfig:
    """Resolve method-specific defaults for baseline/reference runs."""

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported method: {method}")

    if method == "high_compute":
        default_horizon = 1
        default_denoise_steps = 20
    elif method == "low_compute":
        default_horizon = 4
        default_denoise_steps = 4
    else:
        default_horizon = 4
        default_denoise_steps = 20

    resolved_horizon = default_horizon if chunk_horizon is None else int(chunk_horizon)
    resolved_denoise_steps = (
        default_denoise_steps if denoise_steps is None else int(denoise_steps)
    )
    if resolved_horizon <= 0:
        raise ValueError("chunk_horizon must be positive")
    if resolved_denoise_steps <= 0:
        raise ValueError("denoise_steps must be positive")

    if method == "td_error_replan":
        source = "td_error"
    elif method == "denoise_only":
        source = "denoise_adaptor"
    else:
        source = "fixed"

    return BaselineReferenceConfig(
        method=method,
        chunk_horizon=resolved_horizon,
        denoise_steps=resolved_denoise_steps,
        source=source,
    )
