"""Residual-buffer repair helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from adaptive_diffusion.buffers import (
    first_action_jump,
    mean_overlap_jump,
    pad_residual_buffer,
)


@dataclass(frozen=True)
class RepairConfig:
    target_len: int
    noise_ratio: float = 0.5
    anchor_rho: float = 0.5
    accept_min_action_jump: float | None = None
    accept_max_action_jump: float | None = None
    accept_min_buffer_jump: float | None = None
    accept_max_buffer_jump: float | None = None
    clip_min: float | None = -1.0
    clip_max: float | None = 1.0


@dataclass(frozen=True)
class RepairResult:
    repaired_actions: np.ndarray
    action_jump: float
    buffer_jump: float
    accepted: bool
    fallback_reason: str


def anchor_first_action(
    repaired_actions: np.ndarray,
    residual_actions: np.ndarray,
    anchor_rho: float,
) -> np.ndarray:
    """Blend the first repaired action toward the residual first action."""

    if not 0 <= anchor_rho <= 1:
        raise ValueError("anchor_rho must be in [0, 1].")
    repaired = np.asarray(repaired_actions).copy()
    residual = np.asarray(residual_actions)
    if len(repaired) == 0 or len(residual) == 0:
        raise ValueError("Cannot anchor an empty action buffer.")
    repaired[0] = (1 - anchor_rho) * residual[0] + anchor_rho * repaired[0]
    return repaired


def finalize_repair_candidate(
    residual_actions: np.ndarray,
    repaired_actions: np.ndarray,
    config: RepairConfig,
) -> RepairResult:
    """Apply anchoring, clipping, jump metrics, and acceptance checks."""

    residual = np.asarray(residual_actions)
    repaired = anchor_first_action(
        repaired_actions=repaired_actions,
        residual_actions=residual,
        anchor_rho=config.anchor_rho,
    )
    if config.clip_min is not None or config.clip_max is not None:
        repaired = np.clip(repaired, config.clip_min, config.clip_max)

    action_jump = first_action_jump(residual, repaired)
    buffer_jump = mean_overlap_jump(residual, repaired)
    accepted = True
    fallback_reason = ""
    if (
        config.accept_min_action_jump is not None
        and action_jump < config.accept_min_action_jump
    ):
        accepted = False
        fallback_reason = "action_jump_too_small"
    if (
        accepted
        and config.accept_max_action_jump is not None
        and action_jump > config.accept_max_action_jump
    ):
        accepted = False
        fallback_reason = "action_jump"
    if (
        accepted
        and config.accept_min_buffer_jump is not None
        and buffer_jump < config.accept_min_buffer_jump
    ):
        accepted = False
        fallback_reason = "buffer_jump_too_small"
    if (
        accepted
        and config.accept_max_buffer_jump is not None
        and buffer_jump > config.accept_max_buffer_jump
    ):
        accepted = False
        fallback_reason = "buffer_jump"

    return RepairResult(
        repaired_actions=repaired,
        action_jump=action_jump,
        buffer_jump=buffer_jump,
        accepted=accepted,
        fallback_reason=fallback_reason,
    )


def prepare_repair_initialization(
    residual_actions: np.ndarray,
    config: RepairConfig,
) -> np.ndarray:
    """Construct the padded x0 repair initialization."""

    return pad_residual_buffer(residual_actions, target_len=config.target_len)


def repair_noise_step(denoising_steps: int, noise_ratio: float) -> int:
    """Map a repair noise ratio to a valid diffusion step index."""

    if denoising_steps <= 0:
        raise ValueError("denoising_steps must be positive.")
    if not 0 < noise_ratio <= 1:
        raise ValueError("noise_ratio must be in (0, 1].")
    return max(1, min(denoising_steps - 1, int(round(noise_ratio * denoising_steps))))


def forward_noise_with_model(model: Any, x_start: Any, noise_ratio: float):
    """Use a diffusion model's q_sample to forward-noise a repair initialization."""

    import torch

    k_repair = repair_noise_step(model.denoising_steps, noise_ratio)
    batch_size = x_start.shape[0]
    t = torch.full((batch_size,), k_repair, device=x_start.device, dtype=torch.long)
    return model.q_sample(x_start=x_start, t=t), k_repair


def denoise_from(model: Any, x_k: Any, cond: dict[str, Any], start_step: int):
    """Denoise an action sample from an intermediate DDPM step to x0."""

    import torch
    from model.diffusion.sampling import make_timesteps

    if start_step <= 0:
        raise ValueError("start_step must be positive.")
    if start_step >= model.denoising_steps:
        raise ValueError("start_step must be smaller than model.denoising_steps.")

    x = x_k
    batch_size = len(x_k)
    device = x_k.device
    for t in reversed(range(start_step + 1)):
        t_b = make_timesteps(batch_size, t, device)
        index_b = make_timesteps(batch_size, model.denoising_steps - 1 - t, device)
        mean, logvar = model.p_mean_var(
            x=x,
            t=t_b,
            cond=cond,
            index=index_b,
            deterministic=True,
        )
        std = torch.exp(0.5 * logvar)
        if t == 0:
            std = torch.zeros_like(std)
        else:
            std = torch.clip(std, min=1e-3)
        noise = torch.randn_like(x).clamp_(-model.randn_clip_value, model.randn_clip_value)
        x = mean + std * noise

    if model.final_action_clip_value is not None:
        x = torch.clamp(x, -model.final_action_clip_value, model.final_action_clip_value)
    return x
