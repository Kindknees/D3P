"""Interfaces and adapters for diffusion policy chunk samplers."""

from __future__ import annotations

import time
from typing import Any, Optional, Protocol

import numpy as np
import torch


class DiffusionPolicyInterface(Protocol):
    """Black-box interface used by the adaptive executor."""

    action_dim: int
    chunk_horizon: int
    default_denoise_steps: int

    def reset(self) -> None:
        """Reset any policy-local state."""

    def sample_action_chunk(
        self,
        obs: Any,
        denoise_steps: Optional[int] = None,
        deterministic: bool = False,
        return_info: bool = True,
    ) -> tuple[np.ndarray, dict]:
        """Sample an action chunk with shape [chunk_horizon, action_dim]."""


class TorchDiffusionPolicyAdapter:
    """Adapter for existing torch diffusion models in this repository."""

    def __init__(
        self,
        model: torch.nn.Module,
        action_dim: int,
        chunk_horizon: int,
        default_denoise_steps: int,
        device: str,
    ) -> None:
        self.model = model
        self.action_dim = int(action_dim)
        self.chunk_horizon = int(chunk_horizon)
        self.default_denoise_steps = int(default_denoise_steps)
        self.device = torch.device(device)
        self.model.eval()

    def reset(self) -> None:
        """The wrapped diffusion model is stateless for M1 execution."""

    def sample_action_chunk(
        self,
        obs: Any,
        denoise_steps: Optional[int] = None,
        deterministic: bool = False,
        return_info: bool = True,
    ) -> tuple[np.ndarray, dict]:
        """Sample one action chunk and report inference cost metadata."""

        steps = self.default_denoise_steps if denoise_steps is None else int(denoise_steps)
        if steps <= 0:
            raise ValueError("denoise_steps must be positive")
        if steps > self.default_denoise_steps:
            raise ValueError(
                f"denoise_steps={steps} cannot exceed configured "
                f"default_denoise_steps={self.default_denoise_steps}"
            )

        cond = self._to_torch_cond(obs)
        restore_denoising_steps = None
        if steps != self.default_denoise_steps:
            if getattr(self.model, "use_ddim", False):
                raise ValueError("variable denoise_steps are only supported for DDPM sampling")
            if not hasattr(self.model, "denoising_steps"):
                raise ValueError("wrapped model does not expose denoising_steps")
            restore_denoising_steps = int(self.model.denoising_steps)
            self.model.denoising_steps = steps
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        try:
            with torch.no_grad():
                samples = self.model(cond=cond, deterministic=deterministic)
        finally:
            if restore_denoising_steps is not None:
                self.model.denoising_steps = restore_denoising_steps
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start

        chunk = samples.trajectories[0, : self.chunk_horizon].detach().cpu().numpy()
        if chunk.shape != (self.chunk_horizon, self.action_dim):
            raise ValueError(
                f"sampled chunk shape {chunk.shape} does not match "
                f"({self.chunk_horizon}, {self.action_dim})"
            )
        info = {
            "denoise_steps": steps,
            "nfe": steps,
            "wall_time_sec": elapsed,
            "sampler": "ddim" if getattr(self.model, "use_ddim", False) else "ddpm",
        }
        return chunk.astype(np.float32), info if return_info else {}

    def _to_torch_cond(self, obs: Any) -> dict[str, torch.Tensor]:
        if not isinstance(obs, dict):
            raise TypeError("obs must be a dict with a 'state' entry")
        if "state" not in obs:
            raise KeyError("obs must contain key 'state'")

        state = np.asarray(obs["state"], dtype=np.float32)
        if state.ndim == 2:
            state = state[None]
        if state.ndim != 3:
            raise ValueError("state observation must have shape [To, Do] or [B, To, Do]")
        return {"state": torch.from_numpy(state).float().to(self.device)}
