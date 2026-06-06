"""Adapters between repository diffusion models and adaptive executor APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from adaptive_diffusion.repair import (
    RepairConfig,
    RepairResult,
    denoise_from,
    finalize_repair_candidate,
    forward_noise_with_model,
    prepare_repair_initialization,
)


@dataclass
class DiffusionPolicyAdapter:
    """Expose a diffusion model as `sample_action_chunk(obs)`."""

    model: Any
    device: str
    deterministic: bool = True

    @property
    def nfe_per_sample(self) -> int:
        if getattr(self.model, "use_ddim", False):
            return len(self.model.ddim_t)
        return int(self.model.denoising_steps)

    def sample_action_chunk(self, obs) -> np.ndarray:
        import torch

        cond = self.make_cond(obs)
        with torch.no_grad():
            sample = self.model(cond=cond, deterministic=self.deterministic)
        return sample.trajectories[0].detach().cpu().numpy()

    def sample_action_chunk_nfe(self, obs, nfe: int) -> np.ndarray:
        if nfe <= 0:
            raise ValueError("nfe must be positive.")
        if nfe >= self.nfe_per_sample:
            return self.sample_action_chunk(obs)
        if getattr(self.model, "use_ddim", False):
            return self._sample_action_chunk_ddim_subset(obs, nfe)
        return self._sample_action_chunk_ddpm_subset(obs, nfe)

    def _sample_action_chunk_ddpm_subset(self, obs, nfe: int) -> np.ndarray:
        import torch
        from model.diffusion.diffusion import Sample
        from model.diffusion.sampling import make_timesteps

        cond = self.make_cond(obs)
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        batch_size = len(sample_data)
        device = self.model.betas.device
        if nfe == 1:
            timesteps = [0]
        else:
            timesteps = np.linspace(
                self.model.denoising_steps - 1,
                0,
                num=nfe,
            ).round().astype(int).tolist()
            timesteps = sorted(set(timesteps), reverse=True)
            if timesteps[-1] != 0:
                timesteps.append(0)

        with torch.no_grad():
            x = torch.randn(
                (batch_size, self.model.horizon_steps, self.model.action_dim),
                device=device,
            )
            for index, t in enumerate(timesteps):
                t_b = make_timesteps(batch_size, int(t), device)
                index_b = make_timesteps(batch_size, index, device)
                mean, logvar = self.model.p_mean_var(
                    x=x,
                    t=t_b,
                    cond=cond,
                    index=index_b,
                    deterministic=self.deterministic,
                )
                std = torch.exp(0.5 * logvar)
                if t == 0:
                    std = torch.zeros_like(std)
                else:
                    std = torch.clip(std, min=1e-3)
                noise = torch.randn_like(x).clamp_(
                    -self.model.randn_clip_value,
                    self.model.randn_clip_value,
                )
                x = mean + std * noise
                if (
                    self.model.final_action_clip_value is not None
                    and index == len(timesteps) - 1
                ):
                    x = torch.clamp(
                        x,
                        -self.model.final_action_clip_value,
                        self.model.final_action_clip_value,
                    )
            sample = Sample(x, None)
        return sample.trajectories[0].detach().cpu().numpy()

    def _sample_action_chunk_ddim_subset(self, obs, nfe: int) -> np.ndarray:
        old_ddim_t = self.model.ddim_t
        try:
            self.model.ddim_t = old_ddim_t[:nfe]
            return self.sample_action_chunk(obs)
        finally:
            self.model.ddim_t = old_ddim_t

    def sample_action_chunks(self, obs, num_samples: int) -> np.ndarray:
        """Sample multiple chunks for the same observation."""

        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        return np.stack(
            [self.sample_action_chunk(obs) for _ in range(num_samples)],
            axis=0,
        )

    def action_disagreement(
        self,
        obs,
        num_samples: int = 2,
        preserve_rng: bool = True,
    ) -> float:
        """Estimate policy uncertainty from stochastic chunk disagreement."""

        if preserve_rng:
            import torch

            device = torch.device(self.device)
            devices = []
            if device.type == "cuda":
                devices = [device.index if device.index is not None else torch.cuda.current_device()]
            with torch.random.fork_rng(devices=devices):
                samples = self.sample_action_chunks(obs, num_samples)
        else:
            samples = self.sample_action_chunks(obs, num_samples)
        first_actions = samples[:, 0, :]
        center = first_actions.mean(axis=0, keepdims=True)
        return float(np.linalg.norm(first_actions - center, axis=1).mean())

    def make_cond(self, obs) -> dict[str, Any]:
        import torch

        if isinstance(obs, dict):
            if "state" not in obs:
                raise ValueError("DiffusionPolicyAdapter currently expects state obs.")
            state = np.asarray(obs["state"])
        else:
            state = np.asarray(obs)
        if state.ndim == 1:
            state = state[None, None, :]
        elif state.ndim == 2:
            state = state[None, :, :]
        elif state.ndim != 3:
            raise ValueError(f"Unsupported state observation shape {state.shape}.")
        return {"state": torch.from_numpy(state).float().to(self.device)}


@dataclass
class DiffusionResidualRepairer:
    """Repair residual action buffers using the current diffusion model."""

    policy: DiffusionPolicyAdapter

    def __call__(
        self,
        obs,
        residual_actions: np.ndarray,
        config: RepairConfig,
    ) -> RepairResult:
        import torch

        x0 = prepare_repair_initialization(residual_actions, config)
        x0_t = torch.from_numpy(x0[None]).float().to(self.policy.device)
        cond = self.policy.make_cond(obs)
        device = torch.device(self.policy.device)
        devices = []
        if device.type == "cuda":
            devices = [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices):
            with torch.no_grad():
                x_noised, k_repair = forward_noise_with_model(
                    self.policy.model,
                    x0_t,
                    config.noise_ratio,
                )
                repaired = denoise_from(
                    self.policy.model,
                    x_noised,
                    cond=cond,
                    start_step=k_repair,
                )
        repaired_np = repaired[0].detach().cpu().numpy()
        return finalize_repair_candidate(residual_actions, repaired_np, config)
