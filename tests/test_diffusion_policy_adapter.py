from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adaptive_diffusion.diffusion_interface import TorchDiffusionPolicyAdapter


class _FakeDiffusionModel(torch.nn.Module):
    use_ddim = False

    def __init__(self):
        super().__init__()
        self.denoising_steps = 20
        self.calls = []

    def forward(self, cond, deterministic=True):
        self.calls.append(int(self.denoising_steps))
        batch = cond["state"].shape[0]
        trajectories = torch.full((batch, 4, 2), float(self.denoising_steps))
        return SimpleNamespace(trajectories=trajectories)


def test_adapter_allows_lower_ddpm_denoise_steps_and_restores_model():
    model = _FakeDiffusionModel()
    adapter = TorchDiffusionPolicyAdapter(
        model=model,
        action_dim=2,
        chunk_horizon=4,
        default_denoise_steps=20,
        device="cpu",
    )

    chunk, info = adapter.sample_action_chunk(
        {"state": np.zeros((1, 3), dtype=np.float32)},
        denoise_steps=4,
        deterministic=True,
    )

    assert chunk.shape == (4, 2)
    assert np.all(chunk == 4.0)
    assert info["denoise_steps"] == 4
    assert info["nfe"] == 4
    assert model.calls == [4]
    assert model.denoising_steps == 20


def test_adapter_rejects_invalid_or_higher_denoise_steps():
    adapter = TorchDiffusionPolicyAdapter(
        model=_FakeDiffusionModel(),
        action_dim=2,
        chunk_horizon=4,
        default_denoise_steps=20,
        device="cpu",
    )

    with pytest.raises(ValueError):
        adapter.sample_action_chunk(
            {"state": np.zeros((1, 3), dtype=np.float32)},
            denoise_steps=0,
        )
    with pytest.raises(ValueError):
        adapter.sample_action_chunk(
            {"state": np.zeros((1, 3), dtype=np.float32)},
            denoise_steps=21,
        )
