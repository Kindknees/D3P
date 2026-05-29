import pytest

from adaptive_diffusion.reference_configs import resolve_baseline_reference_config


def test_fixed_td_error_and_denoise_only_default_to_standard_chunk_and_denoise():
    fixed = resolve_baseline_reference_config("fixed_chunk")
    td_error = resolve_baseline_reference_config("td_error_replan")
    denoise_only = resolve_baseline_reference_config("denoise_only")

    assert fixed.chunk_horizon == 4
    assert fixed.denoise_steps == 20
    assert fixed.source == "fixed"
    assert td_error.chunk_horizon == 4
    assert td_error.denoise_steps == 20
    assert td_error.source == "td_error"
    assert denoise_only.chunk_horizon == 4
    assert denoise_only.denoise_steps == 20
    assert denoise_only.source == "denoise_adaptor"


def test_high_compute_default_replans_every_step():
    config = resolve_baseline_reference_config("high_compute")

    assert config.chunk_horizon == 1
    assert config.denoise_steps == 20
    assert config.source == "fixed"


def test_low_compute_default_uses_low_denoise_steps():
    config = resolve_baseline_reference_config("low_compute")

    assert config.chunk_horizon == 4
    assert config.denoise_steps == 4
    assert config.source == "fixed"


def test_reference_config_allows_explicit_overrides():
    config = resolve_baseline_reference_config(
        "low_compute",
        chunk_horizon=2,
        denoise_steps=8,
    )

    assert config.chunk_horizon == 2
    assert config.denoise_steps == 8


def test_reference_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        resolve_baseline_reference_config("unknown")
    with pytest.raises(ValueError):
        resolve_baseline_reference_config("fixed_chunk", chunk_horizon=0)
    with pytest.raises(ValueError):
        resolve_baseline_reference_config("fixed_chunk", denoise_steps=0)
