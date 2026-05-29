import pytest

from adaptive_diffusion.buffers import ActionBuffer
from adaptive_diffusion.controllers.denoise_adaptor import (
    DenoiseOnlyController,
    UncertaintyRuleDenoiseAdaptor,
)
from adaptive_diffusion.uncertainty import UncertaintyEstimate


def test_uncertainty_rule_selects_candidate_steps():
    adaptor = UncertaintyRuleDenoiseAdaptor(
        candidate_steps=(4, 8, 12, 20),
        uncertainty_thresholds=(0.5, 1.0, 1.5),
    )

    assert adaptor.select_denoise_steps(UncertaintyEstimate(uncertainty=0.1)) == 4
    assert adaptor.select_denoise_steps({"uncertainty": 0.75}) == 8
    assert adaptor.select_denoise_steps(1.25) == 12
    assert adaptor.select_denoise_steps(2.0) == 20
    assert adaptor.max_denoise_steps == 20


def test_uncertainty_rule_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        UncertaintyRuleDenoiseAdaptor(candidate_steps=(4,), uncertainty_thresholds=())
    with pytest.raises(ValueError):
        UncertaintyRuleDenoiseAdaptor(
            candidate_steps=(8, 4),
            uncertainty_thresholds=(1.0,),
        )
    with pytest.raises(ValueError):
        UncertaintyRuleDenoiseAdaptor(
            candidate_steps=(4, 8, 12),
            uncertainty_thresholds=(1.0,),
        )
    with pytest.raises(ValueError):
        UncertaintyRuleDenoiseAdaptor(
            candidate_steps=(4, 8, 12),
            uncertainty_thresholds=(1.0, 0.5),
        )


def test_denoise_only_controller_replans_only_on_empty_buffer():
    controller = DenoiseOnlyController(
        UncertaintyRuleDenoiseAdaptor(
            candidate_steps=(4, 20),
            uncertainty_thresholds=(1.0,),
        )
    )
    empty = ActionBuffer.empty(horizon=4, action_dim=2)
    full = ActionBuffer.empty(horizon=4, action_dim=2)
    full.replace([[0.0, 0.0], [1.0, 1.0]], denoise_steps=20, source="test")

    decision = controller.decide(empty, uncertainty=UncertaintyEstimate(uncertainty=2.0))
    assert decision.action == 1
    assert decision.forced_replan is True
    assert decision.denoise_steps == 20

    decision = controller.decide(full, uncertainty=UncertaintyEstimate(uncertainty=2.0))
    assert decision.action == 0
    assert decision.forced_replan is False
    assert decision.denoise_steps is None
