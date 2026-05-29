import numpy as np
import pytest

from adaptive_diffusion.buffers import ActionBuffer
from adaptive_diffusion.controllers.td_error import TDErrorHeuristicController
from adaptive_diffusion.uncertainty import UncertaintyEstimate


def test_td_error_controller_forces_replan_on_empty_buffer():
    controller = TDErrorHeuristicController(threshold=1.0)
    buffer = ActionBuffer.empty(horizon=4, action_dim=2)

    decision = controller.decide(buffer, uncertainty=UncertaintyEstimate(uncertainty=0.0))

    assert decision.action == 1
    assert decision.forced_replan


def test_td_error_controller_replans_only_above_z_threshold():
    controller = TDErrorHeuristicController(threshold=1.0)
    buffer = ActionBuffer(actions=np.ones((2, 2), dtype=np.float32), horizon=4)

    low = controller.decide(buffer, uncertainty=UncertaintyEstimate(uncertainty=0.5))
    high = controller.decide(buffer, uncertainty=UncertaintyEstimate(uncertainty=1.5))

    assert low.action == 0
    assert not low.forced_replan
    assert high.action == 1
    assert not high.forced_replan


def test_td_error_controller_percentile_requires_calibration():
    controller = TDErrorHeuristicController(
        threshold_mode="percentile",
        threshold_percentile=75,
    )
    buffer = ActionBuffer(actions=np.ones((2, 2), dtype=np.float32), horizon=4)

    with pytest.raises(RuntimeError):
        controller.decide(buffer, uncertainty=UncertaintyEstimate(uncertainty=1.0))

    threshold = controller.calibrate([0.0, 1.0, 2.0, 3.0])
    decision = controller.decide(buffer, uncertainty=UncertaintyEstimate(uncertainty=2.5))

    assert threshold == pytest.approx(2.25)
    assert decision.action == 1
