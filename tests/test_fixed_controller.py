import numpy as np

from adaptive_diffusion.buffers import ActionBuffer
from adaptive_diffusion.controllers.fixed import FixedChunkController


def test_fixed_controller_forces_replan_when_buffer_empty():
    controller = FixedChunkController()
    buffer = ActionBuffer.empty(horizon=4, action_dim=7)

    decision = controller.decide(buffer)

    assert decision.action == 1
    assert decision.forced_replan


def test_fixed_controller_continues_when_buffer_has_actions():
    controller = FixedChunkController()
    buffer = ActionBuffer(actions=np.ones((2, 7), dtype=np.float32), horizon=4)

    decision = controller.decide(buffer)

    assert decision.action == 0
    assert not decision.forced_replan
