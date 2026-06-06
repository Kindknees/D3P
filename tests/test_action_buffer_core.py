import numpy as np

from adaptive_diffusion.buffers import (
    ActionBuffer,
    first_action_jump,
    mean_overlap_jump,
    pad_residual_buffer,
)
from adaptive_diffusion.controllers.commitment import choose_td_commitment_horizon


def test_action_buffer_consumes_without_mutating_input():
    actions = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    buffer = ActionBuffer(actions)
    actions[0, 0] = 99.0

    consumed = buffer.consume_one()

    assert np.allclose(consumed, [1.0, 2.0])
    assert buffer.remaining_length == 1
    assert np.allclose(buffer.first(), [3.0, 4.0])


def test_pad_residual_buffer_repeats_last_action():
    residual = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    padded = pad_residual_buffer(residual, target_len=4)

    assert padded.shape == (4, 2)
    assert np.allclose(padded[-2:], [[3.0, 4.0], [3.0, 4.0]])


def test_jump_metrics_use_overlap():
    old = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    new = np.array([[3.0, 4.0], [1.0, 3.0]], dtype=np.float32)

    assert first_action_jump(old, new) == 5.0
    assert mean_overlap_jump(old, new) == 3.5


def test_choose_td_commitment_horizon():
    assert choose_td_commitment_horizon(10, high_threshold=9, medium_threshold=5) == 1
    assert choose_td_commitment_horizon(7, high_threshold=9, medium_threshold=5) == 2
    assert choose_td_commitment_horizon(1, high_threshold=9, medium_threshold=5) == 4
