import numpy as np
import pytest

from adaptive_diffusion.buffers import ActionBuffer


def test_empty_buffer_reports_zero_remaining():
    buffer = ActionBuffer.empty(horizon=4, action_dim=7)

    assert buffer.remaining == 0
    assert buffer.is_empty


def test_pop_first_removes_one_action_and_increments_age():
    actions = np.arange(12, dtype=np.float32).reshape(3, 4)
    buffer = ActionBuffer(actions=actions, horizon=4)

    action = buffer.pop_first()

    np.testing.assert_array_equal(action, actions[0])
    assert buffer.remaining == 2
    assert buffer.age == 1
    np.testing.assert_array_equal(buffer.actions[0], actions[1])


def test_pop_first_raises_on_empty_buffer():
    buffer = ActionBuffer.empty(horizon=4, action_dim=7)

    with pytest.raises(IndexError):
        buffer.pop_first()


def test_replace_resets_age_increments_plan_id_and_updates_metadata():
    buffer = ActionBuffer.empty(horizon=4, action_dim=2)
    buffer.age = 3

    buffer.replace(
        np.ones((4, 2), dtype=np.float32),
        denoise_steps=20,
        source="fixed_chunk",
    )

    assert buffer.age == 0
    assert buffer.plan_id == 1
    assert buffer.denoise_steps == 20
    assert buffer.source == "fixed_chunk"
    assert buffer.remaining == 4


def test_summary_shape_is_fixed_for_empty_partial_and_full_buffers():
    empty = ActionBuffer.empty(horizon=4, action_dim=3)
    partial = ActionBuffer(actions=np.ones((2, 3), dtype=np.float32), horizon=4)
    full = ActionBuffer(actions=np.ones((4, 3), dtype=np.float32), horizon=4)

    empty_summary = empty.summary(action_dim=3, max_horizon=4)
    partial_summary = partial.summary(action_dim=3, max_horizon=4)
    full_summary = full.summary(action_dim=3, max_horizon=4)

    assert empty_summary.shape == partial_summary.shape == full_summary.shape
    assert empty_summary.shape == (4 * 3 + 4,)
    np.testing.assert_array_equal(empty_summary[-4:], np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(partial_summary[-4:], [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(full_summary[-4:], np.ones(4, dtype=np.float32))
