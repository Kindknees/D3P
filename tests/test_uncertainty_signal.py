import numpy as np
import pytest

from adaptive_diffusion.uncertainty import NullUncertaintySignal, TDErrorSignal


def _state_value(obs):
    return np.asarray(obs["state"], dtype=np.float32).reshape(-1)[0]


def test_null_uncertainty_signal_returns_zero_estimates():
    signal = NullUncertaintySignal()

    estimate = signal.before_action({}, None)
    update = signal.after_transition({}, np.zeros(1), 1.0, {}, False, {})

    assert signal.value == 0.0
    assert estimate.uncertainty == 0.0
    assert estimate.td_error == 0.0
    assert update.uncertainty == 0.0
    assert update.td_error == 0.0


def test_td_error_signal_updates_running_statistics_and_previous_value():
    signal = TDErrorSignal(value_fn=_state_value, gamma=1.0)
    obs0 = {"state": np.array([0.0], dtype=np.float32)}
    obs1 = {"state": np.array([2.0], dtype=np.float32)}
    obs2 = {"state": np.array([3.0], dtype=np.float32)}

    first = signal.after_transition(obs0, np.zeros(1), 1.0, obs1, False, {})

    assert first.td_error == 3.0
    assert first.raw_abs_td_error == 3.0
    assert first.uncertainty == 0.0
    assert first.count == 1
    assert signal.before_action(obs1, None) == first

    second = signal.after_transition(obs1, np.zeros(1), 0.0, obs2, False, {})

    assert second.td_error == 1.0
    assert second.raw_abs_td_error == 1.0
    assert second.running_mean == 2.0
    assert second.running_std == pytest.approx(np.sqrt(2.0))
    assert second.uncertainty == pytest.approx(-1.0 / np.sqrt(2.0))


def test_td_error_signal_resets_state():
    signal = TDErrorSignal(value_fn=_state_value, gamma=0.5)
    obs = {"state": np.array([1.0], dtype=np.float32)}

    signal.after_transition(obs, np.zeros(1), 1.0, obs, False, {})
    signal.reset()

    assert signal.value == 0.0
    assert signal.before_action(obs, None).count == 0
