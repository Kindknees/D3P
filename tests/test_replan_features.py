import numpy as np
import pytest

from adaptive_diffusion.buffers import ActionBuffer
from adaptive_diffusion.rl.features import (
    ReplanFeatureConfig,
    build_replan_features,
    replan_action_mask,
)
from adaptive_diffusion.uncertainty import UncertaintyEstimate


def test_replan_features_include_obs_buffer_scalars_and_uncertainty():
    config = ReplanFeatureConfig(obs_dim=3, action_dim=2, horizon=4)
    buffer = ActionBuffer(actions=np.ones((2, 2), dtype=np.float32), horizon=4)
    buffer.age = 1

    features = build_replan_features(
        obs={"state": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
        buffer=buffer,
        config=config,
        uncertainty=UncertaintyEstimate(uncertainty=1.25),
    )

    assert features.shape == (config.input_dim,)
    np.testing.assert_array_equal(features[:3], [1.0, 2.0, 3.0])
    assert features[-3] == 0.5
    assert features[-2] == 0.25
    assert features[-1] == 1.25


def test_replan_action_mask_forces_replan_only_when_empty():
    empty = ActionBuffer.empty(horizon=4, action_dim=2)
    non_empty = ActionBuffer(actions=np.ones((1, 2), dtype=np.float32), horizon=4)

    np.testing.assert_array_equal(replan_action_mask(empty), [False, True])
    np.testing.assert_array_equal(replan_action_mask(non_empty), [True, True])


def test_replan_features_validate_state_shape():
    config = ReplanFeatureConfig(obs_dim=3, action_dim=2, horizon=4)
    buffer = ActionBuffer.empty(horizon=4, action_dim=2)

    with pytest.raises(ValueError):
        build_replan_features({"state": np.ones(2, dtype=np.float32)}, buffer, config)
