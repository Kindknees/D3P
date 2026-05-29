import json

import numpy as np
import pytest

from adaptive_diffusion.td_critic import (
    TDCriticValueFunction,
    calibrate_td_thresholds,
    compute_td_error_statistics,
    flatten_state_obs,
    load_transition_dataset,
    running_z_scores,
    train_td_critic,
)


def _write_dataset(path):
    obs = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    next_obs = obs + 0.5
    rewards = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    dones = np.array([False, False, False, True])
    actions = np.zeros((4, 1), dtype=np.float32)
    np.savez_compressed(
        path,
        obs=obs,
        next_obs=next_obs,
        rewards=rewards,
        dones=dones,
        actions=actions,
    )


def test_load_transition_dataset_validates_shapes(tmp_path):
    dataset_path = tmp_path / "transitions.npz"
    _write_dataset(dataset_path)

    dataset = load_transition_dataset(dataset_path)

    assert dataset.size == 4
    assert dataset.obs_dim == 2
    assert dataset.rewards.dtype == np.float32


def test_train_td_critic_writes_checkpoint_and_loadable_value_function(tmp_path):
    dataset_path = tmp_path / "transitions.npz"
    output_dir = tmp_path / "critic"
    _write_dataset(dataset_path)

    summary = train_td_critic(
        dataset_path=dataset_path,
        output_dir=output_dir,
        epochs=3,
        batch_size=2,
        learning_rate=1e-3,
        gamma=0.9,
        seed=123,
        hidden_dims=(8,),
    )

    checkpoint_path = output_dir / "td_critic.pt"
    summary_path = output_dir / "train_summary.json"
    assert checkpoint_path.exists()
    assert summary_path.exists()
    assert summary["num_transitions"] == 4
    assert summary["obs_dim"] == 2
    assert np.isfinite(summary["final_loss"])
    assert json.loads(summary_path.read_text())["checkpoint_path"] == str(checkpoint_path)

    value_fn = TDCriticValueFunction(checkpoint_path)
    value = value_fn({"state": np.array([0.25, 0.75], dtype=np.float32)})

    assert isinstance(value, float)
    assert np.isfinite(value)


def test_running_z_scores_match_online_normalization():
    values = np.array([1.0, 3.0, 5.0], dtype=np.float32)

    z_scores, means, stds = running_z_scores(values)

    np.testing.assert_allclose(means, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(stds, [0.0, np.sqrt(2.0), 2.0])
    np.testing.assert_allclose(z_scores, [0.0, 1.0 / np.sqrt(2.0), 1.0])


def test_calibrate_td_thresholds_writes_percentile_summary(tmp_path):
    dataset_path = tmp_path / "transitions.npz"
    critic_dir = tmp_path / "critic"
    calibration_path = tmp_path / "td_threshold_calibration.json"
    _write_dataset(dataset_path)
    train_td_critic(
        dataset_path=dataset_path,
        output_dir=critic_dir,
        epochs=2,
        batch_size=2,
        hidden_dims=(8,),
        seed=5,
    )

    summary = calibrate_td_thresholds(
        dataset_path=dataset_path,
        critic_checkpoint=critic_dir / "td_critic.pt",
        output_json=calibration_path,
        percentiles=(50, 90),
        z_thresholds=(0.5, 1.0),
    )

    assert calibration_path.exists()
    assert summary["num_transitions"] == 4
    assert summary["obs_dim"] == 2
    assert set(summary["percentile_thresholds"]) == {"50", "90"}
    assert set(summary["abs_td_error_percentiles"]) == {"50", "90"}
    assert summary["z_score_thresholds"] == [0.5, 1.0]
    assert np.isfinite(summary["mean_abs_td_error"])
    assert json.loads(calibration_path.read_text())["num_transitions"] == 4

    value_fn = TDCriticValueFunction(critic_dir / "td_critic.pt")
    stats = compute_td_error_statistics(load_transition_dataset(dataset_path), value_fn)
    assert stats["td_errors"].shape == (4,)
    assert stats["uncertainties"].shape == (4,)


def test_flatten_state_obs_requires_state_key():
    np.testing.assert_array_equal(
        flatten_state_obs({"state": np.ones((1, 2), dtype=np.float32)}),
        np.ones(2, dtype=np.float32),
    )
    with pytest.raises(KeyError):
        flatten_state_obs({"pixels": np.zeros((2, 2), dtype=np.float32)})
