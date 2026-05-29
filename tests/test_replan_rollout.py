import numpy as np
import pytest

from adaptive_diffusion.rl.advantage import compute_gae
from adaptive_diffusion.rl.rollout_buffer import ReplanRolloutBuffer


def test_compute_gae_handles_episode_boundaries():
    advantages, returns = compute_gae(
        rewards=np.array([1.0, 2.0], dtype=np.float32),
        values=np.array([0.0, 0.0], dtype=np.float32),
        dones=np.array([False, True]),
        next_value=10.0,
        gamma=1.0,
        gae_lambda=1.0,
    )

    np.testing.assert_allclose(advantages, [3.0, 2.0])
    np.testing.assert_allclose(returns, [3.0, 2.0])


def test_rollout_buffer_stores_rollout_and_computes_returns():
    buffer = ReplanRolloutBuffer(capacity=2, feature_dim=3)
    buffer.add(
        obs_features=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        action=0,
        logprob=-0.1,
        value=0.2,
        reward_hl=1.0,
        reward_env=1.0,
        done=False,
        action_mask=np.array([True, True]),
        replanned=False,
        denoise_steps=0,
        nfe=0,
        discarded_actions=0,
    )
    buffer.add(
        obs_features=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        action=1,
        logprob=-0.2,
        value=0.3,
        reward_hl=2.0,
        reward_env=2.0,
        done=True,
        action_mask=np.array([False, True]),
        replanned=True,
        denoise_steps=20,
        nfe=20,
        discarded_actions=1,
    )
    buffer.compute_returns_and_advantages(next_value=0.0, gamma=1.0, gae_lambda=1.0)

    batch = buffer.as_batch(normalize_advantages=True)

    assert buffer.size == 2
    assert batch.obs_features.shape == (2, 3)
    assert batch.action_masks.shape == (2, 2)
    assert batch.replanned.tolist() == [False, True]
    assert batch.nfe.tolist() == [0, 20]
    assert batch.advantages.shape == (2,)
    assert abs(float(batch.advantages.mean())) < 1e-6


def test_rollout_buffer_rejects_overflow_and_bad_shapes():
    buffer = ReplanRolloutBuffer(capacity=1, feature_dim=3)
    with pytest.raises(ValueError):
        buffer.add(
            obs_features=np.ones(2, dtype=np.float32),
            action=0,
            logprob=0.0,
            value=0.0,
            reward_hl=0.0,
            reward_env=0.0,
            done=False,
            action_mask=np.array([True, True]),
            replanned=False,
            denoise_steps=0,
            nfe=0,
            discarded_actions=0,
        )
    buffer.add(
        obs_features=np.ones(3, dtype=np.float32),
        action=0,
        logprob=0.0,
        value=0.0,
        reward_hl=0.0,
        reward_env=0.0,
        done=False,
        action_mask=np.array([True, True]),
        replanned=False,
        denoise_steps=0,
        nfe=0,
        discarded_actions=0,
    )
    with pytest.raises(IndexError):
        buffer.add(
            obs_features=np.ones(3, dtype=np.float32),
            action=0,
            logprob=0.0,
            value=0.0,
            reward_hl=0.0,
            reward_env=0.0,
            done=False,
            action_mask=np.array([True, True]),
            replanned=False,
            denoise_steps=0,
            nfe=0,
            discarded_actions=0,
        )
