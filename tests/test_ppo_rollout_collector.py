from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adaptive_diffusion.controllers.ppo_replan import PPOReplanController
from adaptive_diffusion.executor import AdaptiveDiffusionExecutor
from adaptive_diffusion.rl.actor_critic import ReplanActorCritic
from adaptive_diffusion.rl.collector import (
    PPOExecutorRolloutCollector,
    record_ppo_transition,
)
from adaptive_diffusion.rl.features import ReplanFeatureConfig
from adaptive_diffusion.rl.rollout_buffer import ReplanRolloutBuffer


class _Space:
    def __init__(self, shape):
        self.shape = shape


class _FakePolicy:
    action_dim = 2
    chunk_horizon = 4
    default_denoise_steps = 20

    def __init__(self):
        self.sample_calls = 0

    def reset(self):
        pass

    def sample_action_chunk(
        self,
        obs,
        denoise_steps=None,
        deterministic=False,
        return_info=True,
    ):
        self.sample_calls += 1
        chunk = np.tile(np.array([[0.25, -0.25]], dtype=np.float32), (4, 1))
        return chunk, {
            "denoise_steps": 20,
            "nfe": 20,
            "wall_time_sec": 0.01,
            "sampler": "fake",
        }


class _FakeVectorEnv:
    single_action_space = _Space((1, 2))

    def __init__(self):
        self.step_count = 0
        self.actions = []

    def reset(self, seed=None):
        self.step_count = 0
        self.actions = []
        return {"state": np.zeros((1, 1, 3), dtype=np.float32)}

    def step(self, action):
        assert action.shape == (1, 1, 2)
        self.actions.append(action.copy())
        self.step_count += 1
        obs = {"state": np.full((1, 1, 3), self.step_count, dtype=np.float32)}
        reward = np.array([1.0 if self.step_count == 5 else 0.0])
        terminated = np.array([self.step_count >= 5])
        truncated = np.array([False])
        info = {"is_success": np.array([{"task": self.step_count >= 5}], dtype=object)}
        return obs, reward, terminated, truncated, [info]


def _zero_model(input_dim):
    model = ReplanActorCritic(input_dim=input_dim, hidden_dims=())
    for param in model.parameters():
        param.data.zero_()
    return model


def test_record_ppo_transition_stores_controller_decision_and_step_row():
    rollout = ReplanRolloutBuffer(capacity=1, feature_dim=3)
    decision = SimpleNamespace(
        obs_features=np.array([1.0, 0.0, 0.5], dtype=np.float32),
        action_mask=np.array([False, True]),
        action=1,
        logprob=-0.25,
        value=0.5,
    )
    row = {
        "controller_action": 1,
        "replanned": True,
        "hl_reward": 0.75,
        "env_reward": 1.0,
        "done": False,
        "denoise_steps": 20,
        "nfe": 20,
        "discarded_actions": 2,
    }

    record_ppo_transition(rollout, decision, row)

    batch = rollout.as_batch()
    assert rollout.size == 1
    np.testing.assert_allclose(batch.obs_features[0], [1.0, 0.0, 0.5])
    np.testing.assert_array_equal(batch.action_masks[0], [False, True])
    assert batch.actions.tolist() == [1]
    assert batch.logprobs.tolist() == pytest.approx([-0.25])
    assert batch.values.tolist() == pytest.approx([0.5])
    assert batch.rewards_hl.tolist() == pytest.approx([0.75])
    assert batch.rewards_env.tolist() == pytest.approx([1.0])
    assert batch.replanned.tolist() == [True]
    assert batch.nfe.tolist() == [20]
    assert batch.discarded_actions.tolist() == [2]


def test_record_ppo_transition_rejects_mismatched_action():
    rollout = ReplanRolloutBuffer(capacity=1, feature_dim=3)
    decision = SimpleNamespace(
        obs_features=np.zeros(3, dtype=np.float32),
        action_mask=np.array([True, True]),
        action=0,
        logprob=0.0,
        value=0.0,
    )
    row = {
        "controller_action": 1,
        "replanned": True,
        "hl_reward": 0.0,
        "env_reward": 0.0,
        "done": False,
        "denoise_steps": 20,
        "nfe": 20,
        "discarded_actions": 1,
    }

    with pytest.raises(ValueError, match="does not match"):
        record_ppo_transition(rollout, decision, row)


def test_ppo_executor_rollout_collector_records_episode_steps():
    feature_config = ReplanFeatureConfig(obs_dim=3, action_dim=2, horizon=4)
    model = _zero_model(feature_config.input_dim)
    model.actor[-1].bias.data = torch.tensor([5.0, -5.0])
    controller = PPOReplanController(
        model=model,
        feature_config=feature_config,
        deterministic=True,
    )
    policy = _FakePolicy()
    executor = AdaptiveDiffusionExecutor(
        env=_FakeVectorEnv(),
        diffusion_policy=policy,
        controller=controller,
        config={
            "method": "ppo_replan_only",
            "env_name": "fake",
            "seed": 0,
            "max_episode_steps": 10,
            "success_reward_threshold": 1.0,
        },
    )
    rollout = ReplanRolloutBuffer(capacity=5, feature_dim=feature_config.input_dim)
    collector = PPOExecutorRolloutCollector(executor, rollout)

    rows = collector.run_episode(episode_id=7, deterministic=True, seed=123)

    assert len(rows) == 5
    assert rollout.size == 5
    assert policy.sample_calls == 2
    batch = rollout.as_batch()
    assert batch.actions.tolist() == [1, 0, 0, 0, 1]
    assert batch.replanned.tolist() == [True, False, False, False, True]
    assert batch.nfe.tolist() == [20, 0, 0, 0, 20]
    assert batch.denoise_steps.tolist() == [20, 0, 0, 0, 20]
    assert batch.discarded_actions.tolist() == [0, 0, 0, 0, 0]
    np.testing.assert_array_equal(batch.action_masks[0], [False, True])
    np.testing.assert_array_equal(batch.action_masks[1], [True, True])
    np.testing.assert_array_equal(batch.action_masks[4], [False, True])
    assert int(batch.nfe.sum()) == sum(row["nfe"] for row in rows)

    rollout.compute_returns_and_advantages(next_value=0.0, gamma=1.0, gae_lambda=1.0)
    assert rollout.as_batch().returns.shape == (5,)
