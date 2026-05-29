from dataclasses import asdict

import numpy as np
import pytest
import torch

from adaptive_diffusion.rl.actor_critic import ReplanActorCritic
from adaptive_diffusion.rl.ppo import (
    PPOBCRegularizerBatch,
    PPOUpdateConfig,
    load_replan_actor_critic_checkpoint,
    ppo_config_to_dict,
    save_replan_actor_critic_checkpoint,
    update_replan_ppo,
)
from adaptive_diffusion.rl.rollout_buffer import RolloutBatch


def _rollout_batch() -> RolloutBatch:
    return RolloutBatch(
        obs_features=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        actions=np.array([1, 0, 1, 0, 0, 1], dtype=np.int64),
        logprobs=np.array(
            [0.0, np.log(0.5), np.log(0.5), np.log(0.5), np.log(0.5), np.log(0.5)],
            dtype=np.float32,
        ),
        values=np.zeros(6, dtype=np.float32),
        rewards_hl=np.array([1.0, 0.0, 0.5, -0.25, 0.75, -0.5], dtype=np.float32),
        rewards_env=np.array([1.0, 0.0, 0.5, -0.25, 0.75, -0.5], dtype=np.float32),
        dones=np.array([False, False, False, False, False, True]),
        action_masks=np.array(
            [
                [False, True],
                [True, True],
                [True, True],
                [True, True],
                [True, True],
                [True, True],
            ],
            dtype=bool,
        ),
        replanned=np.array([True, False, True, False, False, True]),
        denoise_steps=np.array([20, 0, 20, 0, 0, 20], dtype=np.int64),
        nfe=np.array([20, 0, 20, 0, 0, 20], dtype=np.int64),
        discarded_actions=np.zeros(6, dtype=np.int64),
        advantages=np.array([1.0, -0.5, 0.75, -0.25, 0.5, -0.75], dtype=np.float32),
        returns=np.array([1.0, 0.0, 0.5, -0.25, 0.75, -0.5], dtype=np.float32),
    )


def test_update_replan_ppo_changes_parameters_and_reports_stats():
    torch.manual_seed(0)
    model = ReplanActorCritic(input_dim=3, hidden_dims=(4,))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    params_before = [param.detach().clone() for param in model.parameters()]

    stats = update_replan_ppo(
        model,
        optimizer,
        _rollout_batch(),
        PPOUpdateConfig(epochs=2, minibatch_size=3, entropy_coef=0.01),
        seed=123,
    )

    assert stats.num_updates == 4
    for value in asdict(stats).values():
        assert np.isfinite(value)
    assert any(
        not torch.allclose(before, after)
        for before, after in zip(params_before, model.parameters())
    )
    assert not model.training


def test_update_replan_ppo_reports_bc_regularizer_stats():
    torch.manual_seed(0)
    model = ReplanActorCritic(input_dim=3, hidden_dims=(4,))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    teacher_batch = PPOBCRegularizerBatch(
        obs_features=np.array(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        action_masks=np.ones((3, 2), dtype=bool),
        actions=np.array([1, 0, 1], dtype=np.int64),
        sample_weights=np.array([2.0, 1.0, 3.0], dtype=np.float32),
    )

    stats = update_replan_ppo(
        model,
        optimizer,
        _rollout_batch(),
        PPOUpdateConfig(
            epochs=1,
            minibatch_size=3,
            bc_regularizer_coef=0.5,
            bc_regularizer_minibatch_size=2,
        ),
        seed=123,
        bc_regularizer_batch=teacher_batch,
    )

    assert stats.num_updates == 2
    assert stats.bc_regularizer_loss > 0.0
    assert 0.0 <= stats.bc_regularizer_accuracy <= 1.0


def test_update_replan_ppo_requires_bc_batch_when_regularizer_enabled():
    model = ReplanActorCritic(input_dim=3, hidden_dims=())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    with pytest.raises(ValueError, match="bc_regularizer_batch"):
        update_replan_ppo(
            model,
            optimizer,
            _rollout_batch(),
            PPOUpdateConfig(bc_regularizer_coef=0.1),
        )


def test_update_replan_ppo_rejects_masked_bc_teacher_action():
    model = ReplanActorCritic(input_dim=3, hidden_dims=())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    teacher_batch = PPOBCRegularizerBatch(
        obs_features=np.zeros((1, 3), dtype=np.float32),
        action_masks=np.array([[False, True]], dtype=bool),
        actions=np.array([0], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="allowed"):
        update_replan_ppo(
            model,
            optimizer,
            _rollout_batch(),
            PPOUpdateConfig(bc_regularizer_coef=0.1),
            bc_regularizer_batch=teacher_batch,
        )


def test_update_replan_ppo_rejects_empty_batch():
    model = ReplanActorCritic(input_dim=3, hidden_dims=())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    batch = _rollout_batch()
    empty = RolloutBatch(
        obs_features=batch.obs_features[:0],
        actions=batch.actions[:0],
        logprobs=batch.logprobs[:0],
        values=batch.values[:0],
        rewards_hl=batch.rewards_hl[:0],
        rewards_env=batch.rewards_env[:0],
        dones=batch.dones[:0],
        action_masks=batch.action_masks[:0],
        replanned=batch.replanned[:0],
        denoise_steps=batch.denoise_steps[:0],
        nfe=batch.nfe[:0],
        discarded_actions=batch.discarded_actions[:0],
        advantages=batch.advantages[:0],
        returns=batch.returns[:0],
    )

    with pytest.raises(ValueError, match="empty batch"):
        update_replan_ppo(model, optimizer, empty)


def test_replan_actor_critic_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = ReplanActorCritic(input_dim=3, hidden_dims=(5,), activation="relu")
    features = torch.randn(2, 3)
    masks = torch.tensor([[True, True], [False, True]])
    logits_before, values_before = model(features)

    path = tmp_path / "replan_actor_critic.pt"
    config = PPOUpdateConfig(epochs=2, minibatch_size=3)
    save_replan_actor_critic_checkpoint(
        path,
        model,
        metadata={"config": ppo_config_to_dict(config), "step": 7},
    )

    loaded, metadata = load_replan_actor_critic_checkpoint(path)
    logits_after, values_after = loaded(features)
    dist, _ = loaded.distribution(features, masks)

    assert metadata["step"] == 7
    assert metadata["config"]["epochs"] == 2
    assert loaded.input_dim == 3
    assert loaded.hidden_dims == (5,)
    assert loaded.activation == "relu"
    torch.testing.assert_close(logits_before, logits_after)
    torch.testing.assert_close(values_before, values_after)
    torch.testing.assert_close(dist.probs[1], torch.tensor([0.0, 1.0]))
