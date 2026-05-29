import numpy as np
import torch

from adaptive_diffusion.buffers import ActionBuffer
from adaptive_diffusion.controllers.ppo_replan import PPOReplanController
from adaptive_diffusion.rl.actor_critic import ReplanActorCritic
from adaptive_diffusion.rl.features import ReplanFeatureConfig


def _zero_model(input_dim):
    model = ReplanActorCritic(input_dim=input_dim, hidden_dims=())
    for param in model.parameters():
        param.data.zero_()
    return model


def test_ppo_replan_controller_forces_replan_on_empty_buffer():
    config = ReplanFeatureConfig(obs_dim=3, action_dim=2, horizon=4)
    model = _zero_model(config.input_dim)
    model.actor[-1].bias.data = torch.tensor([5.0, -5.0])
    controller = PPOReplanController(model, config, deterministic=True)
    buffer = ActionBuffer.empty(horizon=4, action_dim=2)

    decision = controller.decide(buffer, obs={"state": np.zeros(3, dtype=np.float32)})

    assert decision.action == 1
    assert decision.forced_replan
    assert controller.last_decision_info is not None
    np.testing.assert_array_equal(controller.last_decision_info.action_mask, [False, True])
    assert controller.last_decision_info.probs[0] == 0.0


def test_ppo_replan_controller_can_continue_non_empty_buffer():
    config = ReplanFeatureConfig(obs_dim=3, action_dim=2, horizon=4)
    model = _zero_model(config.input_dim)
    model.actor[-1].bias.data = torch.tensor([5.0, -5.0])
    controller = PPOReplanController(model, config, deterministic=True)
    buffer = ActionBuffer(actions=np.ones((2, 2), dtype=np.float32), horizon=4)

    decision = controller.decide(buffer, obs={"state": np.zeros(3, dtype=np.float32)})

    assert decision.action == 0
    assert not decision.forced_replan
    assert controller.last_decision_info is not None
    np.testing.assert_array_equal(controller.last_decision_info.action_mask, [True, True])


def test_ppo_replan_controller_can_choose_replan_non_empty_buffer():
    config = ReplanFeatureConfig(obs_dim=3, action_dim=2, horizon=4)
    model = _zero_model(config.input_dim)
    model.actor[-1].bias.data = torch.tensor([-5.0, 5.0])
    controller = PPOReplanController(model, config, deterministic=True)
    buffer = ActionBuffer(actions=np.ones((2, 2), dtype=np.float32), horizon=4)

    decision = controller.decide(buffer, obs={"state": np.zeros(3, dtype=np.float32)})

    assert decision.action == 1
    assert not decision.forced_replan
