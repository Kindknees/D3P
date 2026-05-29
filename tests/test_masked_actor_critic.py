import torch
import pytest

from adaptive_diffusion.rl.actor_critic import ReplanActorCritic
from adaptive_diffusion.rl.masked_categorical import MaskedCategorical


def test_masked_categorical_zeroes_invalid_action_probability():
    logits = torch.tensor([[2.0, -2.0], [0.0, 1.0]])
    mask = torch.tensor([[True, False], [True, True]])

    dist = MaskedCategorical(logits=logits, mask=mask)

    assert dist.probs[0, 0].item() == 1.0
    assert dist.probs[0, 1].item() == 0.0
    assert torch.all(dist.probs[1] > 0.0)


def test_masked_categorical_requires_valid_action():
    with pytest.raises(ValueError):
        MaskedCategorical(torch.zeros((1, 2)), torch.tensor([[False, False]]))


def test_actor_critic_act_respects_forced_replan_mask():
    model = ReplanActorCritic(input_dim=5, hidden_dims=())
    for param in model.parameters():
        param.data.zero_()
    model.actor[-1].bias.data = torch.tensor([5.0, -5.0])
    features = torch.ones(5)

    forced = model.act(features, torch.tensor([False, True]), deterministic=True)
    unforced = model.act(features, torch.tensor([True, True]), deterministic=True)

    assert forced["action"].item() == 1
    assert forced["probs"].shape == (1, 2)
    assert forced["probs"][0, 0].item() == 0.0
    assert unforced["action"].item() == 0
    assert unforced["value"].shape == (1,)


def test_actor_critic_evaluate_actions_returns_batch_shapes():
    model = ReplanActorCritic(input_dim=3, hidden_dims=(4,), activation="relu")
    features = torch.ones((2, 3))
    masks = torch.tensor([[True, True], [False, True]])
    actions = torch.tensor([0, 1])

    output = model.evaluate_actions(features, masks, actions)

    assert output["logprob"].shape == (2,)
    assert output["entropy"].shape == (2,)
    assert output["value"].shape == (2,)
    assert output["probs"].shape == (2, 2)
