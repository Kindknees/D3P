import json
from argparse import Namespace

import numpy as np
import pytest

from adaptive_diffusion.rl.actor_critic import ReplanActorCritic
from adaptive_diffusion.rl.features import ReplanFeatureConfig
from scripts.train_replan_ppo import (
    BCWarmstartExamples,
    build_bc_teacher_controller,
    collect_bc_warmstart_examples,
    compute_bc_sample_weights,
    load_td_percentile_threshold,
    pretrain_replan_actor_bc,
)


def _fake_args(**overrides):
    values = {
        "env": "fake",
        "seed": 0,
        "chunk_horizon": 4,
        "denoise_steps": 20,
        "fake_obs_dim": 3,
        "fake_action_dim": 2,
        "td_critic_checkpoint": None,
        "td_gamma": 0.99,
        "bc_teacher": "fixed",
        "bc_warmstart_steps": 6,
        "bc_td_calibration": None,
        "bc_td_threshold_percentile": 95.0,
    }
    values.update(overrides)
    return Namespace(**values)


def test_load_td_percentile_threshold_reads_integer_key(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps({"percentile_thresholds": {"95": 0.75}}),
        encoding="utf-8",
    )

    assert load_td_percentile_threshold(calibration, 95.0) == 0.75


def test_build_bc_teacher_controller_sets_td_percentile_threshold(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps({"percentile_thresholds": {"90": 0.25}}),
        encoding="utf-8",
    )
    critic = tmp_path / "td_critic.pt"
    critic.write_bytes(b"placeholder")
    args = _fake_args(
        bc_teacher="td_error",
        td_critic_checkpoint=critic,
        bc_td_calibration=calibration,
        bc_td_threshold_percentile=90.0,
    )

    controller = build_bc_teacher_controller(args)

    assert controller.threshold_mode == "percentile"
    assert controller.calibrated_threshold == 0.25


def test_collect_bc_warmstart_examples_from_fixed_fake_teacher():
    args = _fake_args(bc_teacher="fixed", bc_warmstart_steps=6)
    feature_config = ReplanFeatureConfig(obs_dim=3, action_dim=2, horizon=4)

    examples = collect_bc_warmstart_examples(
        args=args,
        feature_config=feature_config,
        max_episode_steps=6,
        device="cpu",
        seed=0,
    )

    assert examples.size == 6
    assert examples.obs_features.shape == (6, feature_config.input_dim)
    assert examples.action_masks.shape == (6, 2)
    assert examples.actions.tolist() == [1, 0, 0, 0, 1, 0]
    assert examples.forced_replans.tolist() == [True, False, False, False, True, False]


def test_compute_bc_sample_weights_upweights_learned_replans():
    examples = BCWarmstartExamples(
        obs_features=np.zeros((4, 3), dtype=np.float32),
        action_masks=np.ones((4, 2), dtype=bool),
        actions=np.asarray([1, 1, 0, 0], dtype=np.int64),
        forced_replans=np.asarray([True, False, False, True], dtype=bool),
    )

    weights = compute_bc_sample_weights(examples, learned_replan_weight=4.0)

    assert weights.tolist() == [1.0, 4.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="positive"):
        compute_bc_sample_weights(examples, learned_replan_weight=0.0)


def test_pretrain_replan_actor_bc_learns_masked_labels():
    feature_dim = 3
    features = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, -0.5, 0.0],
        ],
        dtype=np.float32,
    )
    examples = BCWarmstartExamples(
        obs_features=np.repeat(features, 8, axis=0),
        action_masks=np.ones((32, 2), dtype=bool),
        actions=np.repeat(np.asarray([1, 1, 0, 0], dtype=np.int64), 8),
        forced_replans=np.asarray([False] * 32, dtype=bool),
    )
    model = ReplanActorCritic(input_dim=feature_dim, hidden_dims=(8,), activation="tanh")

    stats = pretrain_replan_actor_bc(
        model=model,
        examples=examples,
        epochs=80,
        minibatch_size=8,
        learning_rate=0.05,
        device="cpu",
        seed=0,
    )

    assert stats.examples == 32
    assert stats.accuracy >= 0.95
    assert stats.replan_fraction == 0.5
    assert stats.learned_replan_fraction == 0.5
    assert stats.learned_replan_weight == 1.0
    assert stats.mean_sample_weight == 1.0
