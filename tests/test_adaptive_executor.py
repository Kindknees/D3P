import csv

import numpy as np

from adaptive_diffusion.controllers.three_way import (
    BinaryResetController,
    ThreeWayControllerConfig,
    ThreeWayHeuristicController,
)
from adaptive_diffusion.executor import ExecutorConfig, run_executor_episodes
from adaptive_diffusion.repair import RepairConfig, finalize_repair_candidate


class TinyEnv:
    def __init__(self):
        self.x = 0.0

    def reset(self, seed=None):
        self.x = 0.0
        return self.x

    def step(self, action):
        self.x += float(action[0])
        reward = 1.0 if self.x >= 1.0 else 0.0
        done = self.x >= 1.0
        return self.x, reward, done, {"x": self.x}


class TinyPolicy:
    nfe_per_sample = 4

    def sample_action_chunk(self, obs):
        return np.array([[0.25], [0.25], [0.25], [0.25]], dtype=np.float32)


def test_executor_writes_episode_summary(tmp_path):
    config = ExecutorConfig(
        method="fixed_chunk",
        task="tiny",
        seed=0,
        action_dim=1,
        chunk_size=4,
        max_steps=4,
        success_threshold=1,
    )

    stats = run_executor_episodes(
        env_factory=TinyEnv,
        policy_factory=TinyPolicy,
        config=config,
        output_dir=tmp_path,
    )

    assert stats[0].success
    with (tmp_path / "episode_summary.csv").open(newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    assert rows[0]["method"] == "fixed_chunk"
    assert rows[0]["success"] == "1"


def test_executor_invokes_repair(tmp_path):
    config = ExecutorConfig(
        method="adaptive_commitment_with_repair",
        task="tiny",
        seed=0,
        action_dim=1,
        chunk_size=4,
        max_steps=4,
        success_threshold=1,
        nfe_per_repair=2,
        repair_config=RepairConfig(target_len=4, anchor_rho=1.0),
    )
    controller = ThreeWayHeuristicController(
        ThreeWayControllerConfig(
            repair_td_threshold=0.5,
            high_td_threshold=0.9,
            medium_td_threshold=0.5,
        )
    )

    def uncertainty_fn(obs, info, step):
        return 1.0 if step == 0 else 0.0

    def repair_fn(obs, residual, repair_config):
        repaired = residual.copy()
        repaired[:, 0] = 0.5
        return finalize_repair_candidate(residual, repaired, repair_config)

    stats = run_executor_episodes(
        env_factory=TinyEnv,
        policy_factory=TinyPolicy,
        config=config,
        output_dir=tmp_path,
        controller=controller,
        uncertainty_fn=uncertainty_fn,
        repair_fn=repair_fn,
    )

    assert stats[0].success
    assert stats[0].num_repairs == 1


def test_executor_can_continue_after_rejected_repair(tmp_path):
    config = ExecutorConfig(
        method="adaptive_commitment_with_repair",
        task="tiny",
        seed=0,
        action_dim=1,
        chunk_size=4,
        max_steps=4,
        success_threshold=1,
        nfe_per_repair=2,
        repair_config=RepairConfig(
            target_len=4,
            anchor_rho=1.0,
            accept_min_action_jump=1.0,
        ),
        repair_reject_fallback="continue",
    )
    controller = ThreeWayHeuristicController(
        ThreeWayControllerConfig(
            repair_td_threshold=0.5,
            high_td_threshold=0.9,
            medium_td_threshold=0.5,
        )
    )

    def uncertainty_fn(obs, info, step):
        return 1.0 if step == 0 else 0.0

    def repair_fn(obs, residual, repair_config):
        return finalize_repair_candidate(residual, residual.copy(), repair_config)

    stats = run_executor_episodes(
        env_factory=TinyEnv,
        policy_factory=TinyPolicy,
        config=config,
        output_dir=tmp_path,
        controller=controller,
        uncertainty_fn=uncertainty_fn,
        repair_fn=repair_fn,
    )

    assert stats[0].success
    assert stats[0].num_repairs == 0
    assert stats[0].num_random_resets == 0


class AlternatingPolicy:
    nfe_per_sample = 4

    def __init__(self):
        self.calls = 0

    def sample_action_chunk(self, obs):
        self.calls += 1
        value = 0.25 if self.calls == 1 else 0.75
        return np.array([[value], [value], [value], [value]], dtype=np.float32)


def test_executor_logs_jump_for_nonempty_reset(tmp_path):
    config = ExecutorConfig(
        method="td_replan_anywhere_random_reset",
        task="tiny",
        seed=0,
        action_dim=1,
        chunk_size=4,
        max_steps=3,
        success_threshold=10,
    )
    controller = BinaryResetController(reset_td_threshold=0.5, chunk_size=4)

    def uncertainty_fn(obs, info, step):
        return 1.0 if step == 0 else 0.0

    stats = run_executor_episodes(
        env_factory=TinyEnv,
        policy_factory=AlternatingPolicy,
        config=config,
        output_dir=tmp_path,
        controller=controller,
        uncertainty_fn=uncertainty_fn,
    )

    assert stats[0].num_random_resets == 1
    assert stats[0].mean_action_jump == 0.5


def test_executor_applies_and_logs_action_noise(tmp_path):
    from adaptive_diffusion.executor import ActionNoiseConfig

    config = ExecutorConfig(
        method="fixed_chunk",
        task="tiny",
        seed=0,
        action_dim=1,
        chunk_size=4,
        max_steps=1,
        success_threshold=10,
        action_noise_config=ActionNoiseConfig(
            enabled=True,
            sigma=0.1,
            probability=1.0,
        ),
    )

    run_executor_episodes(
        env_factory=TinyEnv,
        policy_factory=TinyPolicy,
        config=config,
        output_dir=tmp_path,
    )

    with (tmp_path / "step_logs.csv").open(newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    assert rows[0]["action_noise_applied"] == "1"
    assert float(rows[0]["action_noise_l2"]) > 0
    assert float(rows[0]["executed_action_l2"]) != 0.25


class VariableNfePolicy(TinyPolicy):
    def sample_action_chunk_nfe(self, obs, nfe):
        value = 0.5 if nfe == 2 else 0.25
        return np.array([[value], [value], [value], [value]], dtype=np.float32)


def test_executor_uses_decision_sample_nfe(tmp_path):
    from adaptive_diffusion.controllers.three_way import ControllerDecision, RESET

    class OneShotNfeController:
        def decide(self, state):
            if state.remaining_buffer_length == 0:
                return ControllerDecision(RESET, "buffer_empty", 4, sample_nfe=2)
            return ControllerDecision("continue", "within")

    config = ExecutorConfig(
        method="criticality_aware_commitment_with_repair",
        task="tiny",
        seed=0,
        action_dim=1,
        chunk_size=4,
        max_steps=2,
        success_threshold=1,
    )
    stats = run_executor_episodes(
        env_factory=TinyEnv,
        policy_factory=VariableNfePolicy,
        config=config,
        output_dir=tmp_path,
        controller=OneShotNfeController(),
    )
    assert stats[0].total_nfe == 2
    assert stats[0].success
