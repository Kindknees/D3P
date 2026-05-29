import numpy as np
import pytest

from adaptive_diffusion.controllers.denoise_adaptor import (
    DenoiseOnlyController,
    UncertaintyRuleDenoiseAdaptor,
)
from adaptive_diffusion.controllers.fixed import FixedChunkController
from adaptive_diffusion.executor import AdaptiveDiffusionExecutor


class _Space:
    def __init__(self, shape):
        self.shape = shape


class _FakePolicy:
    action_dim = 2
    chunk_horizon = 4
    default_denoise_steps = 20

    def __init__(self):
        self.sample_calls = 0
        self.requested_denoise_steps = []

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
        steps = (
            self.default_denoise_steps if denoise_steps is None else int(denoise_steps)
        )
        self.requested_denoise_steps.append(steps)
        chunk = np.tile(np.array([[0.25, -0.25]], dtype=np.float32), (4, 1))
        return chunk, {
            "denoise_steps": steps,
            "nfe": steps,
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


def test_executor_runs_one_episode_without_buffer_underflow(tmp_path):
    policy = _FakePolicy()
    executor = AdaptiveDiffusionExecutor(
        env=_FakeVectorEnv(),
        diffusion_policy=policy,
        controller=FixedChunkController(),
        config={
            "method": "fixed_chunk",
            "env_name": "fake",
            "seed": 0,
            "max_episode_steps": 10,
            "success_reward_threshold": 1.0,
            "save_transitions": True,
        },
    )

    episode = executor.run_episode(episode_id=0, deterministic=True, seed=0)

    assert episode["episode_length"] == 5
    assert episode["success"]
    assert episode["replans"] == 2
    assert episode["forced_replans"] == 2
    assert episode["total_nfe"] == 40
    assert episode["high_level_return"] == episode["episode_return"]
    assert policy.sample_calls == 2
    assert all(row["buffer_remaining_after"] >= 0 for row in executor.step_metrics)

    required_step_fields = {
        "episode_id",
        "t",
        "env_name",
        "method",
        "seed",
        "obs_norm",
        "controller_action",
        "forced_replan",
        "replanned",
        "buffer_remaining_before",
        "buffer_remaining_after",
        "plan_age_before",
        "discarded_actions",
        "denoise_steps",
        "nfe",
        "inference_wall_time_sec",
        "controller_wall_time_sec",
        "uncertainty",
        "td_error",
        "env_reward",
        "hl_reward",
        "done",
        "success",
    }
    assert required_step_fields.issubset(executor.step_metrics[0])
    assert executor.step_metrics[0]["obs_norm"] == 0.0
    assert executor.step_metrics[1]["obs_norm"] > 0.0
    assert executor.step_metrics[0]["uncertainty"] == 0.0
    assert executor.step_metrics[0]["td_error"] == 0.0
    assert executor.step_metrics[0]["hl_reward"] == executor.step_metrics[0]["env_reward"]

    summary = executor.save_metrics(tmp_path)

    assert (tmp_path / "metrics_step.csv").exists()
    assert (tmp_path / "metrics_episode.csv").exists()
    assert (tmp_path / "eval_summary.json").exists()
    assert (tmp_path / "transitions.npz").exists()
    with np.load(tmp_path / "transitions.npz") as data:
        assert data["obs"].shape == (5, 3)
        assert data["next_obs"].shape == (5, 3)
        assert data["actions"].shape == (5, 2)
        assert data["rewards"].shape == (5,)
        assert data["dones"].shape == (5,)
        assert data["episode_ids"].tolist() == [0, 0, 0, 0, 0]
        assert data["timesteps"].tolist() == [0, 1, 2, 3, 4]
        assert data["env_name"].item() == "fake"
        assert data["method"].item() == "fixed_chunk"
    assert summary["mean_nfe_per_action"] == 8.0
    assert summary["replan_rate"] == 0.4
    assert summary["learned_replan_rate"] == 0.0
    assert summary["mean_controller_wall_time_per_action"] >= 0.0
    assert summary["mean_total_wall_time_per_action"] >= summary[
        "mean_inference_wall_time_per_action"
    ]


def test_high_level_reward_adds_uncertainty_bonus_only_for_replans():
    executor = AdaptiveDiffusionExecutor(
        env=_FakeVectorEnv(),
        diffusion_policy=_FakePolicy(),
        controller=FixedChunkController(),
        config={
            "method": "fixed_chunk",
            "env_name": "fake",
            "seed": 0,
            "max_episode_steps": 10,
            "lambda_C": 0.01,
            "lambda_D": 0.5,
            "replan_cost_c1": 1.0,
            "uncertainty_replan_bonus": 0.1,
        },
    )

    replanned_reward = executor._high_level_reward(
        reward=1.0,
        replanned=True,
        buffer_remaining_before=2,
        nfe=20,
        uncertainty=2.0,
    )
    continued_reward = executor._high_level_reward(
        reward=1.0,
        replanned=False,
        buffer_remaining_before=2,
        nfe=20,
        uncertainty=2.0,
    )
    negative_uncertainty_reward = executor._high_level_reward(
        reward=1.0,
        replanned=True,
        buffer_remaining_before=2,
        nfe=20,
        uncertainty=-2.0,
    )

    assert replanned_reward == pytest.approx(0.5)
    assert continued_reward == pytest.approx(1.0)
    assert negative_uncertainty_reward == pytest.approx(0.3)


def test_executor_uses_controller_selected_denoise_steps_on_replan():
    policy = _FakePolicy()
    controller = DenoiseOnlyController(
        UncertaintyRuleDenoiseAdaptor(
            candidate_steps=(4, 20),
            uncertainty_thresholds=(1.0,),
        )
    )
    executor = AdaptiveDiffusionExecutor(
        env=_FakeVectorEnv(),
        diffusion_policy=policy,
        controller=controller,
        config={
            "method": "denoise_only",
            "env_name": "fake",
            "seed": 0,
            "max_episode_steps": 10,
            "success_reward_threshold": 1.0,
        },
    )

    episode = executor.run_episode(episode_id=0, deterministic=True, seed=0)

    assert episode["episode_length"] == 5
    assert episode["total_nfe"] == 8
    assert policy.requested_denoise_steps == [4, 4]
    replan_rows = [row for row in executor.step_metrics if row["replanned"]]
    continue_rows = [row for row in executor.step_metrics if not row["replanned"]]
    assert [row["denoise_steps"] for row in replan_rows] == [4, 4]
    assert all(row["denoise_steps"] == 0 for row in continue_rows)
