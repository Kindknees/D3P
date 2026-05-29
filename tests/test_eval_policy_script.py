from argparse import Namespace
import json
import subprocess
import sys
from pathlib import Path

from adaptive_diffusion.rl.actor_critic import ReplanActorCritic
from adaptive_diffusion.rl.features import ReplanFeatureConfig
from adaptive_diffusion.rl.ppo import save_replan_actor_critic_checkpoint
from scripts.eval_policy import build_eval_args


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_eval_policy_fake_checkpoint_creates_eval_artifacts(tmp_path):
    feature_config = ReplanFeatureConfig(obs_dim=3, action_dim=2, horizon=4)
    model = ReplanActorCritic(input_dim=feature_config.input_dim, hidden_dims=(8,))
    checkpoint = tmp_path / "checkpoint.pt"
    save_replan_actor_critic_checkpoint(
        checkpoint,
        model,
        metadata={
            "config": {
                "env": "fake",
                "chunk_horizon": 4,
                "denoise_steps": 20,
                "max_episode_steps": 3,
                "feature_config": {
                    "obs_dim": feature_config.obs_dim,
                    "action_dim": feature_config.action_dim,
                    "horizon": feature_config.horizon,
                    "input_dim": feature_config.input_dim,
                },
                "reward": {
                    "lambda_C": 0.003,
                    "lambda_D": 0.03,
                    "replan_cost_c0": 0.0,
                    "replan_cost_c1": 1.0,
                },
                "td_gamma": 0.99,
                "device": "cpu",
            }
        },
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/eval_policy.py",
            "--checkpoint",
            str(checkpoint),
            "--episodes",
            "2",
            "--seed",
            "0",
            "--deterministic",
            "true",
            "--output_root",
            str(tmp_path / "evals"),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    eval_dirs = sorted((tmp_path / "evals").glob("*_fake_ppo_replan_only_eval_seed0"))
    assert len(eval_dirs) == 1
    eval_dir = eval_dirs[0]
    expected_files = [
        "config.yaml",
        "metrics_step.csv",
        "metrics_episode.csv",
        "eval_metrics_step.csv",
        "eval_metrics_episode.csv",
        "eval_summary.json",
    ]
    for relative_path in expected_files:
        assert (eval_dir / relative_path).exists()

    with (eval_dir / "eval_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["env"] == "fake"
    assert summary["method"] == "ppo_replan_only"
    assert summary["episodes"] == 2
    assert summary["mean_episode_length"] == 3.0
    assert summary["mean_nfe_per_action"] >= 0.0
    assert summary["replan_rate"] >= 0.0



def test_build_eval_args_uses_checkpoint_td_critic_and_reward_bonus():
    checkpoint_config = {
        "env": "fake",
        "max_episode_steps": 5,
        "chunk_horizon": 4,
        "denoise_steps": 20,
        "feature_config": {"obs_dim": 3, "action_dim": 2, "horizon": 4},
        "reward": {
            "lambda_C": 0.003,
            "lambda_D": 0.03,
            "replan_cost_c0": 0.0,
            "replan_cost_c1": 1.0,
            "uncertainty_replan_bonus": 0.1,
        },
        "td_critic_checkpoint": "/tmp/example_td_critic.pt",
        "td_gamma": 0.97,
        "device": "cpu",
    }
    args = Namespace(
        env=None,
        seed=7,
        max_episode_steps=None,
        chunk_horizon=None,
        denoise_steps=None,
        lambda_C=None,
        lambda_D=None,
        uncertainty_replan_bonus=None,
        td_critic_checkpoint=None,
        td_gamma=None,
        device=None,
    )

    eval_args = build_eval_args(args, checkpoint_config)

    assert eval_args.env == "fake"
    assert eval_args.max_episode_steps == 5
    assert eval_args.uncertainty_replan_bonus == 0.1
    assert eval_args.td_critic_checkpoint == Path("/tmp/example_td_critic.pt")
    assert eval_args.td_gamma == 0.97
