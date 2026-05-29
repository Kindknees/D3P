import csv
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from adaptive_diffusion.rl.actor_critic import ReplanActorCritic
from adaptive_diffusion.rl.features import ReplanFeatureConfig
from adaptive_diffusion.rl.ppo import save_replan_actor_critic_checkpoint
from scripts import run_final_eval_suite as suite
from scripts.run_final_eval_suite import (
    PPOCheckpointSpec,
    RuntimeArtifactSpec,
    TDArtifactSpec,
    build_eval_jobs,
    parse_csv,
    parse_int_csv,
    parse_ppo_checkpoint,
    parse_runtime_artifact,
    parse_td_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_fake_checkpoint(path):
    feature_config = ReplanFeatureConfig(obs_dim=3, action_dim=2, horizon=4)
    model = ReplanActorCritic(input_dim=feature_config.input_dim, hidden_dims=(8,))
    save_replan_actor_critic_checkpoint(
        path,
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


def test_parse_helpers_accept_csv_and_labeled_checkpoint(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    assert parse_csv("fixed_chunk,td_error_replan") == (
        "fixed_chunk",
        "td_error_replan",
    )
    assert parse_csv("") == ()
    assert parse_int_csv("0,1,2") == (0, 1, 2)

    spec = parse_ppo_checkpoint(f"ppo={checkpoint}")

    assert spec.label == "ppo"
    assert spec.checkpoint == checkpoint
    assert spec.env is None

    env_spec = parse_ppo_checkpoint(f"robomimic_lift:ppo={checkpoint}")
    assert env_spec.env == "robomimic_lift"
    assert env_spec.label == "ppo"
    assert env_spec.checkpoint == checkpoint

    td_spec = parse_td_artifact(
        f"robomimic_lift:{checkpoint}={tmp_path / 'calibration.json'}"
    )
    assert td_spec.env == "robomimic_lift"
    assert td_spec.critic_checkpoint == checkpoint
    assert td_spec.calibration == tmp_path / "calibration.json"

    runtime_spec = parse_runtime_artifact(
        f"robomimic_square:{checkpoint}={tmp_path / 'normalization.npz'}"
    )
    assert runtime_spec.env == "robomimic_square"
    assert runtime_spec.base_policy_checkpoint == checkpoint
    assert runtime_spec.normalization_path == tmp_path / "normalization.npz"


def test_parse_args_defaults_to_final_baseline_methods(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_final_eval_suite.py", "--td_artifact", "robomimic_lift:td.pt=calibration.json"],
    )

    args = suite.parse_args()

    assert args.baseline_methods == ("fixed_chunk", "td_error_replan", "denoise_only")


def test_build_eval_jobs_includes_baseline_and_ppo_commands(tmp_path):
    args = Namespace(
        envs=("robomimic_lift",),
        seeds=(0, 1),
        episodes=5,
        max_episode_steps=10,
        baseline_methods=("fixed_chunk", "td_error_replan"),
        ppo_checkpoints=(PPOCheckpointSpec("ppo", tmp_path / "ppo.pt"), PPOCheckpointSpec("can_ppo", tmp_path / "can.pt", env="robomimic_can")),
        td_critic_checkpoint=tmp_path / "td.pt",
        td_calibration=tmp_path / "calibration.json",
        td_threshold_percentile=95.0,
        td_gamma=0.99,
        deterministic=True,
        device="cpu",
    )

    jobs = build_eval_jobs(args, tmp_path / "runs", python_executable="python")

    assert len(jobs) == 6
    labels = [job.label for job in jobs]
    assert labels.count("fixed_chunk") == 2
    assert labels.count("td_error_p95") == 2
    assert labels.count("ppo") == 2
    assert any("run_baseline.py" in " ".join(job.command) for job in jobs)
    assert any("eval_policy.py" in " ".join(job.command) for job in jobs)


def test_build_eval_jobs_supports_denoise_only_with_td_artifact(tmp_path):
    args = Namespace(
        envs=("robomimic_lift",),
        seeds=(0,),
        episodes=5,
        max_episode_steps=10,
        baseline_methods=("denoise_only",),
        ppo_checkpoints=(),
        td_critic_checkpoint=None,
        td_calibration=None,
        td_artifacts=(
            TDArtifactSpec(
                "robomimic_lift",
                tmp_path / "lift_td.pt",
                tmp_path / "lift_calibration.json",
            ),
        ),
        td_threshold_percentile=95.0,
        td_gamma=0.99,
        denoise_candidates=(4, 8, 12, 20),
        denoise_uncertainty_thresholds=(0.5, 1.0, 1.5),
        deterministic=True,
        device=None,
    )

    jobs = build_eval_jobs(args, tmp_path / "runs", python_executable="python")

    assert len(jobs) == 1
    command = " ".join(jobs[0].command)
    assert jobs[0].label == "denoise_only"
    assert "--method denoise_only" in command
    assert "--td_critic_checkpoint" in command
    assert "lift_td.pt" in command
    assert "--denoise_candidates 4,8,12,20" in command
    assert "--denoise_uncertainty_thresholds 0.5,1.0,1.5" in command


def test_build_eval_jobs_uses_env_specific_td_artifacts(tmp_path):
    args = Namespace(
        envs=("robomimic_lift", "robomimic_can"),
        seeds=(0,),
        episodes=5,
        max_episode_steps=None,
        baseline_methods=("td_error_replan",),
        ppo_checkpoints=(),
        td_critic_checkpoint=None,
        td_calibration=None,
        td_artifacts=(
            TDArtifactSpec(
                "robomimic_lift",
                tmp_path / "lift_td.pt",
                tmp_path / "lift_calibration.json",
            ),
            TDArtifactSpec(
                "robomimic_can",
                tmp_path / "can_td.pt",
                tmp_path / "can_calibration.json",
            ),
        ),
        td_threshold_percentile=95.0,
        td_gamma=0.99,
        deterministic=True,
        device=None,
    )

    jobs = build_eval_jobs(args, tmp_path / "runs", python_executable="python")

    assert len(jobs) == 2
    lift_command = " ".join(jobs[0].command)
    can_command = " ".join(jobs[1].command)
    assert "lift_td.pt" in lift_command
    assert "lift_calibration.json" in lift_command
    assert "can_td.pt" in can_command
    assert "can_calibration.json" in can_command


def test_build_eval_jobs_supports_square_env(tmp_path):
    args = Namespace(
        envs=("robomimic_square",),
        seeds=(0,),
        episodes=5,
        max_episode_steps=400,
        baseline_methods=("fixed_chunk", "td_error_replan"),
        ppo_checkpoints=(
            PPOCheckpointSpec("square_ppo", tmp_path / "square_ppo.pt", env="robomimic_square"),
        ),
        td_critic_checkpoint=None,
        td_calibration=None,
        td_artifacts=(
            TDArtifactSpec(
                "robomimic_square",
                tmp_path / "square_td.pt",
                tmp_path / "square_calibration.json",
            ),
        ),
        td_threshold_percentile=95.0,
        td_gamma=0.99,
        deterministic=True,
        device=None,
    )

    jobs = build_eval_jobs(args, tmp_path / "runs", python_executable="python")

    assert len(jobs) == 3
    commands = [" ".join(job.command) for job in jobs]
    assert all("robomimic_square" in command for command in commands)
    assert any("run_baseline.py" in command and "fixed_chunk" in command for command in commands)
    assert any("run_baseline.py" in command and "square_td.pt" in command for command in commands)
    assert any("eval_policy.py" in command and "square_ppo.pt" in command for command in commands)


def test_build_eval_jobs_passes_square_runtime_artifacts(tmp_path):
    args = Namespace(
        envs=("robomimic_square",),
        seeds=(0,),
        episodes=5,
        max_episode_steps=400,
        baseline_methods=("fixed_chunk",),
        ppo_checkpoints=(
            PPOCheckpointSpec("square_ppo", tmp_path / "square_ppo.pt", env="robomimic_square"),
        ),
        td_critic_checkpoint=None,
        td_calibration=None,
        td_artifacts=(),
        runtime_artifacts=(
            RuntimeArtifactSpec(
                "robomimic_square",
                tmp_path / "square_policy.pt",
                tmp_path / "normalization.npz",
            ),
        ),
        env_meta_path=tmp_path / "square.json",
        td_threshold_percentile=95.0,
        td_gamma=0.99,
        deterministic=True,
        device=None,
    )

    jobs = build_eval_jobs(args, tmp_path / "runs", python_executable="python")

    assert len(jobs) == 2
    commands = [" ".join(job.command) for job in jobs]
    assert all("--base_policy_checkpoint" in command for command in commands)
    assert all("square_policy.pt" in command for command in commands)
    assert all("--normalization_path" in command for command in commands)
    assert all("normalization.npz" in command for command in commands)
    assert all("--env_meta_path" in command for command in commands)
    assert all("square.json" in command for command in commands)


def test_final_eval_suite_fake_ppo_smoke_creates_plots_and_manifest(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    _write_fake_checkpoint(checkpoint)

    subprocess.run(
        [
            sys.executable,
            "scripts/run_final_eval_suite.py",
            "--envs",
            "fake",
            "--seeds",
            "0",
            "--episodes",
            "2",
            "--max_episode_steps",
            "3",
            "--baseline_methods",
            "",
            "--ppo_checkpoint",
            f"fake:ppo={checkpoint}",
            "--timeline_episodes",
            "1",
            "--output_root",
            str(tmp_path / "suites"),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    suite_dirs = sorted((tmp_path / "suites").glob("*_fake_final_eval_suite"))
    assert len(suite_dirs) == 1
    suite_dir = suite_dirs[0]
    expected_files = [
        "suite_config.yaml",
        "suite_commands.csv",
        "suite_results.csv",
        "suite_summary.json",
        "plots/summary_table.csv",
        "plots/aggregate_summary.csv",
        "plots/analysis_summary.json",
        "plots/timeline_manifest.csv",
        "plots/pareto_fake_success.svg",
        "plots/pareto_fake_return.svg",
    ]
    for relative_path in expected_files:
        assert (suite_dir / relative_path).exists()

    timelines = sorted((suite_dir / "plots" / "timelines").glob("*.svg"))
    assert len(timelines) == 1
    assert "<svg" in timelines[0].read_text(encoding="utf-8")

    with (suite_dir / "suite_results.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["label"] == "ppo"
    assert rows[0]["env"] == "fake"
    assert rows[0]["seed"] == "0"

    with (suite_dir / "suite_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["num_jobs"] == 1
    assert summary["num_completed"] == 1
    assert summary["analysis_summary"]["outputs"]["replan_timeline_plots"]

def test_final_eval_suite_resume_with_max_jobs_completes_fake_suite_in_chunks(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    _write_fake_checkpoint(checkpoint)
    suite_dir = tmp_path / "manual_suite"
    base_command = [
        sys.executable,
        "scripts/run_final_eval_suite.py",
        "--envs",
        "fake",
        "--seeds",
        "0,1",
        "--episodes",
        "2",
        "--max_episode_steps",
        "3",
        "--baseline_methods",
        "",
        "--ppo_checkpoint",
        f"fake:ppo={checkpoint}",
        "--timeline_episodes",
        "1",
        "--suite_dir",
        str(suite_dir),
        "--max_jobs",
        "1",
    ]

    subprocess.run(
        base_command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    with (suite_dir / "suite_summary.json").open("r", encoding="utf-8") as f:
        first_summary = json.load(f)
    assert first_summary["num_jobs"] == 2
    assert first_summary["num_completed"] == 1
    assert first_summary["num_pending"] == 1
    assert first_summary["num_ran_this_invocation"] == 1
    assert first_summary["suite_complete"] is False

    subprocess.run(
        [*base_command, "--resume"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    with (suite_dir / "suite_summary.json").open("r", encoding="utf-8") as f:
        final_summary = json.load(f)
    assert final_summary["num_jobs"] == 2
    assert final_summary["num_completed"] == 2
    assert final_summary["num_pending"] == 0
    assert final_summary["num_ran_this_invocation"] == 1
    assert final_summary["suite_complete"] is True
    assert final_summary["resumed"] is True
    with (suite_dir / "suite_results.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert sorted(row["seed"] for row in rows) == ["0", "1"]

