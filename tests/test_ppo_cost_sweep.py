import csv
import importlib.util
import subprocess
import json
import sys
from argparse import Namespace
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_ppo_cost_sweep.py"
spec = importlib.util.spec_from_file_location("run_ppo_cost_sweep", SCRIPT_PATH)
sweep = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = sweep
spec.loader.exec_module(sweep)


def test_build_sweep_specs_crosses_cost_values():
    specs = sweep.build_sweep_specs((0.0, 0.003), (0.0, 0.03))

    assert [item.label for item in specs] == [
        "c0_d0",
        "c0_d0p03",
        "c0p003_d0",
        "c0p003_d0p03",
    ]
    assert specs[-1].lambda_C == 0.003
    assert specs[-1].lambda_D == 0.03
    assert specs[-1].uncertainty_replan_bonus == 0.0


def test_build_sweep_specs_crosses_uncertainty_bonus_values():
    specs = sweep.build_sweep_specs((0.003,), (0.03,), (0.0, 0.1))

    assert [item.label for item in specs] == [
        "c0p003_d0p03",
        "c0p003_d0p03_b0p1",
    ]
    assert specs[-1].uncertainty_replan_bonus == 0.1


def test_parse_training_output_dir_extracts_training_run():
    stdout = "before\nWrote PPO training run to /tmp/ppo_run\nafter\n"

    assert sweep.parse_training_output_dir(stdout) == Path("/tmp/ppo_run")


def test_row_from_summary_extracts_train_eval_and_update_metrics(tmp_path):
    summary = {
        "env": "fake",
        "seed": 0,
        "actual_env_steps": 12,
        "updates": 2,
        "train_summary": {
            "episodes": 4,
            "success_rate": 1.0,
            "mean_episode_return": 1.2,
            "mean_high_level_return": 1.0,
            "mean_nfe_per_action": 10.0,
            "replan_rate": 0.5,
            "forced_replan_rate": 0.25,
            "learned_replan_rate": 0.25,
            "mean_discarded_actions_per_episode": 1.0,
        },
        "eval_summary": {
            "episodes": 2,
            "success_rate": 1.0,
            "mean_episode_return": 1.1,
            "mean_high_level_return": 0.9,
            "mean_nfe_per_action": 15.0,
            "replan_rate": 0.75,
            "forced_replan_rate": 0.25,
            "learned_replan_rate": 0.5,
            "mean_discarded_actions_per_episode": 2.0,
        },
        "last_update": {
            "continue_actions": 3,
            "replan_actions": 3,
            "forced_replans": 1,
            "loss": 0.1,
            "entropy": 0.5,
        },
    }
    spec = sweep.CostSweepSpec(lambda_C=0.003, lambda_D=0.03)

    row = sweep.row_from_summary(spec, tmp_path / "run", summary)

    assert row["label"] == "c0p003_d0p03"
    assert row["lambda_C"] == 0.003
    assert row["uncertainty_replan_bonus"] == 0.0
    assert row["eval_mean_nfe_per_action"] == 15.0
    assert row["last_update_continue_actions"] == 3
    assert row["last_update_replan_actions"] == 3


def test_write_sweep_outputs_creates_analysis_files(tmp_path):
    rows = [
        {
            "label": "c0_d0",
            "lambda_C": 0.0,
            "lambda_D": 0.0,
            "uncertainty_replan_bonus": 0.0,
            "run_dir": "/tmp/c0",
            "env": "fake",
            "seed": 0,
            "actual_env_steps": 12,
            "updates": 1,
            "train_episodes": 2,
            "train_success_rate": 1.0,
            "train_mean_episode_return": 1.0,
            "train_mean_high_level_return": 1.0,
            "train_mean_nfe_per_action": 20.0,
            "train_replan_rate": 1.0,
            "train_forced_replan_rate": 0.25,
            "train_learned_replan_rate": 0.75,
            "train_discarded_actions_per_episode": 3.0,
            "eval_episodes": 2,
            "eval_success_rate": 1.0,
            "eval_mean_episode_return": 1.0,
            "eval_mean_high_level_return": 1.0,
            "eval_mean_nfe_per_action": 20.0,
            "eval_replan_rate": 1.0,
            "eval_forced_replan_rate": 0.25,
            "eval_learned_replan_rate": 0.75,
            "eval_discarded_actions_per_episode": 3.0,
            "last_update_continue_actions": 0,
            "last_update_replan_actions": 6,
            "last_update_forced_replans": 2,
            "last_update_loss": 0.1,
            "last_update_entropy": 0.5,
        },
        {
            "label": "c0p01_d0p03",
            "lambda_C": 0.01,
            "lambda_D": 0.03,
            "uncertainty_replan_bonus": 0.1,
            "run_dir": "/tmp/c1",
            "env": "fake",
            "seed": 0,
            "actual_env_steps": 12,
            "updates": 1,
            "train_episodes": 2,
            "train_success_rate": 1.0,
            "train_mean_episode_return": 1.0,
            "train_mean_high_level_return": 0.8,
            "train_mean_nfe_per_action": 10.0,
            "train_replan_rate": 0.5,
            "train_forced_replan_rate": 0.25,
            "train_learned_replan_rate": 0.25,
            "train_discarded_actions_per_episode": 1.0,
            "eval_episodes": 2,
            "eval_success_rate": 1.0,
            "eval_mean_episode_return": 1.0,
            "eval_mean_high_level_return": 0.8,
            "eval_mean_nfe_per_action": 10.0,
            "eval_replan_rate": 0.5,
            "eval_forced_replan_rate": 0.25,
            "eval_learned_replan_rate": 0.25,
            "eval_discarded_actions_per_episode": 1.0,
            "last_update_continue_actions": 3,
            "last_update_replan_actions": 3,
            "last_update_forced_replans": 1,
            "last_update_loss": 0.2,
            "last_update_entropy": 0.4,
        },
    ]

    summary = sweep.write_sweep_outputs(tmp_path, rows)

    assert summary["num_runs"] == 2
    assert summary["best_label"] == "c0_d0"
    assert (tmp_path / "sweep_results.csv").exists()
    assert (tmp_path / "sweep_summary.json").exists()
    assert (tmp_path / "sweep_results.md").exists()
    with (tmp_path / "sweep_summary.json").open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["rows"][0]["label"] == "c0_d0"


def test_build_run_command_maps_cost_arguments(tmp_path):
    args = Namespace(
        env="fake",
        seed=0,
        total_env_steps=12,
        rollout_steps=6,
        eval_episodes=2,
        chunk_horizon=4,
        denoise_steps=20,
        hidden_dims="8",
        learning_rate=3e-4,
        minibatch_size=3,
        update_epochs=1,
        max_episode_steps=3,
        td_critic_checkpoint=tmp_path / "td_critic.pt",
        td_gamma=0.97,
        device="cpu",
    )
    spec = sweep.CostSweepSpec(
        lambda_C=0.003,
        lambda_D=0.03,
        uncertainty_replan_bonus=0.1,
    )

    command = sweep.build_run_command(args, spec, tmp_path / "runs")

    assert "--lambda_C" in command
    assert "0.003" in command
    assert "--lambda_D" in command
    assert "0.03" in command
    assert "--uncertainty_replan_bonus" in command
    assert "0.1" in command
    assert "--max_episode_steps" in command
    assert "3" in command
    assert "--td_critic_checkpoint" in command
    assert str(tmp_path / "td_critic.pt") in command
    assert "--td_gamma" in command
    assert "0.97" in command
    assert "--device" in command
    assert "cpu" in command


def test_run_ppo_cost_sweep_fake_smoke_creates_sweep_artifacts(tmp_path):
    command = [
        sys.executable,
        "scripts/run_ppo_cost_sweep.py",
        "--env",
        "fake",
        "--lambda_C_values",
        "0.0,0.01",
        "--lambda_D_values",
        "0.0",
        "--seed",
        "0",
        "--total_env_steps",
        "12",
        "--rollout_steps",
        "6",
        "--max_episode_steps",
        "3",
        "--eval_episodes",
        "1",
        "--hidden_dims",
        "8",
        "--minibatch_size",
        "3",
        "--update_epochs",
        "1",
        "--device",
        "cpu",
        "--output_root",
        str(tmp_path),
    ]

    subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    sweeps = sorted(tmp_path.glob("*_fake_ppo_cost_sweep_seed0"))
    assert len(sweeps) == 1
    sweep_dir = sweeps[0]
    for relative_path in (
        "sweep_config.yaml",
        "sweep_results.csv",
        "sweep_summary.json",
        "sweep_results.md",
    ):
        assert (sweep_dir / relative_path).exists()

    with (sweep_dir / "sweep_results.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [row["label"] for row in rows] == ["c0_d0", "c0p01_d0"]
    assert all(float(row["eval_mean_nfe_per_action"]) >= 0.0 for row in rows)
    assert all((Path(row["run_dir"]) / "train_summary.json").exists() for row in rows)

    with (sweep_dir / "sweep_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["num_runs"] == 2
    assert summary["best_label"] in {"c0_d0", "c0p01_d0"}
