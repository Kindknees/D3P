import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from adaptive_diffusion.rl.ppo import load_replan_actor_critic_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_train_replan_ppo_fake_smoke_creates_training_artifacts(tmp_path):
    cmd = [
        sys.executable,
        "scripts/train_replan_ppo.py",
        "--env",
        "fake",
        "--seed",
        "0",
        "--total_env_steps",
        "12",
        "--rollout_steps",
        "6",
        "--max_episode_steps",
        "3",
        "--eval_episodes",
        "2",
        "--hidden_dims",
        "8",
        "--minibatch_size",
        "3",
        "--update_epochs",
        "1",
        "--output_root",
        str(tmp_path),
    ]

    subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    runs = sorted(tmp_path.glob("*_fake_ppo_replan_only_seed0"))
    assert len(runs) == 1
    run_dir = runs[0]

    expected_files = [
        "config.yaml",
        "metrics_step.csv",
        "metrics_episode.csv",
        "training_updates.csv",
        "train_summary.json",
        "checkpoints/latest.pt",
        "checkpoints/best.pt",
        "eval/eval_summary.json",
        "eval/eval_metrics_step.csv",
        "eval/eval_metrics_episode.csv",
    ]
    for relative_path in expected_files:
        assert (run_dir / relative_path).exists()

    with (run_dir / "train_summary.json").open("r", encoding="utf-8") as f:
        train_summary = json.load(f)
    assert train_summary["env"] == "fake"
    assert train_summary["method"] == "ppo_replan_only"
    assert train_summary["actual_env_steps"] >= 12
    assert train_summary["updates"] >= 1
    assert train_summary["train_summary"]["mean_nfe_per_action"] >= 0.0

    with (run_dir / "eval" / "eval_summary.json").open("r", encoding="utf-8") as f:
        eval_summary = json.load(f)
    assert eval_summary["episodes"] == 2
    assert eval_summary["mean_nfe_per_action"] >= 0.0
    assert eval_summary["replan_rate"] >= 0.0

    with (run_dir / "training_updates.csv").open("r", encoding="utf-8") as f:
        update_rows = list(csv.DictReader(f))
    assert update_rows
    assert int(update_rows[-1]["env_steps"]) >= 12
    assert int(update_rows[-1]["continue_actions"]) >= 0
    assert int(update_rows[-1]["replan_actions"]) > 0
    for key in ("loss", "policy_loss", "value_loss", "entropy", "approx_kl"):
        assert np.isfinite(float(update_rows[-1][key]))

    model, metadata = load_replan_actor_critic_checkpoint(
        run_dir / "checkpoints" / "best.pt"
    )
    assert model.input_dim > 0
    assert metadata["env_steps"] >= 12
    assert metadata["eval_summary"]["episodes"] == 2
