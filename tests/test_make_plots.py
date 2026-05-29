import csv
import json
import subprocess
import sys
from pathlib import Path

from adaptive_diffusion.plots import (
    LabeledRun,
    aggregate_rows,
    load_rows,
    write_analysis_outputs,
    write_timeline_outputs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_eval_summary(path, *, env, method, seed, success, ret, nfe, replan):
    path.mkdir(parents=True, exist_ok=True)
    with (path / "eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "env": env,
                "method": method,
                "seed": seed,
                "episodes": 5,
                "success_rate": success,
                "mean_episode_return": ret,
                "median_episode_return": ret,
                "mean_high_level_return": ret - 0.1,
                "mean_episode_length": 10.0,
                "mean_nfe_per_action": nfe,
                "mean_inference_wall_time_per_action": 0.01,
                "mean_controller_wall_time_per_action": 0.001,
                "mean_total_wall_time_per_action": 0.011,
                "replan_rate": replan,
                "forced_replan_rate": 0.25,
                "learned_replan_rate": max(0.0, replan - 0.25),
                "mean_discarded_actions_per_episode": 2.0,
                "mean_buffer_remaining_when_replanned": 0.5,
            },
            f,
        )


def _write_step_metrics(path):
    fieldnames = [
        "episode_id",
        "t",
        "env_name",
        "method",
        "seed",
        "controller_action",
        "forced_replan",
        "replanned",
        "buffer_remaining_after",
        "plan_age_before",
        "discarded_actions",
        "uncertainty",
        "td_error",
        "env_reward",
    ]
    rows = [
        {
            "episode_id": 0,
            "t": 0,
            "env_name": "lift",
            "method": "td_error_replan",
            "seed": 0,
            "controller_action": 1,
            "forced_replan": True,
            "replanned": True,
            "buffer_remaining_after": 3,
            "plan_age_before": 0,
            "discarded_actions": 0,
            "uncertainty": 0.0,
            "td_error": 0.0,
            "env_reward": 0.0,
        },
        {
            "episode_id": 0,
            "t": 1,
            "env_name": "lift",
            "method": "td_error_replan",
            "seed": 0,
            "controller_action": 0,
            "forced_replan": False,
            "replanned": False,
            "buffer_remaining_after": 2,
            "plan_age_before": 1,
            "discarded_actions": 0,
            "uncertainty": 0.2,
            "td_error": 0.1,
            "env_reward": 0.0,
        },
        {
            "episode_id": 0,
            "t": 2,
            "env_name": "lift",
            "method": "td_error_replan",
            "seed": 0,
            "controller_action": 1,
            "forced_replan": False,
            "replanned": True,
            "buffer_remaining_after": 3,
            "plan_age_before": 2,
            "discarded_actions": 1,
            "uncertainty": 1.2,
            "td_error": 0.8,
            "env_reward": 1.0,
        },
    ]
    path.mkdir(parents=True, exist_ok=True)
    with (path / "metrics_step.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_rows_and_aggregate_multiple_seeds(tmp_path):
    run0 = tmp_path / "fixed_seed0"
    run1 = tmp_path / "fixed_seed1"
    _write_eval_summary(run0, env="lift", method="fixed_chunk", seed=0, success=0.2, ret=2.0, nfe=5.0, replan=0.25)
    _write_eval_summary(run1, env="lift", method="fixed_chunk", seed=1, success=0.4, ret=4.0, nfe=5.2, replan=0.26)

    rows = load_rows([
        LabeledRun("fixed", run0),
        LabeledRun("fixed", run1 / "eval_summary.json"),
    ])
    aggregates = aggregate_rows(rows)

    assert len(rows) == 2
    assert len(aggregates) == 1
    aggregate = aggregates[0]
    assert aggregate["success_rate_mean"] == 0.30000000000000004
    assert aggregate["mean_episode_return_mean"] == 3.0
    assert aggregate["episodes_total"] == 10
    assert aggregate["seeds"] == "0,1"


def test_write_analysis_outputs_creates_tables_and_pareto_svgs(tmp_path):
    rows = [
        {
            "label": "fixed",
            "env": "lift",
            "method": "fixed_chunk",
            "seed": 0,
            "episodes": 5,
            "success_rate": 0.2,
            "mean_episode_return": 2.0,
            "mean_high_level_return": 2.0,
            "mean_nfe_per_action": 5.0,
            "mean_total_wall_time_per_action": 0.01,
            "replan_rate": 0.25,
            "learned_replan_rate": 0.0,
            "mean_discarded_actions_per_episode": 0.0,
            "source_path": "fixed/eval_summary.json",
        },
        {
            "label": "td",
            "env": "lift",
            "method": "td_error_replan",
            "seed": 0,
            "episodes": 5,
            "success_rate": 0.4,
            "mean_episode_return": 4.0,
            "mean_high_level_return": 4.0,
            "mean_nfe_per_action": 6.0,
            "mean_total_wall_time_per_action": 0.012,
            "replan_rate": 0.3,
            "learned_replan_rate": 0.05,
            "mean_discarded_actions_per_episode": 5.0,
            "source_path": "td/eval_summary.json",
        },
    ]

    summary = write_analysis_outputs(tmp_path, rows)

    assert summary["num_rows"] == 2
    assert (tmp_path / "summary_table.csv").exists()
    assert (tmp_path / "aggregate_summary.md").exists()
    assert (tmp_path / "pareto_lift_success.svg").exists()
    assert (tmp_path / "pareto_lift_return.svg").exists()
    assert "<svg" in (tmp_path / "pareto_lift_success.svg").read_text(encoding="utf-8")

    with (tmp_path / "aggregate_summary.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {row["label"] for row in rows} == {"fixed", "td"}


def test_make_plots_cli_accepts_labeled_runs(tmp_path):
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "plots"
    _write_eval_summary(run_dir, env="lift", method="fixed_chunk", seed=0, success=0.2, ret=2.0, nfe=5.0, replan=0.25)

    subprocess.run(
        [
            sys.executable,
            "scripts/make_plots.py",
            "--plot",
            "pareto,summary_table",
            "--run",
            f"fixed={run_dir}",
            "--output",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output_dir / "analysis_summary.json").exists()
    assert (output_dir / "summary_table.md").exists()
    assert (output_dir / "pareto_lift_success.svg").exists()


def test_write_timeline_outputs_creates_replan_timeline_svg(tmp_path):
    run_dir = tmp_path / "run"
    _write_step_metrics(run_dir)

    paths = write_timeline_outputs(tmp_path / "plots", [LabeledRun("td", run_dir)])

    assert len(paths) == 1
    timeline_path = Path(paths[0])
    assert timeline_path.exists()
    text = timeline_path.read_text(encoding="utf-8")
    assert "<svg" in text
    assert "learned replan" in text
    assert (tmp_path / "plots" / "timeline_manifest.csv").exists()


def test_write_timeline_outputs_disambiguates_duplicate_labels(tmp_path):
    run0 = tmp_path / "run0"
    run1 = tmp_path / "run1"
    _write_step_metrics(run0)
    _write_step_metrics(run1)

    paths = write_timeline_outputs(
        tmp_path / "plots",
        [LabeledRun("td", run0), LabeledRun("td", run1)],
    )

    assert len(paths) == 2
    assert len(set(paths)) == 2
    assert Path(paths[0]).name == "replan_timeline_td_ep0.svg"
    assert Path(paths[1]).name == "replan_timeline_td_ep0_run1.svg"


def test_make_plots_cli_writes_timeline_outputs(tmp_path):
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "plots"
    _write_eval_summary(run_dir, env="lift", method="td_error_replan", seed=0, success=0.2, ret=2.0, nfe=6.0, replan=0.3)
    _write_step_metrics(run_dir)

    subprocess.run(
        [
            sys.executable,
            "scripts/make_plots.py",
            "--plot",
            "pareto,summary_table,replan_timeline",
            "--run",
            f"td={run_dir}",
            "--output",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output_dir / "timeline_manifest.csv").exists()
    timelines = sorted((output_dir / "timelines").glob("replan_timeline_td_ep0.svg"))
    assert len(timelines) == 1
    with (output_dir / "analysis_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["outputs"]["replan_timeline_plots"]
