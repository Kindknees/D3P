import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_td_threshold_sweep.py"
spec = importlib.util.spec_from_file_location("run_td_threshold_sweep", SCRIPT_PATH)
sweep = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = sweep
spec.loader.exec_module(sweep)


def _write_calibration(path):
    path.write_text(
        json.dumps(
            {
                "percentile_thresholds": {
                    "70": -0.1,
                    "90": 0.5,
                }
            }
        ),
        encoding="utf-8",
    )


def test_build_sweep_specs_supports_percentile_and_z_score(tmp_path):
    calibration = tmp_path / "calibration.json"
    _write_calibration(calibration)

    specs = sweep.build_sweep_specs(
        threshold_mode="both",
        percentiles=(70, 90),
        z_thresholds=(0.5, 1.0),
        calibration_path=calibration,
    )

    assert [item.label for item in specs] == ["p70", "p90", "z0.5", "z1"]
    assert specs[0].threshold_mode == "percentile"
    assert specs[0].calibrated_threshold == -0.1
    assert specs[2].threshold_mode == "z_score"
    assert specs[2].threshold == 0.5


def test_build_sweep_specs_requires_percentile_in_calibration(tmp_path):
    calibration = tmp_path / "calibration.json"
    _write_calibration(calibration)

    with pytest.raises(KeyError):
        sweep.build_sweep_specs(
            threshold_mode="percentile",
            percentiles=(80,),
            z_thresholds=(),
            calibration_path=calibration,
        )


def test_parse_run_output_dir_extracts_baseline_output():
    stdout = "before\nWrote metrics to /tmp/example_run\nafter\n"

    assert sweep.parse_run_output_dir(stdout) == Path("/tmp/example_run")


def test_write_sweep_outputs_creates_analysis_files(tmp_path):
    rows = [
        {
            "threshold_mode": "percentile",
            "label": "p70",
            "threshold": "",
            "threshold_percentile": 70.0,
            "calibrated_threshold": -0.1,
            "run_dir": "/tmp/p70",
            "env": "lift",
            "method": "td_error_replan",
            "seed": 0,
            "episodes": 2,
            "success_rate": 0.0,
            "mean_episode_return": 0.0,
            "median_episode_return": 0.0,
            "mean_episode_length": 20.0,
            "mean_nfe_per_action": 8.0,
            "mean_inference_wall_time_per_action": 0.01,
            "mean_total_wall_time_per_action": 0.011,
            "replan_rate": 0.4,
            "forced_replan_rate": 0.25,
            "learned_replan_rate": 0.15,
            "mean_discarded_actions_per_episode": 4.0,
            "mean_buffer_remaining_when_replanned": 1.0,
        },
        {
            "threshold_mode": "percentile",
            "label": "p90",
            "threshold": "",
            "threshold_percentile": 90.0,
            "calibrated_threshold": 0.5,
            "run_dir": "/tmp/p90",
            "env": "lift",
            "method": "td_error_replan",
            "seed": 0,
            "episodes": 2,
            "success_rate": 0.0,
            "mean_episode_return": 0.0,
            "median_episode_return": 0.0,
            "mean_episode_length": 20.0,
            "mean_nfe_per_action": 6.0,
            "mean_inference_wall_time_per_action": 0.01,
            "mean_total_wall_time_per_action": 0.011,
            "replan_rate": 0.3,
            "forced_replan_rate": 0.25,
            "learned_replan_rate": 0.05,
            "mean_discarded_actions_per_episode": 2.0,
            "mean_buffer_remaining_when_replanned": 0.5,
        },
    ]

    summary = sweep.write_sweep_outputs(tmp_path, rows)

    assert summary["num_runs"] == 2
    assert summary["best_label"] == "p90"
    assert (tmp_path / "sweep_results.csv").exists()
    assert (tmp_path / "sweep_summary.json").exists()
    assert (tmp_path / "sweep_results.md").exists()
    assert "p90" in (tmp_path / "sweep_results.md").read_text(encoding="utf-8")


def test_build_run_command_maps_threshold_arguments(tmp_path):
    args = Namespace(
        env="robomimic_lift",
        chunk_horizon=4,
        denoise_steps=20,
        episodes=2,
        seed=0,
        debug=True,
        max_episode_steps=80,
        critic_checkpoint=tmp_path / "critic.pt",
        calibration=tmp_path / "calibration.json",
        td_gamma=0.99,
    )
    percentile_spec = sweep.SweepSpec(
        threshold_mode="percentile",
        label="p90",
        percentile=90.0,
        calibrated_threshold=0.5,
    )
    z_spec = sweep.SweepSpec(threshold_mode="z_score", label="z1", threshold=1.0)

    percentile_command = sweep.build_run_command(args, percentile_spec, tmp_path / "runs")
    z_command = sweep.build_run_command(args, z_spec, tmp_path / "runs")

    assert "--max_episode_steps" in percentile_command
    assert "80" in percentile_command
    assert "--td_threshold_percentile" in percentile_command
    assert "--td_calibration" in percentile_command
    assert "--td_threshold" not in percentile_command
    assert "--td_threshold" in z_command
    assert "--td_calibration" not in z_command
