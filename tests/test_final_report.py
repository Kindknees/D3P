import csv
import json
from pathlib import Path

from scripts import build_final_report as report


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_suite(path: Path, env: str):
    rows = [
        {
            "label": "fixed_chunk",
            "env": env,
            "method": "fixed_chunk",
            "num_runs": "3",
            "seeds": "0,1,2",
            "episodes_total": "300",
            "success_rate_mean": "0.7",
            "success_rate_sem": "0.01",
            "mean_episode_return_mean": "10.0",
            "mean_episode_return_sem": "1.0",
            "mean_high_level_return_mean": "10.0",
            "mean_high_level_return_sem": "1.0",
            "mean_nfe_per_action_mean": "5.0",
            "mean_nfe_per_action_sem": "0.0",
            "mean_total_wall_time_per_action_mean": "0.1",
            "mean_total_wall_time_per_action_sem": "0.0",
            "replan_rate_mean": "0.25",
            "replan_rate_sem": "0.0",
            "learned_replan_rate_mean": "0.0",
            "learned_replan_rate_sem": "0.0",
            "mean_discarded_actions_per_episode_mean": "0.0",
            "mean_discarded_actions_per_episode_sem": "0.0",
        },
        {
            "label": "adaptive",
            "env": env,
            "method": "ppo_replan_only",
            "num_runs": "3",
            "seeds": "0,1,2",
            "episodes_total": "300",
            "success_rate_mean": "0.8",
            "success_rate_sem": "0.02",
            "mean_episode_return_mean": "9.0",
            "mean_episode_return_sem": "0.5",
            "mean_high_level_return_mean": "12.0",
            "mean_high_level_return_sem": "0.5",
            "mean_nfe_per_action_mean": "7.0",
            "mean_nfe_per_action_sem": "0.1",
            "mean_total_wall_time_per_action_mean": "0.2",
            "mean_total_wall_time_per_action_sem": "0.0",
            "replan_rate_mean": "0.35",
            "replan_rate_sem": "0.01",
            "learned_replan_rate_mean": "0.1",
            "learned_replan_rate_sem": "0.01",
            "mean_discarded_actions_per_episode_mean": "20.0",
            "mean_discarded_actions_per_episode_sem": "1.0",
        },
    ]
    _write_csv(path / "plots" / "aggregate_summary.csv", rows)
    return path


def _audit(tmp_path):
    lift_suite = _write_suite(tmp_path / "lift_suite", "lift")
    return {
        "source_path": str(tmp_path / "audit.json"),
        "all_goal_evidence_ready": False,
        "all_suites_ready": False,
        "readiness": {
            "ready": False,
            "missing": ["robomimic_square"],
            "envs": [
                {"env": "robomimic_lift", "missing": []},
                {
                    "env": "robomimic_square",
                    "missing": ["base_policy_checkpoint", "normalization"],
                },
            ],
        },
        "markdown": {"ready": True},
        "envs": [
            {
                "env": "robomimic_lift",
                "suite_path": str(lift_suite),
                "suite_ready": True,
                "checks": [],
            },
            {
                "env": "robomimic_square",
                "suite_path": None,
                "suite_ready": False,
                "missing_rows": ["fixed_chunk:seed0", "denoise_only:seed0"],
                "checks": [{"label": "suite_provided", "passed": False}],
            },
        ],
    }


def test_build_report_collects_aggregate_rows_and_missing_env(tmp_path):
    built = report.build_report(_audit(tmp_path))

    assert built["all_goal_evidence_ready"] is False
    assert len(built["rows"]) == 3
    lift = next(summary for summary in built["env_summaries"] if summary["env"] == "lift")
    square = next(summary for summary in built["env_summaries"] if summary["env"] == "square")

    assert lift["ready"] is True
    assert lift["best_success"]["label"] == "adaptive"
    assert lift["best_high_level_return"]["label"] == "adaptive"
    assert lift["lowest_compute"]["label"] == "fixed_chunk"
    assert lift["lowest_wall_time"]["label"] == "fixed_chunk"
    fixed_row = next(row for row in built["rows"] if row["label"] == "fixed_chunk")
    adaptive_row = next(row for row in built["rows"] if row["label"] == "adaptive")
    assert fixed_row["mean_high_level_return_mean"] == 10.0
    assert fixed_row["success_delta_vs_fixed"] == 0.0
    assert fixed_row["nfe_ratio_vs_fixed"] == 1.0
    assert adaptive_row["mean_total_wall_time_per_action_mean"] == 0.2
    assert adaptive_row["learned_replan_rate_sem"] == 0.01
    assert round(adaptive_row["success_delta_vs_fixed"], 6) == 0.1
    assert adaptive_row["return_delta_vs_fixed"] == -1.0
    assert adaptive_row["high_level_return_delta_vs_fixed"] == 2.0
    assert adaptive_row["nfe_ratio_vs_fixed"] == 1.4
    assert adaptive_row["wall_time_ratio_vs_fixed"] == 2.0
    assert round(adaptive_row["replan_rate_delta_vs_fixed"], 6) == 0.1
    assert adaptive_row["learned_replan_rate_delta_vs_fixed"] == 0.1
    assert adaptive_row["discarded_actions_delta_vs_fixed"] == 20.0
    assert square["ready"] is False
    assert square["status"] == "missing:suite_provided"
    assert square["missing_rows"] == ["fixed_chunk:seed0", "denoise_only:seed0"]
    assert square["readiness_missing"] == ["base_policy_checkpoint", "normalization"]
    assert square["failed_checks"] == ["suite_provided"]


def test_build_report_collects_multiple_suite_paths_for_same_env(tmp_path):
    base_suite = _write_suite(tmp_path / "lift_suite", "lift")
    denoise_suite = tmp_path / "denoise_suite"
    _write_csv(
        denoise_suite / "plots" / "aggregate_summary.csv",
        [
            {
                "label": "fixed_chunk",
                "env": "lift",
                "method": "fixed_chunk",
                "num_runs": "1",
                "seeds": "0",
                "episodes_total": "15",
                "success_rate_mean": "0.0",
                "success_rate_sem": "0.0",
                "mean_episode_return_mean": "0.0",
                "mean_episode_return_sem": "0.0",
                "mean_high_level_return_mean": "0.0",
                "mean_high_level_return_sem": "0.0",
                "mean_nfe_per_action_mean": "5.0",
                "mean_nfe_per_action_sem": "0.0",
                "mean_total_wall_time_per_action_mean": "0.1",
                "mean_total_wall_time_per_action_sem": "0.0",
                "replan_rate_mean": "0.25",
                "replan_rate_sem": "0.0",
                "learned_replan_rate_mean": "0.0",
                "learned_replan_rate_sem": "0.0",
                "mean_discarded_actions_per_episode_mean": "0.0",
                "mean_discarded_actions_per_episode_sem": "0.0",
            },
            {
                "label": "denoise_only",
                "env": "lift",
                "method": "denoise_only",
                "num_runs": "3",
                "seeds": "0,1,2",
                "episodes_total": "300",
                "success_rate_mean": "0.6",
                "success_rate_sem": "0.03",
                "mean_episode_return_mean": "8.0",
                "mean_episode_return_sem": "0.4",
                "mean_high_level_return_mean": "8.0",
                "mean_high_level_return_sem": "0.4",
                "mean_nfe_per_action_mean": "2.0",
                "mean_nfe_per_action_sem": "0.2",
                "mean_total_wall_time_per_action_mean": "0.05",
                "mean_total_wall_time_per_action_sem": "0.0",
                "replan_rate_mean": "0.25",
                "replan_rate_sem": "0.0",
                "learned_replan_rate_mean": "0.0",
                "learned_replan_rate_sem": "0.0",
                "mean_discarded_actions_per_episode_mean": "0.0",
                "mean_discarded_actions_per_episode_sem": "0.0",
            },
        ],
    )
    audit = _audit(tmp_path)
    audit["envs"][0]["suite_paths"] = [str(base_suite), str(denoise_suite)]

    built = report.build_report(audit)
    lift = next(summary for summary in built["env_summaries"] if summary["env"] == "lift")

    lift_rows = [row for row in built["rows"] if row["env"] == "lift"]
    assert len(lift_rows) == 3
    assert [row["label"] for row in lift_rows].count("fixed_chunk") == 1
    fixed_row = next(row for row in lift_rows if row["label"] == "fixed_chunk")
    assert fixed_row["episodes_total"] == "300"
    assert fixed_row["success_rate_mean"] == 0.7
    assert lift["best_success"]["label"] == "adaptive"
    assert lift["lowest_compute"]["label"] == "denoise_only"
    denoise_row = next(row for row in built["rows"] if row["label"] == "denoise_only")
    assert denoise_row["source_suite"] == str(denoise_suite)


def test_write_report_outputs_json_csv_markdown_and_html(tmp_path):
    built = report.build_report(_audit(tmp_path))
    outputs = report.write_report(tmp_path / "out", built)

    assert Path(outputs["json"]).exists()
    assert Path(outputs["csv"]).exists()
    assert Path(outputs["env_summary_csv"]).exists()
    assert Path(outputs["markdown"]).exists()
    assert Path(outputs["html"]).exists()
    assert "adaptive" in Path(outputs["markdown"]).read_text(encoding="utf-8")
    html = Path(outputs["html"]).read_text(encoding="utf-8")
    assert "Adaptive Replanning Experiment Overview" in html
    assert "Goal evidence" in html
    assert "Environment Summary" in html
    assert "Method Comparison" in html
    assert "Success Delta" in html
    assert "Internal source details" in html
    assert "base_policy_checkpoint;normalization" in html
    assert "fixed_chunk:seed0;denoise_only:seed0" in html
    assert "<script" not in html.lower()
    assert "<link" not in html.lower()
    assert 'href="http' not in html.lower()
    with Path(outputs["csv"]).open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with Path(outputs["env_summary_csv"]).open("r", encoding="utf-8") as f:
        summaries = list(csv.DictReader(f))
    square = next(summary for summary in summaries if summary["env"] == "square")
    lift = next(summary for summary in summaries if summary["env"] == "lift")
    assert rows[-1]["label"] == "missing_final_suite"
    assert lift["best_high_level_return_label"] == "adaptive"
    assert lift["best_high_level_return_mean"] == "12.0"
    assert lift["lowest_wall_time_label"] == "fixed_chunk"
    assert lift["lowest_wall_time_per_action_mean"] == "0.1"
    assert rows[0]["mean_high_level_return_mean"] == "10.0"
    assert rows[0]["mean_high_level_return_sem"] == "1.0"
    assert rows[0]["success_delta_vs_fixed"] == "0.0"
    assert rows[0]["nfe_ratio_vs_fixed"] == "1.0"
    assert rows[1]["mean_total_wall_time_per_action_mean"] == "0.2"
    assert rows[1]["replan_rate_sem"] == "0.01"
    assert rows[1]["mean_discarded_actions_per_episode_sem"] == "1.0"
    assert rows[1]["return_delta_vs_fixed"] == "-1.0"
    assert rows[1]["high_level_return_delta_vs_fixed"] == "2.0"
    assert rows[1]["nfe_ratio_vs_fixed"] == "1.4"
    assert rows[1]["wall_time_ratio_vs_fixed"] == "2.0"
    assert rows[1]["learned_replan_rate_delta_vs_fixed"] == "0.1"
    assert rows[1]["discarded_actions_delta_vs_fixed"] == "20.0"
    assert square["readiness_missing"] == "base_policy_checkpoint;normalization"
    assert square["missing_rows"] == "fixed_chunk:seed0;denoise_only:seed0"




def test_write_report_html_escapes_dynamic_text(tmp_path):
    built = report.build_report(_audit(tmp_path))
    built["rows"][0]["label"] = "<script>alert(1)</script>"
    outputs = report.write_report(tmp_path / "html_escape", built)

    html = Path(outputs["html"]).read_text(encoding="utf-8")
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert "<script" not in html.lower()


def test_main_writes_html_snapshot(tmp_path, monkeypatch):
    audit = _audit(tmp_path)
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    snapshot = tmp_path / "stable" / "adaptive_replanning_report.html"
    output_root = tmp_path / "reports"
    monkeypatch.setattr(
        report.sys,
        "argv",
        [
            "build_final_report.py",
            "--audit_json",
            str(audit_path),
            "--output_dir",
            str(output_root),
            "--html_snapshot",
            str(snapshot),
        ],
    )

    report.main()

    generated = list(output_root.glob("*_final_report/final_report.html"))
    assert len(generated) == 1
    assert snapshot.exists()
    assert snapshot.read_text(encoding="utf-8") == generated[0].read_text(encoding="utf-8")


def test_format_mean_sem_handles_missing_values():
    assert report.format_mean_sem({}, "mean", "sem") == "n/a"
    assert report.format_mean_sem({"mean": "1.2345", "sem": "0.1"}, "mean", "sem") == "1.234 +/- 0.100"


def test_write_report_markdown_includes_blockers(tmp_path):
    built = report.build_report(_audit(tmp_path))
    outputs = report.write_report(tmp_path / "report", built)

    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    assert "Readiness Missing" in markdown
    assert "Best High-Level Return" in markdown
    assert "Lowest Wall Time/action" in markdown
    assert "High-Level Return" in markdown
    assert "Wall Time/action" in markdown
    assert "adaptive (12.000)" in markdown
    assert "fixed_chunk (0.100)" in markdown
    assert "12.000 +/- 0.500" in markdown
    assert "0.200 +/- 0.000" in markdown
    assert "base_policy_checkpoint, normalization" in markdown
    assert "fixed_chunk:seed0, denoise_only:seed0" in markdown
