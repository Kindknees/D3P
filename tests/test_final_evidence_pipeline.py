import csv
import json
import shlex
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from scripts import audit_final_evidence as audit
from scripts import run_final_evidence_pipeline as pipeline


def _write_text(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_suite(path: Path, env: str, label: str = "fixed_chunk") -> Path:
    plots = path / "plots"
    run_dir = path / "runs" / f"{label}_seed0"
    source = _write_text(
        run_dir / "eval_summary.json",
        json.dumps({"env": env.replace("robomimic_", ""), "episodes": 100}),
    )
    results_row = {
        "label": label,
        "env": env,
        "seed": "0",
        "kind": "baseline",
        "run_dir": str(run_dir),
        "success_rate": "0.5",
        "mean_episode_return": "1.0",
        "mean_high_level_return": "1.0",
        "mean_nfe_per_action": "5.0",
        "replan_rate": "0.25",
        "learned_replan_rate": "0.0",
        "mean_discarded_actions_per_episode": "0.0",
        "source_path": str(source),
    }
    results = path / "suite_results.csv"
    results.parent.mkdir(parents=True, exist_ok=True)
    with results.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_row.keys()))
        writer.writeheader()
        writer.writerow(results_row)

    aggregate_row = {
        "label": label,
        "env": env.replace("robomimic_", ""),
        "method": label,
        "num_runs": "1",
        "seeds": "0",
        "episodes_total": "100",
        "success_rate_mean": "0.5",
        "success_rate_sem": "0.0",
        "mean_episode_return_mean": "1.0",
        "mean_episode_return_sem": "0.0",
        "mean_high_level_return_mean": "1.0",
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
    }
    aggregate = plots / "aggregate_summary.csv"
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    with aggregate.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_row.keys()))
        writer.writeheader()
        writer.writerow(aggregate_row)

    outputs = {
        "summary_table_csv": str(_write_text(plots / "summary_table.csv", "a,b\n1,2\n")),
        "summary_table_md": str(_write_text(plots / "summary_table.md", "| a |\n")),
        "aggregate_summary_csv": str(aggregate),
        "aggregate_summary_md": str(_write_text(plots / "aggregate_summary.md", "| a |\n")),
        "pareto_plots": [
            str(_write_text(plots / "pareto_success.svg", "<svg></svg>")),
            str(_write_text(plots / "pareto_return.svg", "<svg></svg>")),
        ],
        "replan_timeline_plots": [
            str(_write_text(plots / "timelines" / "replan_timeline_fixed_ep0.svg", "<svg></svg>")),
        ],
        "timeline_manifest": str(_write_text(plots / "timeline_manifest.csv", "path\n")),
    }
    summary = {
        "dry_run": False,
        "num_jobs": 1,
        "num_completed": 1,
        "results_csv": str(results),
        "analysis_summary": {"outputs": outputs, "num_rows": 1, "num_aggregates": 1},
    }
    _write_text(path / "suite_summary.json", json.dumps(summary))
    return path


def _args(tmp_path, *, include_square=False):
    suite = _write_suite(tmp_path / "suite", "robomimic_lift")
    readiness_envs = [{"env": "robomimic_lift", "final_eval_ready": True}]
    envs = ("robomimic_lift",)
    expected = (audit.LabelSpec("robomimic_lift", "fixed_chunk"),)
    if include_square:
        readiness_envs.append({"env": "robomimic_square", "final_eval_ready": False})
        envs = ("robomimic_lift", "robomimic_square")
        expected = (*expected, audit.LabelSpec("robomimic_square", "fixed_chunk"))
    readiness = _write_text(tmp_path / "readiness.json", json.dumps({"envs": readiness_envs}))
    markdown = _write_text(tmp_path / "adaptive_replanning.md", "token\n")
    return Namespace(
        envs=envs,
        seeds=(0,),
        min_episodes=100,
        suite=(audit.SuiteSpec("robomimic_lift", suite),),
        expected_label=expected,
        expected_label_preset="none",
        readiness_json=readiness,
        generate_readiness=False,
        readiness_episodes=100,
        td_bootstrap_episodes=3,
        td_bootstrap_max_episode_steps=None,
        td_artifact=(),
        ppo_checkpoint=(),
        runtime_artifact=(),
        env_meta_path=None,
        square_work_root=tmp_path / "square_work",
        python_executable="python",
        experiment_markdown=markdown,
        markdown_token=["token"],
        output_dir=tmp_path / "pipeline_out",
        require_complete=False,
    )


def test_run_pipeline_writes_audit_report_and_summary(tmp_path):
    summary = pipeline.run_pipeline(_args(tmp_path))

    assert summary["all_goal_evidence_ready"] is True
    assert Path(summary["audit_json"]).exists()
    assert Path(summary["final_report_markdown"]).exists()
    assert Path(summary["final_report_html"]).exists()
    assert Path(summary["final_env_summary_csv"]).exists()
    assert Path(summary["pipeline_summary_json"]).exists()
    with Path(summary["final_results_csv"]).open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with Path(summary["final_env_summary_csv"]).open("r", encoding="utf-8") as f:
        summaries = list(csv.DictReader(f))
    assert rows[0]["label"] == "fixed_chunk"
    assert summaries[0]["env"] == "lift"
    summary_markdown = Path(summary["pipeline_summary_markdown"]).read_text(encoding="utf-8")
    assert "Final Report HTML" in summary_markdown


def test_run_pipeline_keeps_missing_square_visible(tmp_path):
    summary = pipeline.run_pipeline(_args(tmp_path, include_square=True))

    assert summary["all_goal_evidence_ready"] is False
    square = next(env for env in summary["envs"] if env["env"] == "robomimic_square")
    assert square["failed_checks"] == ["suite_provided"]
    assert square["missing_rows"] == ["fixed_chunk:seed0"]
    assert square["readiness_missing"] == ["final_eval_ready"]
    summary_markdown = Path(summary["pipeline_summary_markdown"]).read_text(encoding="utf-8")
    assert "Readiness Missing" in summary_markdown
    assert "fixed_chunk:seed0" in summary_markdown
    with Path(summary["final_env_summary_csv"]).open("r", encoding="utf-8") as f:
        summaries = list(csv.DictReader(f))
    square_summary = next(row for row in summaries if row["env"] == "square")
    assert square_summary["missing_rows"] == "fixed_chunk:seed0"
    assert "missing_final_suite" in Path(summary["final_report_markdown"]).read_text(encoding="utf-8")


def test_run_pipeline_generated_readiness_keeps_square_checks_and_commands(tmp_path):
    args = _args(tmp_path, include_square=True)
    extra_suite = _write_suite(tmp_path / "denoise_suite", "robomimic_lift", label="denoise_only")
    args.suite = (*args.suite, audit.SuiteSpec("robomimic_lift", extra_suite))
    args.readiness_json = None
    args.generate_readiness = True
    args.runtime_artifact = (
        pipeline.readiness.RuntimeArtifactSpec(
            "robomimic_square",
            tmp_path / "missing_square_policy.pt",
            tmp_path / "missing_square_normalization.npz",
        ),
    )

    summary = pipeline.run_pipeline(args)

    square = next(env for env in summary["envs"] if env["env"] == "robomimic_square")
    checks = {check["label"]: check for check in square["readiness_checks"]}
    assert checks["base_policy_checkpoint"]["reason"] == "missing"
    assert checks["normalization"]["reason"] == "missing"
    assert "collect_fixed_transitions" in square["next_commands"]
    assert "run_final_pipeline" in square["next_commands"]
    closeout_command = summary["final_closeout_command"]
    assert f"robomimic_lift={args.suite[0].path}" in closeout_command
    assert f"robomimic_lift={extra_suite}" in closeout_command
    assert "robomimic_lift=<" not in closeout_command
    assert "robomimic_square=" in closeout_command
    assert "<square_final_eval_suite_dir>" in closeout_command
    assert "--require_complete" in closeout_command
    closeout_parts = shlex.split(closeout_command)
    td_values = [
        closeout_parts[index + 1]
        for index, part in enumerate(closeout_parts[:-1])
        if part == "--td_artifact"
    ]
    ppo_values = [
        closeout_parts[index + 1]
        for index, part in enumerate(closeout_parts[:-1])
        if part == "--ppo_checkpoint"
    ]
    assert any(value.startswith("robomimic_square:") for value in td_values)
    assert "robomimic_square:square_bc_p90_reg02=<square_ppo_checkpoint.pt>" in ppo_values
    placeholders = summary["final_closeout_placeholders"]
    placeholder_tokens = {placeholder["token"] for placeholder in placeholders}
    assert summary["final_closeout_ready"] is False
    assert "<square_final_eval_suite_dir>" in placeholder_tokens
    assert "<fixed_run_dir>" in placeholder_tokens
    assert "<square_ppo_checkpoint.pt>" in placeholder_tokens
    assert not any(placeholder["env"] == "robomimic_lift" for placeholder in placeholders)
    square_suite_placeholder = next(
        placeholder for placeholder in placeholders if placeholder["token"] == "<square_final_eval_suite_dir>"
    )
    assert square_suite_placeholder["kind"] == "final_suite"
    assert square_suite_placeholder["argument"] == "--suite"
    summary_markdown = Path(summary["pipeline_summary_markdown"]).read_text(encoding="utf-8")
    assert "Final Close-Out Command" in summary_markdown
    assert "Close-Out Placeholders" in summary_markdown
    assert "<square_final_eval_suite_dir>" in summary_markdown
    assert closeout_command in summary_markdown
    assert "Readiness Artifact Details" in summary_markdown
    assert "base_policy_checkpoint" in summary_markdown
    assert "Suggested Square sequence" in summary_markdown
    assert "scripts/run_final_evidence_pipeline.py" in summary_markdown


def test_parse_args_defaults_to_final_full_expected_label_preset(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_final_evidence_pipeline.py", "--readiness_json", "readiness.json"],
    )

    args = pipeline.parse_args()

    assert args.expected_label_preset == "final_full"


def test_expected_label_preset_requires_final_denoise_rows(tmp_path):
    args = _args(tmp_path)
    args.expected_label = ()
    args.expected_label_preset = "final_full"

    summary = pipeline.run_pipeline(args)

    assert summary["all_goal_evidence_ready"] is False
    audit_report = json.loads(Path(summary["audit_json"]).read_text(encoding="utf-8"))
    lift = next(env for env in audit_report["envs"] if env["env"] == "robomimic_lift")
    checks = {check["label"]: check for check in lift["checks"]}
    expected = checks["expected_label_seed_rows"]
    assert expected["passed"] is False
    assert "denoise_only:seed0" in expected["detail"]
    assert "lift_bc_p90_reg02:seed0" in expected["detail"]


def test_require_complete_cli_exits_nonzero_but_writes_outputs(tmp_path):
    args = _args(tmp_path, include_square=True)
    command = [
        sys.executable,
        "scripts/run_final_evidence_pipeline.py",
        "--envs",
        "robomimic_lift,robomimic_square",
        "--seeds",
        "0",
        "--suite",
        f"robomimic_lift={args.suite[0].path}",
        "--expected_label",
        "robomimic_lift:fixed_chunk",
        "--expected_label",
        "robomimic_square:fixed_chunk",
        "--readiness_json",
        str(args.readiness_json),
        "--experiment_markdown",
        str(args.experiment_markdown),
        "--markdown_token",
        "token",
        "--output_dir",
        str(tmp_path / "cli_out"),
        "--require_complete",
    ]

    result = subprocess.run(command, check=False, capture_output=True, text=True)

    assert result.returncode == 1
    assert "all_goal_evidence_ready: false" in result.stdout
    summaries = sorted((tmp_path / "cli_out").glob("*/pipeline_summary.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["require_complete"] is True
    assert summary["all_goal_evidence_ready"] is False


def test_run_pipeline_can_generate_readiness_json(tmp_path):
    args = _args(tmp_path)
    policy = _write_text(tmp_path / "policy.pt")
    normalization = _write_text(tmp_path / "normalization.npz")
    env_meta = _write_text(tmp_path / "env_meta.json")
    td_critic = _write_text(tmp_path / "td.pt")
    calibration = _write_text(tmp_path / "calibration.json")
    ppo = _write_text(tmp_path / "ppo.pt")
    args.readiness_json = None
    args.generate_readiness = True
    args.runtime_artifact = (
        pipeline.readiness.RuntimeArtifactSpec("robomimic_lift", policy, normalization),
    )
    args.env_meta_path = env_meta
    args.td_artifact = (
        pipeline.readiness.TDArtifactSpec("robomimic_lift", td_critic, calibration),
    )
    args.ppo_checkpoint = (
        pipeline.readiness.PPOCheckpointSpec("ppo", ppo, "robomimic_lift"),
    )

    summary = pipeline.run_pipeline(args)

    assert summary["generated_readiness"] is True
    assert summary["all_goal_evidence_ready"] is True
    assert summary["final_closeout_ready"] is True
    assert summary["final_closeout_placeholders"] == []
    assert Path(summary["readiness_json"]).exists()
    assert Path(summary["readiness_markdown"]).exists()
