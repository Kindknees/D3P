import csv
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_final_evidence.py"
spec = importlib.util.spec_from_file_location("audit_final_evidence", SCRIPT_PATH)
audit = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def _write_text(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_fake_suite(
    path: Path,
    *,
    env: str,
    labels=("fixed_chunk", "td_error_p95"),
    seeds=(0, 1),
    episodes=100,
    dry_run=False,
):
    path.mkdir(parents=True)
    plots = path / "plots"
    runs = path / "runs"
    rows = []
    timeline_paths = []
    for seed in seeds:
        for label in labels:
            run_dir = runs / f"{label}_seed{seed}"
            source = run_dir / "eval_summary.json"
            _write_text(
                source,
                json.dumps(
                    {
                        "env": env.replace("robomimic_", ""),
                        "method": label,
                        "seed": seed,
                        "episodes": episodes,
                        "success_rate": 0.5,
                        "mean_episode_return": 1.0,
                        "mean_nfe_per_action": 5.0,
                        "replan_rate": 0.25,
                    }
                ),
            )
            rows.append(
                {
                    "label": label,
                    "env": env,
                    "seed": str(seed),
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
            )
            timeline = plots / "timelines" / f"replan_timeline_{label}_seed{seed}.svg"
            _write_text(timeline, "<svg></svg>")
            timeline_paths.append(str(timeline))

    results = path / "suite_results.csv"
    results.parent.mkdir(parents=True, exist_ok=True)
    with results.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    outputs = {
        "summary_table_csv": str(_write_text(plots / "summary_table.csv", "a,b\n1,2\n")),
        "summary_table_md": str(_write_text(plots / "summary_table.md", "| a |\n")),
        "aggregate_summary_csv": str(_write_text(plots / "aggregate_summary.csv", "a,b\n1,2\n")),
        "aggregate_summary_md": str(_write_text(plots / "aggregate_summary.md", "| a |\n")),
        "pareto_plots": [
            str(_write_text(plots / "pareto_lift_success.svg", "<svg></svg>")),
            str(_write_text(plots / "pareto_lift_return.svg", "<svg></svg>")),
        ],
        "replan_timeline_plots": timeline_paths,
        "timeline_manifest": str(_write_text(plots / "timeline_manifest.csv", "path\n")),
    }
    summary = {
        "suite_dir": str(path),
        "dry_run": dry_run,
        "num_jobs": len(rows),
        "num_completed": 0 if dry_run else len(rows),
        "results_csv": str(results),
        "plots_dir": str(plots) if not dry_run else None,
        "analysis_summary": {} if dry_run else {"num_rows": len(rows), "num_aggregates": len(labels), "outputs": outputs},
    }
    _write_text(path / "suite_summary.json", json.dumps(summary))
    return path


def _args(tmp_path, **overrides):
    args = Namespace(
        envs=("robomimic_lift",),
        seeds=(0, 1),
        min_episodes=100,
        suite=(audit.SuiteSpec("robomimic_lift", tmp_path / "suite"),),
        expected_label=(
            audit.LabelSpec("robomimic_lift", "fixed_chunk"),
            audit.LabelSpec("robomimic_lift", "td_error_p95"),
        ),
        readiness_json=tmp_path / "readiness.json",
        experiment_markdown=tmp_path / "adaptive_replanning.md",
        markdown_token=["M5h", "Square evaluation is complete"],
        output_dir=tmp_path / "audit_out",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_default_markdown_tokens_require_current_denoise_and_gate_sections(tmp_path):
    markdown = tmp_path / "adaptive_replanning.md"
    markdown.write_text(
        "M5h Lift Full-Horizon 100-Episode/Seed Suite\n"
        "M5i Can Full-Horizon 100-Episode/Seed Suite\n"
        "M5m Readiness Artifact Integrity Guard\n"
        "M6b Denoise-Only Short Lift/Can Suite\n"
        "M6d Full-Horizon Denoise-Only Lift/Can Suite\n"
        "M6f Final Expected-Label Preset\n"
        "M6h Final-Full Default Gate\n"
        "M6i Final-Eval Default Baselines\n"
        "M6j Generated Final-Pipeline Close-Out Command\n"
        "M6k Env-Meta Forwarding In Generated Close-Out Commands\n"
        "M6m Pipeline Summary Blocker Details\n"
        "M6o Final Report Blocker Details\n"
        "M6p Duplicate Suite Path Guard\n"
        "M6q Final Environment Summary CSV\n"
        "M6r Complete Final Matrix Metrics\n"
        "M6s Env Summary High-Level And Wall-Time Winners\n"
        "M6t Fixed Baseline Comparison Columns\n"
        "M6u Final Pipeline Readiness Details\n"
        "M6v Final Close-Out Command With Known Suites\n"
        "M6w Final Close-Out Placeholder Manifest\n"
        "M6x Final Close-Out Command Resolver\n"
        "M6y Final Close-Out Path Validation\n"
        "M6z Square Close-Out Replacement Discovery\n"
        "M7a Square Close-Out Status Artifact\n"
        "M7b Square Runtime Artifact And Network Config\n"
        "M7c Resumable Final Suite Execution\n"
        "M7d Square Formal 3-Seed Final Suite\n"
        "Square evaluation is complete\n",
        encoding="utf-8",
    )

    result = audit.audit_markdown(markdown, audit.DEFAULT_MARKDOWN_TOKENS)

    assert result["ready"] is False
    assert result["missing_tokens"] == ["M7e Final Lift/Can/Square Evidence Pipeline"]


def test_build_audit_accepts_complete_suite_readiness_and_markdown(tmp_path):
    _write_fake_suite(tmp_path / "suite", env="robomimic_lift")
    _write_text(
        tmp_path / "readiness.json",
        json.dumps({"envs": [{"env": "robomimic_lift", "final_eval_ready": True}]}),
    )
    _write_text(tmp_path / "adaptive_replanning.md", "M5h\nSquare evaluation is complete\n")

    report = audit.build_audit(_args(tmp_path))

    assert report["all_goal_evidence_ready"] is True
    assert report["envs"][0]["suite_ready"] is True
    assert report["readiness"]["ready"] is True
    assert report["markdown"]["ready"] is True


def test_build_audit_rejects_duplicate_suite_paths(tmp_path):
    _write_fake_suite(tmp_path / "suite", env="robomimic_lift")
    _write_text(
        tmp_path / "readiness.json",
        json.dumps({"envs": [{"env": "robomimic_lift", "final_eval_ready": True}]}),
    )
    _write_text(tmp_path / "adaptive_replanning.md", "M5h\nSquare evaluation is complete\n")

    args = _args(
        tmp_path,
        suite=(
            audit.SuiteSpec("robomimic_lift", tmp_path / "suite"),
            audit.SuiteSpec("robomimic_lift", tmp_path / "suite"),
        ),
    )

    with pytest.raises(ValueError, match="duplicate suite path for robomimic_lift"):
        audit.build_audit(args)


def test_build_audit_merges_multiple_suites_for_same_env(tmp_path):
    _write_fake_suite(
        tmp_path / "base_suite",
        env="robomimic_lift",
        labels=("fixed_chunk", "td_error_p95"),
    )
    _write_fake_suite(
        tmp_path / "denoise_suite",
        env="robomimic_lift",
        labels=("denoise_only",),
    )
    _write_text(
        tmp_path / "readiness.json",
        json.dumps({"envs": [{"env": "robomimic_lift", "final_eval_ready": True}]}),
    )
    _write_text(tmp_path / "adaptive_replanning.md", "M5h\nSquare evaluation is complete\n")

    args = _args(
        tmp_path,
        suite=(
            audit.SuiteSpec("robomimic_lift", tmp_path / "base_suite"),
            audit.SuiteSpec("robomimic_lift", tmp_path / "denoise_suite"),
        ),
        expected_label=(
            audit.LabelSpec("robomimic_lift", "fixed_chunk"),
            audit.LabelSpec("robomimic_lift", "td_error_p95"),
            audit.LabelSpec("robomimic_lift", "denoise_only"),
        ),
    )

    report = audit.build_audit(args)
    env_report = report["envs"][0]
    checks = {check["label"]: check for check in env_report["checks"]}

    assert report["all_goal_evidence_ready"] is True
    assert env_report["suite_ready"] is True
    assert len(env_report["suite_paths"]) == 2
    assert len(env_report["rows"]) == 6
    assert env_report["missing_rows"] == []
    assert checks["expected_label_seed_rows"]["passed"] is True
    assert checks["env_rows_present"]["detail"] == 6


def test_build_audit_rejects_dry_run_and_short_eval(tmp_path):
    _write_fake_suite(tmp_path / "suite", env="robomimic_lift", episodes=2, dry_run=True)
    _write_text(
        tmp_path / "readiness.json",
        json.dumps({"envs": [{"env": "robomimic_lift", "final_eval_ready": True}]}),
    )
    _write_text(tmp_path / "adaptive_replanning.md", "M5h\nSquare evaluation is complete\n")

    report = audit.build_audit(_args(tmp_path))
    checks = {check["label"]: check for check in report["envs"][0]["checks"]}

    assert report["all_goal_evidence_ready"] is False
    assert checks["not_dry_run"]["passed"] is False
    assert checks["all_jobs_completed"]["passed"] is False
    assert checks["min_episodes"]["passed"] is False


def test_build_audit_reports_missing_square_suite_and_readiness(tmp_path):
    _write_text(
        tmp_path / "readiness.json",
        json.dumps({"envs": [{"env": "robomimic_square", "final_eval_ready": False}]}),
    )
    _write_text(tmp_path / "adaptive_replanning.md", "Square evaluation is complete\n")

    args = _args(
        tmp_path,
        envs=("robomimic_square",),
        suite=(),
        expected_label=(audit.LabelSpec("robomimic_square", "fixed_chunk"),),
        markdown_token=["Square evaluation is complete"],
    )
    report = audit.build_audit(args)

    assert report["all_goal_evidence_ready"] is False
    assert report["envs"][0]["suite_ready"] is False
    assert report["envs"][0]["checks"][0]["label"] == "suite_provided"
    assert report["readiness"]["ready"] is False
    assert report["readiness"]["missing"] == ["robomimic_square"]
