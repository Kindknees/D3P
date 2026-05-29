#!/usr/bin/env python3
"""Audit final adaptive-replanning evidence across readiness, suites, and notes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_final_readiness import check_artifact  # noqa: E402


DEFAULT_ENVS = ("robomimic_lift", "robomimic_can", "robomimic_square")
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_MARKDOWN_TOKENS = (
    "M5h Lift Full-Horizon 100-Episode/Seed Suite",
    "M5i Can Full-Horizon 100-Episode/Seed Suite",
    "M5m Readiness Artifact Integrity Guard",
    "M6b Denoise-Only Short Lift/Can Suite",
    "M6d Full-Horizon Denoise-Only Lift/Can Suite",
    "M6f Final Expected-Label Preset",
    "M6h Final-Full Default Gate",
    "M6i Final-Eval Default Baselines",
    "M6j Generated Final-Pipeline Close-Out Command",
    "M6k Env-Meta Forwarding In Generated Close-Out Commands",
    "M6m Pipeline Summary Blocker Details",
    "M6o Final Report Blocker Details",
    "M6p Duplicate Suite Path Guard",
    "M6q Final Environment Summary CSV",
    "M6r Complete Final Matrix Metrics",
    "M6s Env Summary High-Level And Wall-Time Winners",
    "M6t Fixed Baseline Comparison Columns",
    "M6u Final Pipeline Readiness Details",
    "M6v Final Close-Out Command With Known Suites",
    "M6w Final Close-Out Placeholder Manifest",
    "M6x Final Close-Out Command Resolver",
    "M6y Final Close-Out Path Validation",
    "M6z Square Close-Out Replacement Discovery",
    "M7a Square Close-Out Status Artifact",
    "M7b Square Runtime Artifact And Network Config",
    "M7c Resumable Final Suite Execution",
    "M7d Square Formal 3-Seed Final Suite",
    "M7e Final Lift/Can/Square Evidence Pipeline",
    "Square evaluation is complete",
)


@dataclass(frozen=True)
class SuiteSpec:
    env: str
    path: Path


@dataclass(frozen=True)
class LabelSpec:
    env: str
    label: str


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_int_csv(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one integer")
    return values


def parse_suite(value: str) -> SuiteSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("suite must use ENV=/path/to/final_eval_suite")
    env, path = value.split("=", 1)
    env = env.strip()
    if not env:
        raise argparse.ArgumentTypeError("suite env cannot be empty")
    return SuiteSpec(env=env, path=Path(path))


def parse_label(value: str) -> LabelSpec:
    if ":" not in value:
        raise argparse.ArgumentTypeError("expected label must use ENV:LABEL")
    env, label = value.split(":", 1)
    env = env.strip()
    label = label.strip()
    if not env or not label:
        raise argparse.ArgumentTypeError("expected label env and label cannot be empty")
    return LabelSpec(env=env, label=label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=parse_csv, default=DEFAULT_ENVS)
    parser.add_argument("--seeds", type=parse_int_csv, default=DEFAULT_SEEDS)
    parser.add_argument("--min_episodes", type=int, default=100)
    parser.add_argument("--suite", action="append", type=parse_suite, default=[])
    parser.add_argument("--expected_label", action="append", type=parse_label, default=[])
    parser.add_argument("--readiness_json", type=Path, default=None)
    parser.add_argument(
        "--experiment_markdown",
        type=Path,
        default=REPO_ROOT / "experiments" / "adaptive_replanning.md",
    )
    parser.add_argument(
        "--markdown_token",
        action="append",
        default=list(DEFAULT_MARKDOWN_TOKENS),
        help="Required text token in the experiment markdown. Can be repeated.",
    )
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs" / "audits")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def file_check(label: str, path: str | Path | None) -> dict[str, Any]:
    check = check_artifact(label, Path(path) if path is not None else None)
    return check.as_dict()


def env_suite_map(suites: Sequence[SuiteSpec]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    seen: set[tuple[str, Path]] = set()
    for suite in suites:
        normalized_path = suite.path.expanduser().resolve()
        key = (suite.env, normalized_path)
        if key in seen:
            raise ValueError(f"duplicate suite path for {suite.env}: {normalized_path}")
        seen.add(key)
        result.setdefault(suite.env, []).append(suite.path)
    return result


def expected_labels_for_env(labels: Sequence[LabelSpec], env: str) -> tuple[str, ...]:
    return tuple(label.label for label in labels if label.env == env)


def add_check(checks: list[dict[str, Any]], label: str, passed: bool, detail: Any = None) -> None:
    checks.append({"label": label, "passed": bool(passed), "detail": detail})


def missing_expected_rows(
    rows: Sequence[dict[str, str]],
    expected_labels: Sequence[str],
    expected_seeds: Sequence[int],
) -> list[str]:
    row_keys = {
        (row.get("label", ""), int(row.get("seed", -999)))
        for row in rows
        if row.get("seed", "").lstrip("-").isdigit()
    }
    return [
        f"{label}:seed{seed}"
        for label in expected_labels
        for seed in expected_seeds
        if (label, int(seed)) not in row_keys
    ]


def audit_source_eval_summaries(rows: Sequence[dict[str, str]], min_episodes: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_checks: list[dict[str, Any]] = []
    episode_failures: list[dict[str, Any]] = []
    for row in rows:
        source_path = row.get("source_path")
        source_check = file_check("source_eval_summary", source_path)
        source_checks.append(source_check)
        if not source_check["valid"]:
            episode_failures.append({"source_path": source_path, "reason": source_check["reason"]})
            continue
        source = read_json(Path(source_path))
        episodes = int(source.get("episodes") or 0)
        if episodes < min_episodes:
            episode_failures.append(
                {
                    "source_path": source_path,
                    "episodes": episodes,
                    "min_episodes": min_episodes,
                }
            )
    return source_checks, episode_failures


def audit_analysis_outputs(outputs: dict[str, Any], num_rows: int) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    output_checks: list[dict[str, Any]] = []
    for key in (
        "summary_table_csv",
        "summary_table_md",
        "aggregate_summary_csv",
        "aggregate_summary_md",
        "timeline_manifest",
    ):
        output_checks.append(file_check(key, outputs.get(key)))
    pareto_plots = list(outputs.get("pareto_plots") or [])
    timeline_plots = list(outputs.get("replan_timeline_plots") or [])
    for path in pareto_plots:
        output_checks.append(file_check("pareto_plot", path))
    for path in timeline_plots:
        output_checks.append(file_check("replan_timeline_plot", path))
    return output_checks, pareto_plots, timeline_plots


def _check_label(label: str, suite_index: int, num_suites: int) -> str:
    if num_suites == 1:
        return label
    return f"suite{suite_index}_{label}"


def _rows_for_env(rows: Sequence[dict[str, str]], env: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("env") == env]


def audit_suite(
    env: str,
    suite_paths: Sequence[Path] | None,
    expected_seeds: Sequence[int],
    expected_labels: Sequence[str],
    min_episodes: int,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    suite_paths = list(suite_paths or [])
    if not suite_paths:
        missing_rows = [f"{label}:seed{seed}" for label in expected_labels for seed in expected_seeds]
        add_check(checks, "suite_provided", False, "no suite path provided")
        return {
            "env": env,
            "suite_path": None,
            "suite_paths": [],
            "suite_ready": False,
            "checks": checks,
            "rows": [],
            "missing_rows": missing_rows,
            "outputs": [],
        }

    all_env_rows: list[dict[str, str]] = []
    all_outputs: list[dict[str, Any]] = []
    all_pareto_plots: list[str] = []
    all_timeline_plots: list[str] = []
    num_jobs_total = 0
    num_completed_total = 0
    num_aggregates_total = 0
    num_suites = len(suite_paths)

    for suite_index, raw_suite_path in enumerate(suite_paths):
        suite_path = raw_suite_path.resolve()
        suite_dir_valid = suite_path.exists() and suite_path.is_dir()
        suite_dir_check = {
            "label": "suite_dir",
            "path": str(suite_path),
            "exists": suite_path.exists(),
            "valid": suite_dir_valid,
            "reason": "ok" if suite_dir_valid else "missing_or_not_dir",
        }
        add_check(
            checks,
            _check_label("suite_dir", suite_index, num_suites),
            suite_dir_check["valid"],
            suite_dir_check,
        )
        all_outputs.append(suite_dir_check)
        summary_path = suite_path / "suite_summary.json"
        summary_check = file_check("suite_summary", summary_path)
        add_check(
            checks,
            _check_label("suite_summary", suite_index, num_suites),
            summary_check["valid"],
            summary_check,
        )
        all_outputs.append(summary_check)
        if not summary_check["valid"]:
            continue

        summary = read_json(summary_path)
        add_check(
            checks,
            _check_label("not_dry_run", suite_index, num_suites),
            not bool(summary.get("dry_run")),
            summary.get("dry_run"),
        )
        num_jobs = int(summary.get("num_jobs") or 0)
        num_completed = int(summary.get("num_completed") or 0)
        num_jobs_total += num_jobs
        num_completed_total += num_completed
        add_check(
            checks,
            _check_label("all_jobs_completed", suite_index, num_suites),
            num_jobs > 0 and num_completed == num_jobs,
            {"num_jobs": num_jobs, "num_completed": num_completed},
        )

        results_path = Path(summary.get("results_csv") or suite_path / "suite_results.csv")
        results_check = file_check("suite_results", results_path)
        add_check(
            checks,
            _check_label("suite_results", suite_index, num_suites),
            results_check["valid"],
            results_check,
        )
        all_outputs.append(results_check)
        rows = read_csv_rows(results_path)
        env_rows = _rows_for_env(rows, env)
        all_env_rows.extend(env_rows)
        add_check(
            checks,
            _check_label("result_rows_present", suite_index, num_suites),
            bool(env_rows),
            {"total_rows": len(rows), "env_rows": len(env_rows)},
        )

        source_checks, episode_failures = audit_source_eval_summaries(env_rows, min_episodes)
        add_check(
            checks,
            _check_label("source_eval_summaries", suite_index, num_suites),
            all(check["valid"] for check in source_checks),
            source_checks,
        )
        add_check(
            checks,
            _check_label("min_episodes", suite_index, num_suites),
            not episode_failures,
            episode_failures,
        )

        analysis = summary.get("analysis_summary") or {}
        outputs = analysis.get("outputs") or {}
        output_checks, pareto_plots, timeline_plots = audit_analysis_outputs(outputs, len(rows))
        add_check(
            checks,
            _check_label("analysis_outputs", suite_index, num_suites),
            bool(output_checks) and all(check["valid"] for check in output_checks),
            output_checks,
        )
        add_check(
            checks,
            _check_label("pareto_plots_present", suite_index, num_suites),
            len(pareto_plots) >= 2,
            pareto_plots,
        )
        add_check(
            checks,
            _check_label("timelines_present", suite_index, num_suites),
            len(timeline_plots) >= len(env_rows),
            {"timeline_plots": len(timeline_plots), "env_rows": len(env_rows)},
        )
        all_outputs.extend(output_checks)
        all_pareto_plots.extend(pareto_plots)
        all_timeline_plots.extend(timeline_plots)
        num_aggregates_total += int(analysis.get("num_aggregates") or 0)

    missing_rows = missing_expected_rows(all_env_rows, expected_labels, expected_seeds)
    if expected_labels:
        add_check(checks, "expected_label_seed_rows", not missing_rows, missing_rows)
    add_check(checks, "env_rows_present", bool(all_env_rows), len(all_env_rows))

    suite_ready = all(check["passed"] for check in checks)
    return {
        "env": env,
        "suite_path": str(suite_paths[0].resolve()) if suite_paths else None,
        "suite_paths": [str(path.resolve()) for path in suite_paths],
        "suite_ready": suite_ready,
        "checks": checks,
        "rows": all_env_rows,
        "missing_rows": missing_rows,
        "outputs": all_outputs,
        "summary": {
            "dry_run": False,
            "num_jobs": num_jobs_total,
            "num_completed": num_completed_total,
            "num_rows": len(all_env_rows),
            "num_aggregates": num_aggregates_total,
            "num_suites": len(suite_paths),
            "pareto_plots": len(all_pareto_plots),
            "timeline_plots": len(all_timeline_plots),
        },
    }


def audit_readiness(path: Path | None, envs: Sequence[str]) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "ready": False,
            "missing": list(envs),
            "envs": [],
            "reason": "not_provided",
        }
    path_check = file_check("readiness_json", path)
    if not path_check["valid"]:
        return {
            "path": str(path),
            "ready": False,
            "missing": list(envs),
            "envs": [],
            "reason": path_check["reason"],
            "check": path_check,
        }
    data = read_json(path)
    env_reports = {item.get("env"): item for item in data.get("envs", [])}
    missing = [env for env in envs if not env_reports.get(env, {}).get("final_eval_ready")]
    return {
        "path": str(path),
        "ready": not missing,
        "missing": missing,
        "envs": [env_reports.get(env, {"env": env, "final_eval_ready": False}) for env in envs],
        "reason": "ok" if not missing else "envs_not_ready",
    }


def audit_markdown(path: Path, tokens: Sequence[str]) -> dict[str, Any]:
    check = file_check("experiment_markdown", path)
    if not check["valid"]:
        return {"path": str(path), "ready": False, "missing_tokens": list(tokens), "check": check}
    text = path.read_text(encoding="utf-8")
    missing_tokens = [token for token in tokens if token not in text]
    return {
        "path": str(path),
        "ready": not missing_tokens,
        "missing_tokens": missing_tokens,
        "check": check,
    }


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    suite_paths = env_suite_map(args.suite)
    env_reports = []
    for env in args.envs:
        env_reports.append(
            audit_suite(
                env=env,
                suite_paths=suite_paths.get(env),
                expected_seeds=args.seeds,
                expected_labels=expected_labels_for_env(args.expected_label, env),
                min_episodes=args.min_episodes,
            )
        )
    readiness = audit_readiness(args.readiness_json, args.envs)
    markdown = audit_markdown(args.experiment_markdown, args.markdown_token)
    all_suites_ready = all(report["suite_ready"] for report in env_reports)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "envs": env_reports,
        "readiness": readiness,
        "markdown": markdown,
        "all_suites_ready": all_suites_ready,
        "all_goal_evidence_ready": all_suites_ready and readiness["ready"] and markdown["ready"],
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Adaptive Replanning Final Evidence Audit",
        "",
        f"Created: `{report['created_at']}`",
        "",
        f"All goal evidence ready: `{str(report['all_goal_evidence_ready']).lower()}`",
        f"All suites ready: `{str(report['all_suites_ready']).lower()}`",
        f"Readiness ready: `{str(report['readiness']['ready']).lower()}`",
        f"Markdown ready: `{str(report['markdown']['ready']).lower()}`",
        "",
        "| Env | Suite Ready | Suite | Missing Rows | Failed Checks |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for env in report["envs"]:
        failed = [check["label"] for check in env["checks"] if not check["passed"]]
        missing_rows = ", ".join(env.get("missing_rows") or []) or "none"
        failed_text = ", ".join(failed) or "none"
        suite_paths = env.get("suite_paths") or ([env.get("suite_path")] if env.get("suite_path") else [])
        suite_text = "<br>".join(str(path) for path in suite_paths) or "none"
        lines.append(
            f"| {env['env']} | {str(env['suite_ready']).lower()} | "
            f"{suite_text} | {missing_rows} | {failed_text} |"
        )
    lines.extend(["", "## Readiness", ""])
    lines.append(f"- Path: `{report['readiness'].get('path')}`")
    lines.append(f"- Ready: `{str(report['readiness']['ready']).lower()}`")
    lines.append(f"- Missing: `{', '.join(report['readiness'].get('missing') or []) or 'none'}`")
    lines.extend(["", "## Markdown", ""])
    lines.append(f"- Path: `{report['markdown'].get('path')}`")
    lines.append(f"- Ready: `{str(report['markdown']['ready']).lower()}`")
    lines.append(f"- Missing tokens: `{', '.join(report['markdown'].get('missing_tokens') or []) or 'none'}`")
    lines.extend(["", "## Suite Checks", ""])
    for env in report["envs"]:
        lines.extend([f"### {env['env']}", ""])
        for check in env["checks"]:
            lines.append(f"- `{check['label']}`: `{str(check['passed']).lower()}`")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.min_episodes <= 0:
        raise ValueError("--min_episodes must be positive")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"{timestamp}_final_evidence_audit"
    output_dir.mkdir(parents=True, exist_ok=False)
    report = build_audit(args)
    json_path = output_dir / "final_evidence_audit.json"
    markdown_path = output_dir / "final_evidence_audit.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_markdown(markdown_path, report)
    print(f"Wrote final evidence audit to {output_dir}")
    print(OmegaConf.to_yaml(OmegaConf.create({
        "output_dir": str(output_dir),
        "json": str(json_path),
        "markdown": str(markdown_path),
        "all_goal_evidence_ready": report["all_goal_evidence_ready"],
        "all_suites_ready": report["all_suites_ready"],
    }), resolve=True))


if __name__ == "__main__":
    main()
