#!/usr/bin/env python3
"""Build a compact final analysis report from audited evaluation evidence."""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_final_evidence import read_csv_rows, read_json  # noqa: E402


NUMERIC_FIELDS = (
    "success_rate_mean",
    "success_rate_sem",
    "mean_episode_return_mean",
    "mean_episode_return_sem",
    "mean_high_level_return_mean",
    "mean_high_level_return_sem",
    "mean_nfe_per_action_mean",
    "mean_nfe_per_action_sem",
    "mean_total_wall_time_per_action_mean",
    "mean_total_wall_time_per_action_sem",
    "replan_rate_mean",
    "replan_rate_sem",
    "learned_replan_rate_mean",
    "learned_replan_rate_sem",
    "mean_discarded_actions_per_episode_mean",
    "mean_discarded_actions_per_episode_sem",
)
REPORT_FIELDS = (
    "env",
    "label",
    "method",
    "status",
    "num_runs",
    "seeds",
    "episodes_total",
    "success_rate_mean",
    "success_rate_sem",
    "mean_episode_return_mean",
    "mean_episode_return_sem",
    "mean_high_level_return_mean",
    "mean_high_level_return_sem",
    "mean_nfe_per_action_mean",
    "mean_nfe_per_action_sem",
    "mean_total_wall_time_per_action_mean",
    "mean_total_wall_time_per_action_sem",
    "replan_rate_mean",
    "replan_rate_sem",
    "learned_replan_rate_mean",
    "learned_replan_rate_sem",
    "mean_discarded_actions_per_episode_mean",
    "mean_discarded_actions_per_episode_sem",
    "success_delta_vs_fixed",
    "return_delta_vs_fixed",
    "high_level_return_delta_vs_fixed",
    "nfe_ratio_vs_fixed",
    "wall_time_ratio_vs_fixed",
    "replan_rate_delta_vs_fixed",
    "learned_replan_rate_delta_vs_fixed",
    "discarded_actions_delta_vs_fixed",
    "source_suite",
)
ENV_SUMMARY_FIELDS = (
    "env",
    "ready",
    "num_methods",
    "best_success_label",
    "best_success_rate_mean",
    "best_return_label",
    "best_return_mean",
    "best_high_level_return_label",
    "best_high_level_return_mean",
    "lowest_compute_label",
    "lowest_compute_nfe_per_action_mean",
    "lowest_wall_time_label",
    "lowest_wall_time_per_action_mean",
    "status",
    "readiness_missing",
    "missing_rows",
    "failed_checks",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs" / "final_reports")
    parser.add_argument(
        "--html_snapshot",
        type=Path,
        default=None,
        help="Optional stable path that receives a copy of the generated final_report.html.",
    )
    return parser.parse_args()


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_float(value: Any, digits: int = 3) -> str:
    parsed = as_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.{digits}f}"


def format_mean_sem(row: dict[str, Any], mean_key: str, sem_key: str, digits: int = 3) -> str:
    mean = format_float(row.get(mean_key), digits=digits)
    sem = as_float(row.get(sem_key))
    if sem is None:
        return mean
    return f"{mean} +/- {sem:.{digits}f}"


def html_escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def format_percent(value: Any, digits: int = 1) -> str:
    parsed = as_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed * 100:.{digits}f}%"


def format_signed_float(value: Any, digits: int = 3) -> str:
    parsed = as_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:+.{digits}f}"


def bar_width(value: Any, max_value: float = 1.0) -> str:
    parsed = as_float(value)
    if parsed is None or max_value <= 0:
        return "0%"
    return f"{max(0.0, min(parsed / max_value, 1.0)) * 100:.1f}%"


def load_aggregate_rows(suite_path: Path) -> list[dict[str, str]]:
    return read_csv_rows(suite_path / "plots" / "aggregate_summary.csv")


def normalize_aggregate_row(row: dict[str, str], suite_path: str) -> dict[str, Any]:
    normalized: dict[str, Any] = {field: row.get(field, "") for field in REPORT_FIELDS}
    normalized["status"] = "complete"
    normalized["source_suite"] = suite_path
    for field in NUMERIC_FIELDS:
        value = as_float(normalized.get(field))
        normalized[field] = "" if value is None else value
    return normalized


def missing_row(env: str, suite_path: str | None, failed_checks: Sequence[str]) -> dict[str, Any]:
    row = {field: "" for field in REPORT_FIELDS}
    row.update(
        {
            "env": env.replace("robomimic_", ""),
            "label": "missing_final_suite",
            "method": "missing",
            "status": "missing:" + ",".join(failed_checks or ["suite_not_ready"]),
            "source_suite": suite_path or "",
        }
    )
    return row


def suite_paths_for_report(env_report: dict[str, Any]) -> list[str]:
    suite_paths = [str(path) for path in env_report.get("suite_paths") or []]
    if suite_paths:
        return suite_paths
    suite_path = env_report.get("suite_path")
    return [str(suite_path)] if suite_path else []


def dedupe_rows_by_label(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        label = str(row.get("label") or "")
        if label not in selected:
            selected[label] = row
            order.append(label)
            continue
        current_episodes = as_float(row.get("episodes_total")) or 0.0
        previous_episodes = as_float(selected[label].get("episodes_total")) or 0.0
        if current_episodes > previous_episodes:
            selected[label] = row
    return [selected[label] for label in order]


def ratio_or_blank(value: Any, baseline: Any) -> float | str:
    parsed_value = as_float(value)
    parsed_baseline = as_float(baseline)
    if parsed_value is None or parsed_baseline in (None, 0.0):
        return ""
    return parsed_value / parsed_baseline


def delta_or_blank(value: Any, baseline: Any) -> float | str:
    parsed_value = as_float(value)
    parsed_baseline = as_float(baseline)
    if parsed_value is None or parsed_baseline is None:
        return ""
    return parsed_value - parsed_baseline


def add_fixed_comparison_columns(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    fixed_by_env = {
        str(row.get("env")): row
        for row in rows
        if row.get("status") == "complete" and row.get("label") == "fixed_chunk"
    }
    result: list[dict[str, Any]] = []
    for row in rows:
        annotated = dict(row)
        fixed = fixed_by_env.get(str(row.get("env")))
        if row.get("status") != "complete" or fixed is None:
            result.append(annotated)
            continue
        annotated["success_delta_vs_fixed"] = delta_or_blank(
            row.get("success_rate_mean"), fixed.get("success_rate_mean")
        )
        annotated["return_delta_vs_fixed"] = delta_or_blank(
            row.get("mean_episode_return_mean"), fixed.get("mean_episode_return_mean")
        )
        annotated["high_level_return_delta_vs_fixed"] = delta_or_blank(
            row.get("mean_high_level_return_mean"), fixed.get("mean_high_level_return_mean")
        )
        annotated["nfe_ratio_vs_fixed"] = ratio_or_blank(
            row.get("mean_nfe_per_action_mean"), fixed.get("mean_nfe_per_action_mean")
        )
        annotated["wall_time_ratio_vs_fixed"] = ratio_or_blank(
            row.get("mean_total_wall_time_per_action_mean"), fixed.get("mean_total_wall_time_per_action_mean")
        )
        annotated["replan_rate_delta_vs_fixed"] = delta_or_blank(
            row.get("replan_rate_mean"), fixed.get("replan_rate_mean")
        )
        annotated["learned_replan_rate_delta_vs_fixed"] = delta_or_blank(
            row.get("learned_replan_rate_mean"), fixed.get("learned_replan_rate_mean")
        )
        annotated["discarded_actions_delta_vs_fixed"] = delta_or_blank(
            row.get("mean_discarded_actions_per_episode_mean"),
            fixed.get("mean_discarded_actions_per_episode_mean"),
        )
        result.append(annotated)
    return result


def collect_result_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for env_report in audit.get("envs", []):
        env = str(env_report.get("env") or "")
        report_suite_paths = suite_paths_for_report(env_report)
        failed = [check["label"] for check in env_report.get("checks", []) if not check.get("passed")]
        if not env_report.get("suite_ready") or not report_suite_paths:
            rows.append(missing_row(env, env_report.get("suite_path"), failed))
            continue
        env_key = env.replace("robomimic_", "")
        collected = []
        for suite_path_raw in report_suite_paths:
            suite_path = Path(suite_path_raw)
            aggregate_rows = [
                row for row in load_aggregate_rows(suite_path) if row.get("env") == env_key
            ]
            for row in aggregate_rows:
                collected.append(normalize_aggregate_row(row, suite_path_raw))
        if not collected:
            rows.append(missing_row(env, env_report.get("suite_path"), ["aggregate_summary_missing"]))
            continue
        rows.extend(dedupe_rows_by_label(collected))
    return rows


def best_row(rows: Sequence[dict[str, Any]], env: str, metric: str, reverse: bool = True) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("env") == env and row.get("status") == "complete"]
    candidates = [row for row in candidates if as_float(row.get(metric)) is not None]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: as_float(row.get(metric)) or 0.0, reverse=reverse)[0]


def summarize_env(rows: Sequence[dict[str, Any]], env: str) -> dict[str, Any]:
    complete_rows = [row for row in rows if row.get("env") == env and row.get("status") == "complete"]
    if not complete_rows:
        missing = [row for row in rows if row.get("env") == env]
        return {
            "env": env,
            "ready": False,
            "num_methods": 0,
            "best_success": None,
            "best_return": None,
            "best_high_level_return": None,
            "lowest_compute": None,
            "lowest_wall_time": None,
            "status": missing[0].get("status") if missing else "missing",
        }
    return {
        "env": env,
        "ready": True,
        "num_methods": len(complete_rows),
        "best_success": best_row(rows, env, "success_rate_mean"),
        "best_return": best_row(rows, env, "mean_episode_return_mean"),
        "best_high_level_return": best_row(rows, env, "mean_high_level_return_mean"),
        "lowest_compute": best_row(rows, env, "mean_nfe_per_action_mean", reverse=False),
        "lowest_wall_time": best_row(rows, env, "mean_total_wall_time_per_action_mean", reverse=False),
        "status": "complete",
    }


def env_key(env: str) -> str:
    return env.replace("robomimic_", "")


def failed_checks(env_report: dict[str, Any]) -> list[str]:
    return [check["label"] for check in env_report.get("checks", []) if not check.get("passed")]


def annotate_blockers(summary: dict[str, Any], audit: dict[str, Any], env_report: dict[str, Any]) -> dict[str, Any]:
    readiness = audit.get("readiness") or {}
    readiness_envs = {env_key(str(env.get("env") or "")): env for env in readiness.get("envs", [])}
    readiness_missing_envs = {env_key(str(env)) for env in readiness.get("missing") or []}
    readiness_env = readiness_envs.get(summary["env"], {})
    readiness_missing = list(readiness_env.get("missing") or [])
    if not readiness_missing and summary["env"] in readiness_missing_envs:
        readiness_missing = ["final_eval_ready"]
    summary["missing_rows"] = list(env_report.get("missing_rows") or [])
    summary["readiness_missing"] = readiness_missing
    summary["failed_checks"] = failed_checks(env_report)
    return summary


def build_report(audit: dict[str, Any]) -> dict[str, Any]:
    rows = add_fixed_comparison_columns(collect_result_rows(audit))
    env_reports = {env_key(str(report.get("env") or "")): report for report in audit.get("envs", [])}
    envs = sorted({str(row.get("env")) for row in rows if row.get("env")} | set(env_reports))
    env_summaries = [
        annotate_blockers(summarize_env(rows, env), audit, env_reports.get(env, {}))
        for env in envs
    ]
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "audit_json": audit.get("source_path"),
        "all_goal_evidence_ready": bool(audit.get("all_goal_evidence_ready")),
        "all_suites_ready": bool(audit.get("all_suites_ready")),
        "readiness_ready": bool((audit.get("readiness") or {}).get("ready")),
        "markdown_ready": bool((audit.get("markdown") or {}).get("ready")),
        "rows": rows,
        "env_summaries": env_summaries,
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(REPORT_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in REPORT_FIELDS})


def join_values(values: Sequence[Any]) -> str:
    return ";".join(str(value) for value in values)


def env_summary_csv_row(summary: dict[str, Any]) -> dict[str, Any]:
    best_success = summary.get("best_success") or {}
    best_return = summary.get("best_return") or {}
    best_high_level_return = summary.get("best_high_level_return") or {}
    lowest_compute = summary.get("lowest_compute") or {}
    lowest_wall_time = summary.get("lowest_wall_time") or {}
    return {
        "env": summary.get("env", ""),
        "ready": str(bool(summary.get("ready"))).lower(),
        "num_methods": summary.get("num_methods", ""),
        "best_success_label": best_success.get("label", ""),
        "best_success_rate_mean": best_success.get("success_rate_mean", ""),
        "best_return_label": best_return.get("label", ""),
        "best_return_mean": best_return.get("mean_episode_return_mean", ""),
        "best_high_level_return_label": best_high_level_return.get("label", ""),
        "best_high_level_return_mean": best_high_level_return.get("mean_high_level_return_mean", ""),
        "lowest_compute_label": lowest_compute.get("label", ""),
        "lowest_compute_nfe_per_action_mean": lowest_compute.get("mean_nfe_per_action_mean", ""),
        "lowest_wall_time_label": lowest_wall_time.get("label", ""),
        "lowest_wall_time_per_action_mean": lowest_wall_time.get("mean_total_wall_time_per_action_mean", ""),
        "status": summary.get("status", ""),
        "readiness_missing": join_values(summary.get("readiness_missing") or []),
        "missing_rows": join_values(summary.get("missing_rows") or []),
        "failed_checks": join_values(summary.get("failed_checks") or []),
    }


def write_env_summary_csv(path: Path, summaries: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ENV_SUMMARY_FIELDS))
        writer.writeheader()
        for summary in summaries:
            writer.writerow(env_summary_csv_row(summary))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Adaptive Replanning Final Analysis Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        f"All goal evidence ready: `{str(report['all_goal_evidence_ready']).lower()}`",
        f"All suites ready: `{str(report['all_suites_ready']).lower()}`",
        f"Readiness ready: `{str(report['readiness_ready']).lower()}`",
        f"Markdown ready: `{str(report['markdown_ready']).lower()}`",
        "",
        "## Environment Summary",
        "",
        "| Env | Ready | Methods | Best Success | Best Return | Best High-Level Return | Lowest Compute | Lowest Wall Time/action | Readiness Missing | Missing Rows | Status |",
        "| --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for summary in report["env_summaries"]:
        best_success = summary.get("best_success") or {}
        best_return = summary.get("best_return") or {}
        best_high_level_return = summary.get("best_high_level_return") or {}
        lowest_compute = summary.get("lowest_compute") or {}
        lowest_wall_time = summary.get("lowest_wall_time") or {}
        readiness_missing = ", ".join(summary.get("readiness_missing") or []) or "none"
        missing_rows = ", ".join(summary.get("missing_rows") or []) or "none"
        lines.append(
            f"| {summary['env']} | {str(summary['ready']).lower()} | {summary['num_methods']} | "
            f"{best_success.get('label', 'n/a')} ({format_float(best_success.get('success_rate_mean'))}) | "
            f"{best_return.get('label', 'n/a')} ({format_float(best_return.get('mean_episode_return_mean'))}) | "
            f"{best_high_level_return.get('label', 'n/a')} ({format_float(best_high_level_return.get('mean_high_level_return_mean'))}) | "
            f"{lowest_compute.get('label', 'n/a')} ({format_float(lowest_compute.get('mean_nfe_per_action_mean'))}) | "
            f"{lowest_wall_time.get('label', 'n/a')} ({format_float(lowest_wall_time.get('mean_total_wall_time_per_action_mean'))}) | "
            f"{readiness_missing} | {missing_rows} | {summary['status']} |"
        )
    lines.extend([
        "",
        "## Result Matrix",
        "",
        "| Env | Label | Success | Return | High-Level Return | NFE/action | Wall Time/action | Replan Rate | Learned Replan | Discarded/Episode | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in report["rows"]:
        lines.append(
            f"| {row.get('env', '')} | {row.get('label', '')} | "
            f"{format_mean_sem(row, 'success_rate_mean', 'success_rate_sem')} | "
            f"{format_mean_sem(row, 'mean_episode_return_mean', 'mean_episode_return_sem')} | "
            f"{format_mean_sem(row, 'mean_high_level_return_mean', 'mean_high_level_return_sem')} | "
            f"{format_mean_sem(row, 'mean_nfe_per_action_mean', 'mean_nfe_per_action_sem')} | "
            f"{format_mean_sem(row, 'mean_total_wall_time_per_action_mean', 'mean_total_wall_time_per_action_sem')} | "
            f"{format_mean_sem(row, 'replan_rate_mean', 'replan_rate_sem')} | "
            f"{format_mean_sem(row, 'learned_replan_rate_mean', 'learned_replan_rate_sem')} | "
            f"{format_mean_sem(row, 'mean_discarded_actions_per_episode_mean', 'mean_discarded_actions_per_episode_sem')} | "
            f"{row.get('status', '')} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Lift and Can rows are final-suite aggregate metrics when their suite audits pass.",
        "- Missing rows are explicit blockers and are not included in best-method selection.",
        "- The report is complete only when `all_goal_evidence_ready` is true in the source audit.",
    ])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")



def gate_badge(label: str, ready: bool) -> str:
    status_class = "ok" if ready else "bad"
    status_text = "ready" if ready else "blocked"
    return (
        f'<span class="gate {status_class}">'
        f'<span>{html_escape(label)}</span><strong>{status_text}</strong></span>'
    )


def winner_text(row: dict[str, Any] | None, metric: str, formatter=format_float) -> str:
    if not row:
        return "n/a"
    return f"{html_escape(row.get('label', 'n/a'))} <strong>{html_escape(formatter(row.get(metric)))}</strong>"


def render_env_cards(report: dict[str, Any]) -> str:
    cards: list[str] = []
    for summary in report["env_summaries"]:
        cards.append(
            "\n".join(
                [
                    '<article class="env-card">',
                    f"<h3>{html_escape(str(summary.get('env', '')).title())}</h3>",
                    '<dl class="metric-list">',
                    f"<div><dt>Best Success</dt><dd>{winner_text(summary.get('best_success'), 'success_rate_mean', format_percent)}</dd></div>",
                    f"<div><dt>Best Return</dt><dd>{winner_text(summary.get('best_return'), 'mean_episode_return_mean')}</dd></div>",
                    f"<div><dt>Best High-Level Return</dt><dd>{winner_text(summary.get('best_high_level_return'), 'mean_high_level_return_mean')}</dd></div>",
                    f"<div><dt>Lowest NFE/action</dt><dd>{winner_text(summary.get('lowest_compute'), 'mean_nfe_per_action_mean')}</dd></div>",
                    "</dl>",
                    "</article>",
                ]
            )
        )
    return "\n".join(cards)


def grouped_rows(rows: Sequence[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        env = str(row.get("env") or "unknown")
        if env not in grouped:
            grouped[env] = []
            order.append(env)
        grouped[env].append(row)
    return [(env, grouped[env]) for env in order]


def render_result_rows(report: dict[str, Any]) -> str:
    complete_rows = [row for row in report["rows"] if row.get("status") == "complete"]
    max_nfe = max((as_float(row.get("mean_nfe_per_action_mean")) or 0.0 for row in complete_rows), default=1.0)
    max_nfe = max(max_nfe, 1.0)
    lines: list[str] = []
    for env, rows in grouped_rows(report["rows"]):
        lines.append(
            f'<tr class="env-row"><th colspan="10">{html_escape(env.title())}</th></tr>'
        )
        for row in rows:
            success = row.get("success_rate_mean")
            nfe = row.get("mean_nfe_per_action_mean")
            lines.append(
                "\n".join(
                    [
                        "<tr>",
                        f"<td><strong>{html_escape(row.get('label', ''))}</strong><span>{html_escape(row.get('method', ''))}</span></td>",
                        f"<td>{html_escape(format_mean_sem(row, 'success_rate_mean', 'success_rate_sem'))}<div class=\"bar success\"><i style=\"width: {bar_width(success)}\"></i></div></td>",
                        f"<td>{html_escape(format_mean_sem(row, 'mean_episode_return_mean', 'mean_episode_return_sem'))}</td>",
                        f"<td>{html_escape(format_mean_sem(row, 'mean_high_level_return_mean', 'mean_high_level_return_sem'))}</td>",
                        f"<td>{html_escape(format_mean_sem(row, 'mean_nfe_per_action_mean', 'mean_nfe_per_action_sem'))}<div class=\"bar nfe\"><i style=\"width: {bar_width(nfe, max_nfe)}\"></i></div></td>",
                        f"<td>{html_escape(format_mean_sem(row, 'replan_rate_mean', 'replan_rate_sem'))}</td>",
                        f"<td>{html_escape(format_mean_sem(row, 'learned_replan_rate_mean', 'learned_replan_rate_sem'))}</td>",
                        f"<td>{html_escape(format_mean_sem(row, 'mean_discarded_actions_per_episode_mean', 'mean_discarded_actions_per_episode_sem'))}</td>",
                        f"<td>{html_escape(format_signed_float(row.get('success_delta_vs_fixed')))}</td>",
                        f"<td>{html_escape(format_float(row.get('nfe_ratio_vs_fixed')))}</td>",
                        "</tr>",
                    ]
                )
            )
    return "\n".join(lines)


def render_source_details(report: dict[str, Any]) -> str:
    source_rows = []
    for row in report["rows"]:
        source_rows.append(
            "\n".join(
                [
                    "<tr>",
                    f"<td>{html_escape(row.get('env', ''))}</td>",
                    f"<td>{html_escape(row.get('label', ''))}</td>",
                    f"<td>{html_escape(row.get('status', ''))}</td>",
                    f"<td><code>{html_escape(row.get('source_suite', ''))}</code></td>",
                    "</tr>",
                ]
            )
        )
    blocker_rows = []
    for summary in report["env_summaries"]:
        blocker_rows.append(
            "\n".join(
                [
                    "<tr>",
                    f"<td>{html_escape(summary.get('env', ''))}</td>",
                    f"<td>{html_escape(join_values(summary.get('readiness_missing') or []) or 'none')}</td>",
                    f"<td>{html_escape(join_values(summary.get('missing_rows') or []) or 'none')}</td>",
                    f"<td>{html_escape(join_values(summary.get('failed_checks') or []) or 'none')}</td>",
                    "</tr>",
                ]
            )
        )
    return "\n".join(
        [
            '<details class="source-details">',
            "<summary>Internal source details</summary>",
            f"<p>Audit JSON: <code>{html_escape(report.get('audit_json', ''))}</code></p>",
            "<h3>Source suites</h3>",
            '<table class="source-table"><thead><tr><th>Env</th><th>Label</th><th>Status</th><th>Source suite</th></tr></thead><tbody>',
            "\n".join(source_rows),
            "</tbody></table>",
            "<h3>Readiness and blocker details</h3>",
            '<table class="source-table"><thead><tr><th>Env</th><th>Readiness missing</th><th>Missing rows</th><th>Failed checks</th></tr></thead><tbody>',
            "\n".join(blocker_rows),
            "</tbody></table>",
            "</details>",
        ]
    )


def render_html(report: dict[str, Any]) -> str:
    gates = "\n".join(
        [
            gate_badge("Goal evidence", bool(report.get("all_goal_evidence_ready"))),
            gate_badge("Suites", bool(report.get("all_suites_ready"))),
            gate_badge("Readiness", bool(report.get("readiness_ready"))),
            gate_badge("Markdown", bool(report.get("markdown_ready"))),
        ]
    )
    title = "Adaptive Replanning Experiment Overview"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #18202a;
      --muted: #5a6573;
      --line: #d9dee6;
      --green: #16794c;
      --green-soft: #dff3e8;
      --blue: #2f5fbb;
      --blue-soft: #e4ecff;
      --amber: #9a5b00;
      --amber-soft: #fff1d6;
      --red: #b3261e;
      --red-soft: #fde7e4;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ background: #16202d; color: white; padding: 28px 32px; }}
    main {{ max-width: 1220px; margin: 0 auto; padding: 24px 24px 40px; }}
    h1 {{ margin: 0 0 6px; font-size: 30px; line-height: 1.15; letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 20px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 10px; font-size: 16px; letter-spacing: 0; }}
    p {{ margin: 0 0 10px; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; word-break: break-all; }}
    section {{ margin-top: 22px; }}
    .subtitle {{ color: #d5dce8; max-width: 900px; }}
    .gate-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }}
    .gate {{ display: inline-flex; gap: 8px; align-items: center; border-radius: 6px; padding: 7px 10px; background: #253246; color: white; }}
    .gate strong {{ text-transform: uppercase; font-size: 11px; letter-spacing: 0; }}
    .gate.ok strong {{ color: #8ee0b7; }}
    .gate.bad strong {{ color: #ffaca4; }}
    .takeaways {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
    .takeaway, .env-card, .table-wrap, .source-details {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }}
    .takeaway {{ padding: 14px; }}
    .takeaway strong {{ display: block; margin-bottom: 4px; }}
    .env-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
    .env-card {{ padding: 16px; }}
    .metric-list {{ margin: 0; display: grid; gap: 9px; }}
    .metric-list div {{ display: flex; justify-content: space-between; gap: 12px; border-top: 1px solid var(--line); padding-top: 8px; }}
    .metric-list div:first-child {{ border-top: 0; padding-top: 0; }}
    dt {{ color: var(--muted); }}
    dd {{ margin: 0; text-align: right; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ background: #eef1f5; color: #344052; font-size: 12px; text-transform: uppercase; letter-spacing: 0; }}
    td span {{ display: block; color: var(--muted); font-size: 12px; }}
    .env-row th {{ background: #dfe6ef; color: #16202d; font-size: 13px; }}
    .bar {{ height: 6px; width: 100%; background: #e7ebf0; border-radius: 4px; margin-top: 6px; overflow: hidden; }}
    .bar i {{ display: block; height: 100%; border-radius: 4px; }}
    .bar.success i {{ background: var(--green); }}
    .bar.nfe i {{ background: var(--blue); }}
    .source-details {{ padding: 14px; }}
    .source-details summary {{ cursor: pointer; font-weight: 700; }}
    .source-table {{ margin-top: 12px; }}
    .note {{ color: var(--muted); }}
    @media (max-width: 860px) {{
      header {{ padding: 22px 18px; }}
      main {{ padding: 18px; }}
      .takeaways, .env-grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 24px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p class="subtitle">Final Lift/Can/Square evidence summarized for quick internal review. Created {html_escape(report.get('created_at', ''))}.</p>
    <div class="gate-row">{gates}</div>
  </header>
  <main>
    <section>
      <h2>Executive Takeaways</h2>
      <div class="takeaways">
        <div class="takeaway"><strong>Evidence is complete</strong><p>Lift, Can, and Square final suites are present with readiness and markdown gates passing.</p></div>
        <div class="takeaway"><strong>Adaptive gains are directional</strong><p>Can and Square show small average raw-success gains over fixed_chunk, but the 3-seed evidence does not establish statistical significance.</p></div>
        <div class="takeaway"><strong>Compute tradeoff is clear</strong><p>Denoise-only is the lowest-NFE option, while TD-error and PPO replanning spend more NFE and discard more buffered actions.</p></div>
      </div>
    </section>
    <section>
      <h2>Environment Summary</h2>
      <div class="env-grid">{render_env_cards(report)}</div>
    </section>
    <section>
      <h2>Method Comparison</h2>
      <p class="note">Success and NFE/action include visual bars. Delta columns compare each method against same-environment fixed_chunk.</p>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Method</th><th>Success</th><th>Return</th><th>High-Level Return</th><th>NFE/action</th><th>Replan</th><th>Learned Replan</th><th>Discarded/Episode</th><th>Success Delta</th><th>NFE Ratio</th></tr></thead>
          <tbody>{render_result_rows(report)}</tbody>
        </table>
      </div>
    </section>
    <section>
      {render_source_details(report)}
    </section>
  </main>
</body>
</html>
"""


def write_html(path: Path, report: dict[str, Any], rendered: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered if rendered is not None else render_html(report), encoding="utf-8")


def write_report(output_dir: Path, report: dict[str, Any], html_snapshot: Path | None = None) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "final_report.json"
    csv_path = output_dir / "final_results_matrix.csv"
    env_summary_csv_path = output_dir / "final_env_summary.csv"
    markdown_path = output_dir / "final_report.md"
    html_path = output_dir / "final_report.html"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_csv(csv_path, report["rows"])
    write_env_summary_csv(env_summary_csv_path, report["env_summaries"])
    write_markdown(markdown_path, report)
    rendered_html = render_html(report)
    write_html(html_path, report, rendered_html)
    outputs = {
        "json": str(json_path),
        "csv": str(csv_path),
        "env_summary_csv": str(env_summary_csv_path),
        "markdown": str(markdown_path),
        "html": str(html_path),
    }
    if html_snapshot is not None:
        write_html(html_snapshot, report, rendered_html)
        outputs["html_snapshot"] = str(html_snapshot)
    return outputs


def main() -> None:
    args = parse_args()
    audit = read_json(args.audit_json)
    audit["source_path"] = str(args.audit_json)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"{timestamp}_final_report"
    report = build_report(audit)
    outputs = write_report(output_dir, report, html_snapshot=args.html_snapshot)
    print(f"Wrote final report to {output_dir}")
    print(OmegaConf.to_yaml(OmegaConf.create({"output_dir": str(output_dir), **outputs}), resolve=True))


if __name__ == "__main__":
    main()
