"""Analyze adaptive commitment experiment logs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", help="Directory containing experiment CSV logs.")
    parser.add_argument(
        "--analysis-dir",
        default=None,
        help="Output directory for summary tables and report.",
    )
    parser.add_argument("--baseline", default="td_replan_anywhere_random_reset")
    parser.add_argument("--fallback-baseline", default="fixed_chunk")
    parser.add_argument("--min-success-delta", type=float, default=0.01)
    parser.add_argument("--max-nfe-ratio", type=float, default=1.25)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows = load_episode_summaries(output_root)
    summary = summarize(rows)
    positives = detect_positive_signals(
        summary,
        baseline=args.baseline,
        fallback_baseline=args.fallback_baseline,
        min_success_delta=args.min_success_delta,
        max_nfe_ratio=args.max_nfe_ratio,
    )

    write_summary(analysis_dir / "summary_by_task_method.csv", summary)
    write_positive_signals(analysis_dir / "positive_signals.csv", positives)
    write_report(analysis_dir / "report.md", summary, positives)
    print(f"Wrote analysis to {analysis_dir}")
    if positives:
        print(f"positive_signals={len(positives)}")
    else:
        print("positive_signals=0")


def load_episode_summaries(output_root: Path) -> list[dict[str, str]]:
    rows = []
    for path in output_root.rglob("episode_summary.csv"):
        with path.open(newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            for row in reader:
                row["_source"] = str(path)
                rows.append(row)
    if not rows:
        raise SystemExit(f"No episode_summary.csv files found under {output_root}")
    return rows


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("task", ""), row.get("method", ""))].append(row)

    summary = []
    for (task, method), group_rows in sorted(grouped.items()):
        returns = [_float(row.get("episode_return")) for row in group_rows]
        successes = [_float(row.get("success")) for row in group_rows]
        nfe_per_action = [_float(row.get("nfe_per_action")) for row in group_rows]
        action_jump = [_float(row.get("mean_action_jump")) for row in group_rows]
        buffer_jump = [_float(row.get("mean_buffer_jump")) for row in group_rows]
        summary.append(
            {
                "task": task,
                "method": method,
                "num_episodes": len(group_rows),
                "success_rate": mean(successes),
                "mean_return": mean(returns),
                "mean_nfe_per_action": mean(nfe_per_action),
                "mean_action_jump": mean(action_jump),
                "mean_buffer_jump": mean(buffer_jump),
            }
        )
    return summary


def detect_positive_signals(
    summary: list[dict[str, object]],
    baseline: str,
    fallback_baseline: str,
    min_success_delta: float,
    max_nfe_ratio: float,
) -> list[dict[str, object]]:
    by_task_method = {
        (str(row["task"]), str(row["method"])): row for row in summary
    }
    tasks = sorted({str(row["task"]) for row in summary})
    positives = []
    for task in tasks:
        baseline_row = by_task_method.get((task, baseline)) or by_task_method.get(
            (task, fallback_baseline)
        )
        if baseline_row is None:
            continue
        for row in summary:
            if row["task"] != task or row["method"] == baseline_row["method"]:
                continue
            success_delta = row["success_rate"] - baseline_row["success_rate"]
            return_delta = row["mean_return"] - baseline_row["mean_return"]
            nfe_ratio = _safe_ratio(
                row["mean_nfe_per_action"], baseline_row["mean_nfe_per_action"]
            )
            improves_success = success_delta >= min_success_delta
            improves_return = return_delta > 0 and success_delta >= 0
            within_budget = nfe_ratio <= max_nfe_ratio
            if (improves_success or improves_return) and within_budget:
                positives.append(
                    {
                        "task": task,
                        "method": row["method"],
                        "baseline": baseline_row["method"],
                        "success_delta": success_delta,
                        "return_delta": return_delta,
                        "nfe_ratio": nfe_ratio,
                    }
                )
    return positives


def write_summary(path: Path, summary: list[dict[str, object]]) -> None:
    fields = [
        "task",
        "method",
        "num_episodes",
        "success_rate",
        "mean_return",
        "mean_nfe_per_action",
        "mean_action_jump",
        "mean_buffer_jump",
    ]
    _write_csv(path, fields, summary)


def write_positive_signals(path: Path, positives: list[dict[str, object]]) -> None:
    fields = [
        "task",
        "method",
        "baseline",
        "success_delta",
        "return_delta",
        "nfe_ratio",
    ]
    _write_csv(path, fields, positives)


def write_report(
    path: Path,
    summary: list[dict[str, object]],
    positives: list[dict[str, object]],
) -> None:
    lines = ["# Adaptive Commitment Analysis", ""]
    lines.append("## Summary")
    lines.append("")
    for row in summary:
        lines.append(
            f"- {row['task']} / {row['method']}: success={row['success_rate']:.4f}, "
            f"return={row['mean_return']:.4f}, nfe/action={row['mean_nfe_per_action']:.4f}"
        )
    lines.append("")
    lines.append("## Positive Signals")
    lines.append("")
    if positives:
        for row in positives:
            lines.append(
                f"- {row['task']} / {row['method']} beats {row['baseline']}: "
                f"success_delta={row['success_delta']:.4f}, "
                f"return_delta={row['return_delta']:.4f}, "
                f"nfe_ratio={row['nfe_ratio']:.4f}"
            )
    else:
        lines.append("- No positive signal detected by the configured thresholds.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _float(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _safe_ratio(value: float, baseline: float) -> float:
    if baseline == 0:
        return float("inf") if value > 0 else 1.0
    return value / baseline


if __name__ == "__main__":
    main()
