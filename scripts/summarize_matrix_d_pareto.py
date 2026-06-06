"""Summarize Matrix D return-cost Pareto runs."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


CONFIG_RE = re.compile(r"reset_nfe_(\d+)_repair_nfe_(\d+)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root")
    parser.add_argument("--analysis-dir", default=None)
    args = parser.parse_args()

    root = Path(args.output_root)
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(root)
    summary = _summarize(rows)
    efficient = _pareto_efficient(summary)
    for row in summary:
        row["pareto_efficient"] = int(id(row) in efficient)

    _write_csv(analysis_dir / "matrix_d_pareto_summary.csv", summary)
    _write_report(analysis_dir / "matrix_d_pareto_report.md", summary)
    print(f"Wrote Matrix D Pareto summary to {analysis_dir / 'matrix_d_pareto_summary.csv'}")


def _load_rows(root: Path) -> list[dict[str, object]]:
    rows = []
    for path in root.rglob("episode_summary.csv"):
        config_name = next((part for part in path.parts if CONFIG_RE.fullmatch(part)), None)
        if config_name is None:
            continue
        match = CONFIG_RE.fullmatch(config_name)
        assert match is not None
        with path.open(newline="") as file_obj:
            for row in csv.DictReader(file_obj):
                row["reset_nfe"] = int(match.group(1))
                row["repair_nfe"] = int(match.group(2))
                rows.append(row)
    if not rows:
        raise SystemExit(f"No episode_summary.csv files found under {root}")
    return rows


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["reset_nfe"], row["repair_nfe"], row["task"], row["method"])].append(row)

    summary = []
    for (reset_nfe, repair_nfe, task, method), group in sorted(groups.items()):
        summary.append(
            {
                "reset_nfe": reset_nfe,
                "repair_nfe": repair_nfe,
                "task": task,
                "method": method,
                "num_episodes": len(group),
                "success_rate": mean(float(row["success"]) for row in group),
                "mean_return": mean(float(row["episode_return"]) for row in group),
                "mean_nfe_per_action": mean(float(row["nfe_per_action"]) for row in group),
                "mean_action_jump": mean(float(row["mean_action_jump"]) for row in group),
                "mean_buffer_jump": mean(float(row["mean_buffer_jump"]) for row in group),
            }
        )
    return summary


def _pareto_efficient(rows: list[dict[str, object]]) -> set[int]:
    efficient = set()
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            at_least_as_good = (
                float(other["mean_return"]) >= float(row["mean_return"])
                and float(other["success_rate"]) >= float(row["success_rate"])
                and float(other["mean_nfe_per_action"]) <= float(row["mean_nfe_per_action"])
            )
            strictly_better = (
                float(other["mean_return"]) > float(row["mean_return"])
                or float(other["success_rate"]) > float(row["success_rate"])
                or float(other["mean_nfe_per_action"]) < float(row["mean_nfe_per_action"])
            )
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            efficient.add(id(row))
    return efficient


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "reset_nfe",
        "repair_nfe",
        "task",
        "method",
        "num_episodes",
        "success_rate",
        "mean_return",
        "mean_nfe_per_action",
        "mean_action_jump",
        "mean_buffer_jump",
        "pareto_efficient",
    ]
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_report(path: Path, rows: list[dict[str, object]]) -> None:
    lines = ["# Matrix D Pareto Summary", "", "## Successful Pareto-Efficient Points", ""]
    efficient = [
        row for row in rows
        if row.get("pareto_efficient") and float(row["success_rate"]) > 0
    ]
    for row in sorted(efficient, key=lambda r: (float(r["mean_nfe_per_action"]), -float(r["mean_return"]))):
        lines.append(
            f"- reset_nfe={row['reset_nfe']} repair_nfe={row['repair_nfe']} "
            f"{row['method']}: success={float(row['success_rate']):.4f}, "
            f"return={float(row['mean_return']):.4f}, "
            f"nfe/action={float(row['mean_nfe_per_action']):.4f}"
        )
    lines.extend(["", "## Note", "", "Zero-return low-cost points remain marked in the CSV when they are mathematically non-dominated, but they are omitted from the successful frontier above."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
