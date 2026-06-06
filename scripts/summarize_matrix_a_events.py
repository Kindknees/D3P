"""Summarize Matrix A episode and event metrics."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


FIELDS = [
    "task",
    "method",
    "num_episodes",
    "success_rate",
    "mean_return",
    "mean_nfe_per_action",
    "mean_repairs",
    "mean_random_resets",
    "mean_discard_nonempty_buffer",
    "mean_action_jump",
    "mean_buffer_jump",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output = Path(args.output) if args.output else output_root / "analysis" / "matrix_a_event_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for path in output_root.rglob("episode_summary.csv"):
        with path.open(newline="") as file_obj:
            for row in csv.DictReader(file_obj):
                groups[(row["task"], row["method"])].append(row)

    rows = []
    for (task, method), group in sorted(groups.items()):
        rows.append(
            {
                "task": task,
                "method": method,
                "num_episodes": len(group),
                "success_rate": mean(_float(row["success"]) for row in group),
                "mean_return": mean(_float(row["episode_return"]) for row in group),
                "mean_nfe_per_action": mean(_float(row["nfe_per_action"]) for row in group),
                "mean_repairs": mean(_float(row["num_repairs"]) for row in group),
                "mean_random_resets": mean(_float(row["num_random_resets"]) for row in group),
                "mean_discard_nonempty_buffer": mean(
                    _float(row["num_discard_nonempty_buffer"]) for row in group
                ),
                "mean_action_jump": mean(_float(row["mean_action_jump"]) for row in group),
                "mean_buffer_jump": mean(_float(row["mean_buffer_jump"]) for row in group),
            }
        )

    with output.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote Matrix A event summary to {output}")


def _float(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


if __name__ == "__main__":
    main()
