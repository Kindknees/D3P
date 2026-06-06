"""Analyze paired confirmatory residual-buffer repair runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np


ADAPTIVE = "adaptive_commitment_with_repair"
METRICS = [
    "episode_return",
    "success",
    "nfe_per_action",
    "mean_action_jump",
    "mean_buffer_jump",
    "num_random_resets",
    "num_discard_nonempty_buffer",
]
SUMMARY_FIELDS = [
    "task",
    "method",
    "episodes",
    "success_rate",
    "mean_return",
    "std_return",
    "mean_nfe_per_action",
    "nfe_ratio_vs_fixed",
    "nfe_ratio_vs_random_reset",
    "mean_repairs_accepted_per_episode",
    "mean_repair_candidates_per_episode",
    "repair_accept_rate",
    "repair_intervention_ratio",
    "repairs_per_100_steps",
    "mean_repairs_rejected_per_episode",
    "mean_random_resets_per_episode",
    "mean_discard_nonempty_buffer_per_episode",
    "mean_action_jump",
    "mean_buffer_jump",
    "max_action_jump",
    "max_buffer_jump",
]
DELTA_FIELDS = [
    "task",
    "comparison",
    "seed",
    "episode_id",
    "delta_return",
    "delta_success",
    "delta_nfe_per_action",
    "delta_action_jump",
    "delta_buffer_jump",
    "delta_random_resets",
    "delta_discard_nonempty_buffer",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root")
    parser.add_argument("--baseline", default="fixed_chunk")
    parser.add_argument("--secondary-baseline", default="td_replan_anywhere_random_reset")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    analysis_dir = output_root / "analysis"
    plot_dir = analysis_dir / "plots"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(output_root)
    baselines = [args.baseline, args.secondary_baseline]
    assert_paired(rows, baselines + [ADAPTIVE])

    summary = summarize(rows, args.baseline, args.secondary_baseline)
    paired_rows, stats = paired_analysis(
        rows,
        baselines=baselines,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    write_csv(analysis_dir / "confirmatory_summary.csv", SUMMARY_FIELDS, summary)
    write_csv(analysis_dir / "paired_deltas.csv", DELTA_FIELDS, paired_rows)
    (analysis_dir / "stat_tests.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(analysis_dir / "report.md", summary, stats)
    make_plots(rows, summary, stats, baselines, plot_dir)

    print(f"Wrote confirmatory analysis to {analysis_dir}")


def load_rows(output_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in output_root.rglob("episode_summary.csv"):
        with path.open(newline="") as file_obj:
            for row in csv.DictReader(file_obj):
                row = dict(row)
                row["_source"] = str(path)
                normalize_row(row)
                rows.append(row)
    if not rows:
        raise SystemExit(f"No episode_summary.csv files found under {output_root}")
    return rows


def normalize_row(row: dict[str, object]) -> None:
    for field in [
        "seed",
        "episode_id",
        "success",
        "episode_length",
        "num_replans",
        "num_repairs",
        "num_repairs_accepted",
        "num_repairs_rejected",
        "num_repair_candidates",
        "num_random_resets",
        "num_discard_nonempty_buffer",
    ]:
        row[field] = int(float(row.get(field) or 0))
    for field in [
        "episode_return",
        "nfe_per_action",
        "mean_action_jump",
        "mean_buffer_jump",
        "max_action_jump",
        "max_buffer_jump",
    ]:
        row[field] = float(row.get(field) or 0.0)
    if "num_repairs_accepted" not in row or row["num_repairs_accepted"] == 0:
        row["num_repairs_accepted"] = int(row.get("num_repairs", 0))
    if row["episode_length"] == 0:
        source = str(row.get("_source", ""))
        # Legacy summaries used max steps as implicit episode length. The
        # confirmatory runner writes the explicit value; this path is fallback.
        row["episode_length"] = 400 if "square" in source else 300


def assert_paired(rows: list[dict[str, object]], methods: list[str]) -> None:
    grouped: dict[tuple[str, str], set[tuple[str, int, int]]] = defaultdict(set)
    counts: dict[tuple[str, str, int, int], int] = defaultdict(int)
    for row in rows:
        task = str(row["task"])
        method = str(row["method"])
        if method not in methods:
            continue
        key = (task, method, int(row["seed"]), int(row["episode_id"]))
        counts[key] += 1
        grouped[(task, method)].add((task, int(row["seed"]), int(row["episode_id"])))
    duplicates = [key for key, count in counts.items() if count != 1]
    if duplicates:
        raise SystemExit(f"Expected exactly one row per matched key, duplicates={duplicates[:10]}")
    tasks = sorted({str(row["task"]) for row in rows})
    for task in tasks:
        expected = None
        for method in methods:
            keys = grouped.get((task, method), set())
            if not keys:
                raise SystemExit(f"Missing method rows for task={task}, method={method}")
            if expected is None:
                expected = keys
            elif keys != expected:
                missing = sorted(expected - keys)[:10]
                extra = sorted(keys - expected)[:10]
                raise SystemExit(
                    f"Paired keys mismatch for task={task}, method={method}, "
                    f"missing={missing}, extra={extra}"
                )


def summarize(
    rows: list[dict[str, object]],
    fixed: str,
    random_reset: str,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), str(row["method"]))].append(row)
    base_nfe = {}
    for (task, method), group in grouped.items():
        if method in {fixed, random_reset}:
            base_nfe[(task, method)] = mean_float(group, "nfe_per_action")

    summary = []
    for (task, method), group in sorted(grouped.items()):
        accepted = sum(int(row["num_repairs_accepted"]) for row in group)
        candidates = sum(int(row["num_repair_candidates"]) for row in group)
        random_resets = sum(int(row["num_random_resets"]) for row in group)
        episode_lengths = sum(int(row["episode_length"]) for row in group)
        nfe = mean_float(group, "nfe_per_action")
        returns = [float(row["episode_return"]) for row in group]
        summary.append(
            {
                "task": task,
                "method": method,
                "episodes": len(group),
                "success_rate": mean_float(group, "success"),
                "mean_return": mean(returns),
                "std_return": stdev(returns) if len(returns) > 1 else 0.0,
                "mean_nfe_per_action": nfe,
                "nfe_ratio_vs_fixed": safe_ratio(nfe, base_nfe.get((task, fixed), 0.0)),
                "nfe_ratio_vs_random_reset": safe_ratio(nfe, base_nfe.get((task, random_reset), 0.0)),
                "mean_repairs_accepted_per_episode": accepted / max(len(group), 1),
                "mean_repair_candidates_per_episode": candidates / max(len(group), 1),
                "repair_accept_rate": accepted / max(candidates, 1),
                "repair_intervention_ratio": accepted / max(accepted + random_resets, 1),
                "repairs_per_100_steps": 100.0 * accepted / max(episode_lengths, 1),
                "mean_repairs_rejected_per_episode": mean_float(group, "num_repairs_rejected"),
                "mean_random_resets_per_episode": random_resets / max(len(group), 1),
                "mean_discard_nonempty_buffer_per_episode": mean_float(group, "num_discard_nonempty_buffer"),
                "mean_action_jump": mean_float(group, "mean_action_jump"),
                "mean_buffer_jump": mean_float(group, "mean_buffer_jump"),
                "max_action_jump": max(float(row["max_action_jump"]) for row in group),
                "max_buffer_jump": max(float(row["max_buffer_jump"]) for row in group),
            }
        )
    return summary


def paired_analysis(
    rows: list[dict[str, object]],
    baselines: list[str],
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_key = {
        (str(row["task"]), str(row["method"]), int(row["seed"]), int(row["episode_id"])): row
        for row in rows
    }
    tasks = sorted({str(row["task"]) for row in rows})
    rng = np.random.default_rng(seed)
    paired_rows: list[dict[str, object]] = []
    stats: dict[str, object] = {"bootstrap_samples": bootstrap_samples, "comparisons": []}
    for task in tasks:
        keys = sorted(
            (str(row["task"]), int(row["seed"]), int(row["episode_id"]))
            for row in rows
            if str(row["task"]) == task and str(row["method"]) == ADAPTIVE
        )
        for baseline in baselines:
            deltas = {metric: [] for metric in METRICS}
            mcnemar = {"adaptive_wins": 0, "baseline_wins": 0, "both_success": 0, "both_fail": 0}
            for _, seed_value, episode_id in keys:
                adaptive = by_key[(task, ADAPTIVE, seed_value, episode_id)]
                base = by_key[(task, baseline, seed_value, episode_id)]
                row = {
                    "task": task,
                    "comparison": f"{ADAPTIVE} - {baseline}",
                    "seed": seed_value,
                    "episode_id": episode_id,
                    "delta_return": float(adaptive["episode_return"]) - float(base["episode_return"]),
                    "delta_success": int(adaptive["success"]) - int(base["success"]),
                    "delta_nfe_per_action": float(adaptive["nfe_per_action"]) - float(base["nfe_per_action"]),
                    "delta_action_jump": float(adaptive["mean_action_jump"]) - float(base["mean_action_jump"]),
                    "delta_buffer_jump": float(adaptive["mean_buffer_jump"]) - float(base["mean_buffer_jump"]),
                    "delta_random_resets": int(adaptive["num_random_resets"]) - int(base["num_random_resets"]),
                    "delta_discard_nonempty_buffer": int(adaptive["num_discard_nonempty_buffer"]) - int(base["num_discard_nonempty_buffer"]),
                }
                paired_rows.append(row)
                deltas["episode_return"].append(row["delta_return"])
                deltas["success"].append(row["delta_success"])
                deltas["nfe_per_action"].append(row["delta_nfe_per_action"])
                deltas["mean_action_jump"].append(row["delta_action_jump"])
                deltas["mean_buffer_jump"].append(row["delta_buffer_jump"])
                deltas["num_random_resets"].append(row["delta_random_resets"])
                deltas["num_discard_nonempty_buffer"].append(row["delta_discard_nonempty_buffer"])
                adaptive_success = int(adaptive["success"])
                base_success = int(base["success"])
                if adaptive_success and not base_success:
                    mcnemar["adaptive_wins"] += 1
                elif base_success and not adaptive_success:
                    mcnemar["baseline_wins"] += 1
                elif adaptive_success and base_success:
                    mcnemar["both_success"] += 1
                else:
                    mcnemar["both_fail"] += 1
            comp = {
                "task": task,
                "baseline": baseline,
                "comparison": f"{ADAPTIVE} - {baseline}",
                "num_pairs": len(keys),
                "return_win_rate": float(np.mean(np.asarray(deltas["episode_return"]) > 0)),
                "success_win_rate": float(np.mean(np.asarray(deltas["success"]) > 0)),
                "mcnemar_counts": mcnemar,
                "metrics": {},
            }
            for metric, values in deltas.items():
                arr = np.asarray(values, dtype=float)
                ci = bootstrap_mean_ci(arr, bootstrap_samples, rng)
                comp["metrics"][metric] = {
                    "mean_delta": float(np.mean(arr)),
                    "ci95": [float(ci[0]), float(ci[1])],
                }
            comp["decision"] = classify_decision(comp, baseline)
            stats["comparisons"].append(comp)
    return paired_rows, stats


def bootstrap_mean_ci(values: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    if samples <= 0:
        return float(np.mean(values)), float(np.mean(values))
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def classify_decision(comp: dict[str, object], baseline: str) -> str:
    metrics = comp["metrics"]
    ret = metrics["episode_return"]
    if ret["mean_delta"] <= 0:
        return "not_supported"
    if ret["ci95"][0] > 0:
        return "confirmatory_evidence"
    return "positive_trend"


def write_report(path: Path, summary: list[dict[str, object]], stats: dict[str, object]) -> None:
    lines = ["# Confirmatory Repair Analysis", "", "## Summary", ""]
    lines.append("| Task | Method | Episodes | Success | Return | NFE/action | Repairs/ep | Random resets/ep | Discard non-empty/ep | Action jump | Buffer jump |")
    lines.append("| ---- | ------ | -------: | ------: | -----: | ---------: | ---------: | ---------------: | -------------------: | ----------: | ----------: |")
    for row in summary:
        lines.append(
            f"| {row['task']} | {row['method']} | {row['episodes']} | "
            f"{row['success_rate']:.4f} | {row['mean_return']:.4f} | "
            f"{row['mean_nfe_per_action']:.4f} | "
            f"{row['mean_repairs_accepted_per_episode']:.4f} | "
            f"{row['mean_random_resets_per_episode']:.4f} | "
            f"{row['mean_discard_nonempty_buffer_per_episode']:.4f} | "
            f"{row['mean_action_jump']:.4f} | {row['mean_buffer_jump']:.4f} |"
        )
    lines.extend(["", "## Paired Deltas", ""])
    lines.append("| Task | Comparison | Delta Return | 95% CI | Return win rate | Delta Success | Delta NFE/action | Delta Action jump | Delta Discard non-empty | Decision |")
    lines.append("| ---- | ---------- | -----------: | ------ | --------------: | ------------: | ---------------: | ----------------: | ----------------------: | -------- |")
    for comp in stats["comparisons"]:
        metrics = comp["metrics"]
        ret = metrics["episode_return"]
        lines.append(
            f"| {comp['task']} | {comp['comparison']} | "
            f"{ret['mean_delta']:.4f} | [{ret['ci95'][0]:.4f}, {ret['ci95'][1]:.4f}] | "
            f"{comp['return_win_rate']:.4f} | "
            f"{metrics['success']['mean_delta']:.4f} | "
            f"{metrics['nfe_per_action']['mean_delta']:.4f} | "
            f"{metrics['mean_action_jump']['mean_delta']:.4f} | "
            f"{metrics['num_discard_nonempty_buffer']['mean_delta']:.4f} | "
            f"{comp['decision']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(
    rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    stats: dict[str, object],
    baselines: list[str],
    plot_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on environment
        (plot_dir / "PLOTS_SKIPPED.txt").write_text(f"matplotlib unavailable: {exc}\n")
        return

    by_key = {
        (str(row["task"]), str(row["method"]), int(row["seed"]), int(row["episode_id"])): row
        for row in rows
    }
    tasks = sorted({str(row["task"]) for row in rows})
    summary_by_task = defaultdict(list)
    for row in summary:
        summary_by_task[str(row["task"])].append(row)

    for task in tasks:
        adaptive_keys = sorted(
            (int(row["seed"]), int(row["episode_id"]))
            for row in rows
            if str(row["task"]) == task and str(row["method"]) == ADAPTIVE
        )
        for baseline in baselines:
            x = [by_key[(task, baseline, seed, ep)]["episode_return"] for seed, ep in adaptive_keys]
            y = [by_key[(task, ADAPTIVE, seed, ep)]["episode_return"] for seed, ep in adaptive_keys]
            plt.figure(figsize=(5, 5))
            plt.scatter(x, y, alpha=0.75)
            lo = min(x + y)
            hi = max(x + y)
            plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
            plt.xlabel(f"{baseline} return")
            plt.ylabel(f"{ADAPTIVE} return")
            plt.title(f"{task}: paired return")
            plt.tight_layout()
            plt.savefig(plot_dir / f"{task}_return_scatter_vs_{baseline}.png", dpi=160)
            plt.close()

        task_summary = summary_by_task[task]
        methods = [row["method"] for row in task_summary]
        x_pos = np.arange(len(methods))

        plt.figure(figsize=(7, 4))
        plt.bar(x_pos, [row["mean_nfe_per_action"] for row in task_summary])
        plt.xticks(x_pos, methods, rotation=25, ha="right")
        plt.ylabel("NFE/action")
        plt.title(f"{task}: NFE/action by method")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{task}_nfe_by_method.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 4))
        width = 0.2
        series = [
            ("candidates", "mean_repair_candidates_per_episode"),
            ("accepted", "mean_repairs_accepted_per_episode"),
            ("rejected", "mean_repairs_rejected_per_episode"),
            ("random reset", "mean_random_resets_per_episode"),
        ]
        for i, (label, field) in enumerate(series):
            plt.bar(x_pos + (i - 1.5) * width, [row[field] for row in task_summary], width, label=label)
        plt.xticks(x_pos, methods, rotation=25, ha="right")
        plt.ylabel("count/episode")
        plt.title(f"{task}: repair/reset composition")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{task}_repairs_per_episode.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 4))
        series = [
            ("action jump", "mean_action_jump"),
            ("buffer jump", "mean_buffer_jump"),
            ("discard non-empty", "mean_discard_nonempty_buffer_per_episode"),
        ]
        width = 0.25
        for i, (label, field) in enumerate(series):
            plt.bar(x_pos + (i - 1) * width, [row[field] for row in task_summary], width, label=label)
        plt.xticks(x_pos, methods, rotation=25, ha="right")
        plt.title(f"{task}: discontinuity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{task}_discontinuity.png", dpi=160)
        plt.close()

        plt.figure(figsize=(6, 4))
        adaptive_rows = [row for row in rows if str(row["task"]) == task and str(row["method"]) == ADAPTIVE]
        plt.scatter(
            [row["num_repairs_accepted"] for row in adaptive_rows],
            [row["episode_return"] for row in adaptive_rows],
            alpha=0.75,
        )
        plt.xlabel("accepted repairs")
        plt.ylabel("episode return")
        plt.title(f"{task}: return vs accepted repairs")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{task}_return_vs_repairs.png", dpi=160)
        plt.close()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def mean_float(rows: list[dict[str, object]], field: str) -> float:
    return float(np.mean([float(row.get(field, 0.0)) for row in rows]))


def safe_ratio(value: float, baseline: float) -> float:
    if baseline == 0:
        return float("inf") if value > 0 else 1.0
    return value / baseline


if __name__ == "__main__":
    main()
