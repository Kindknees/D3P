#!/usr/bin/env python3
"""Run a TD-error threshold sweep and aggregate evaluation summaries."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_BASELINE = REPO_ROOT / "scripts" / "run_baseline.py"


@dataclass(frozen=True)
class SweepSpec:
    threshold_mode: str
    label: str
    threshold: float | None = None
    percentile: float | None = None
    calibrated_threshold: float | None = None


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in value.split(",") if part)
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one value")
    return values


def format_float(value: float) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def load_percentile_thresholds(path: str | Path) -> dict[str, float]:
    with Path(path).open("r", encoding="utf-8") as f:
        calibration = json.load(f)
    return {
        str(key): float(value)
        for key, value in calibration.get("percentile_thresholds", {}).items()
    }


def build_sweep_specs(
    threshold_mode: str,
    percentiles: Sequence[float],
    z_thresholds: Sequence[float],
    calibration_path: str | Path | None = None,
) -> list[SweepSpec]:
    modes = ["percentile", "z_score"] if threshold_mode == "both" else [threshold_mode]
    specs: list[SweepSpec] = []
    percentile_thresholds = None
    if "percentile" in modes:
        if calibration_path is None:
            raise ValueError("calibration_path is required for percentile sweeps")
        percentile_thresholds = load_percentile_thresholds(calibration_path)

    for mode in modes:
        if mode == "percentile":
            assert percentile_thresholds is not None
            for percentile in percentiles:
                key = format_float(percentile)
                if key not in percentile_thresholds:
                    available = ", ".join(sorted(percentile_thresholds)) or "none"
                    raise KeyError(
                        f"percentile {key} not found in {calibration_path}; "
                        f"available percentiles: {available}"
                    )
                specs.append(
                    SweepSpec(
                        threshold_mode="percentile",
                        label=f"p{key}",
                        percentile=float(percentile),
                        calibrated_threshold=percentile_thresholds[key],
                    )
                )
        elif mode == "z_score":
            for threshold in z_thresholds:
                specs.append(
                    SweepSpec(
                        threshold_mode="z_score",
                        label=f"z{format_float(threshold)}",
                        threshold=float(threshold),
                    )
                )
        else:
            raise ValueError("threshold_mode must be percentile, z_score, or both")
    return specs


def parse_run_output_dir(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("Wrote metrics to "):
            return Path(line.removeprefix("Wrote metrics to ").strip())
    raise RuntimeError("run_baseline output did not include an output directory")


def load_eval_summary(run_dir: str | Path) -> dict[str, Any]:
    with (Path(run_dir) / "eval_summary.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def row_from_summary(spec: SweepSpec, run_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "threshold_mode": spec.threshold_mode,
        "label": spec.label,
        "threshold": spec.threshold if spec.threshold is not None else "",
        "threshold_percentile": spec.percentile if spec.percentile is not None else "",
        "calibrated_threshold": spec.calibrated_threshold
        if spec.calibrated_threshold is not None
        else "",
        "run_dir": str(run_dir),
        "env": summary.get("env", ""),
        "method": summary.get("method", ""),
        "seed": summary.get("seed", ""),
        "episodes": summary.get("episodes", ""),
        "success_rate": summary.get("success_rate", ""),
        "mean_episode_return": summary.get("mean_episode_return", ""),
        "median_episode_return": summary.get("median_episode_return", ""),
        "mean_episode_length": summary.get("mean_episode_length", ""),
        "mean_nfe_per_action": summary.get("mean_nfe_per_action", ""),
        "mean_inference_wall_time_per_action": summary.get(
            "mean_inference_wall_time_per_action", ""
        ),
        "mean_total_wall_time_per_action": summary.get(
            "mean_total_wall_time_per_action", ""
        ),
        "replan_rate": summary.get("replan_rate", ""),
        "forced_replan_rate": summary.get("forced_replan_rate", ""),
        "learned_replan_rate": summary.get("learned_replan_rate", ""),
        "mean_discarded_actions_per_episode": summary.get(
            "mean_discarded_actions_per_episode", ""
        ),
        "mean_buffer_remaining_when_replanned": summary.get(
            "mean_buffer_remaining_when_replanned", ""
        ),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def numeric_value(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value == "":
        return default
    return float(value)


def select_best_row(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            numeric_value(row, "success_rate"),
            numeric_value(row, "mean_episode_return"),
            -numeric_value(row, "mean_nfe_per_action", default=float("inf")),
        ),
    )


def write_markdown(path: Path, rows: Sequence[dict[str, Any]], best: dict[str, Any] | None) -> None:
    lines = [
        "# TD-Error Threshold Sweep",
        "",
        "This file is generated by `scripts/run_td_threshold_sweep.py`.",
        "",
    ]
    if best is not None:
        lines.extend(
            [
                "## Best Debug Row",
                "",
                (
                    f"Selected by success rate, then mean return, then lower NFE/action: "
                    f"`{best['label']}`."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Results",
            "",
            "| Label | Mode | Threshold | Success | Return | NFE/Action | Replan | Learned Replan | Discarded/Episode |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        threshold = row["calibrated_threshold"] or row["threshold"]
        lines.append(
            "| {label} | {mode} | {threshold} | {success} | {ret} | {nfe} | "
            "{replan} | {learned} | {discarded} |".format(
                label=row["label"],
                mode=row["threshold_mode"],
                threshold=threshold,
                success=row["success_rate"],
                ret=row["mean_episode_return"],
                nfe=row["mean_nfe_per_action"],
                replan=row["replan_rate"],
                learned=row["learned_replan_rate"],
                discarded=row["mean_discarded_actions_per_episode"],
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_sweep_outputs(output_dir: str | Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    best = select_best_row(rows)
    write_csv(output_path / "sweep_results.csv", rows)
    summary = {
        "num_runs": len(rows),
        "best_label": best["label"] if best else None,
        "best_run_dir": best["run_dir"] if best else None,
        "rows": list(rows),
    }
    with (output_path / "sweep_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_markdown(output_path / "sweep_results.md", rows, best)
    return summary


def build_run_command(args: argparse.Namespace, spec: SweepSpec, runs_root: Path) -> list[str]:
    command = [
        sys.executable,
        str(RUN_BASELINE),
        "--env",
        args.env,
        "--method",
        "td_error_replan",
        "--chunk_horizon",
        str(args.chunk_horizon),
        "--denoise_steps",
        str(args.denoise_steps),
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--debug",
        str(args.debug).lower(),
        "--td_critic_checkpoint",
        str(args.critic_checkpoint),
        "--td_threshold_mode",
        spec.threshold_mode,
        "--td_gamma",
        str(args.td_gamma),
        "--output_root",
        str(runs_root),
    ]
    if args.max_episode_steps is not None:
        command.extend(["--max_episode_steps", str(args.max_episode_steps)])
    if spec.threshold_mode == "percentile":
        command.extend(
            [
                "--td_threshold_percentile",
                str(spec.percentile),
                "--td_calibration",
                str(args.calibration),
            ]
        )
    else:
        command.extend(["--td_threshold", str(spec.threshold)])
    return command


def run_one_spec(args: argparse.Namespace, spec: SweepSpec, runs_root: Path) -> dict[str, Any]:
    command = build_run_command(args, spec, runs_root)
    print(f"Running {spec.label}: {' '.join(command)}", flush=True)
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"run failed for {spec.label} with exit code {result.returncode}")
    run_dir = parse_run_output_dir(result.stdout)
    summary = load_eval_summary(run_dir)
    return row_from_summary(spec, run_dir, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        choices=["robomimic_lift", "robomimic_can", "robomimic_square"],
        default="robomimic_lift",
    )
    parser.add_argument(
        "--threshold_mode", choices=["percentile", "z_score", "both"], default="percentile"
    )
    parser.add_argument("--percentiles", type=parse_float_list, default=(70, 80, 90, 95))
    parser.add_argument("--z_thresholds", type=parse_float_list, default=(0.5, 1.0, 1.5, 2.0))
    parser.add_argument("--critic_checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--td_gamma", type=float, default=0.99)
    parser.add_argument("--chunk_horizon", type=int, default=4)
    parser.add_argument("--denoise_steps", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", type=str_to_bool, default=False)
    parser.add_argument("--max_episode_steps", type=int, default=None)
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "outputs" / "evals")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.critic_checkpoint.exists():
        raise FileNotFoundError(f"critic checkpoint not found: {args.critic_checkpoint}")
    if args.threshold_mode in {"percentile", "both"}:
        if args.calibration is None:
            raise ValueError("--calibration is required for percentile sweeps")
        if not args.calibration.exists():
            raise FileNotFoundError(f"calibration file not found: {args.calibration}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = args.output_root / f"{timestamp}_{args.env}_td_threshold_sweep_seed{args.seed}"
    runs_root = sweep_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=False)
    config = {
        key: str(value) if isinstance(value, Path) else list(value)
        if isinstance(value, tuple)
        else value
        for key, value in vars(args).items()
    }
    OmegaConf.save(OmegaConf.create(config), sweep_dir / "sweep_config.yaml")

    specs = build_sweep_specs(
        threshold_mode=args.threshold_mode,
        percentiles=args.percentiles,
        z_thresholds=args.z_thresholds,
        calibration_path=args.calibration,
    )
    rows = [run_one_spec(args, spec, runs_root) for spec in specs]
    summary = write_sweep_outputs(sweep_dir, rows)
    print(f"Wrote sweep outputs to {sweep_dir}")
    print(OmegaConf.to_yaml(OmegaConf.create(summary), resolve=True))


if __name__ == "__main__":
    main()
