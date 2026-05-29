#!/usr/bin/env python3
"""Run a small PPO replanning cost-penalty sweep."""

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
TRAIN_REPLAN_PPO = REPO_ROOT / "scripts" / "train_replan_ppo.py"


@dataclass(frozen=True)
class CostSweepSpec:
    lambda_C: float
    lambda_D: float
    uncertainty_replan_bonus: float = 0.0

    @property
    def label(self) -> str:
        label = f"c{format_float(self.lambda_C)}_d{format_float(self.lambda_D)}"
        if self.uncertainty_replan_bonus != 0.0:
            label += f"_b{format_float(self.uncertainty_replan_bonus)}"
        return label


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in value.split(",") if part)
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one value")
    return values


def format_float(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def build_sweep_specs(
    lambda_C_values: Sequence[float],
    lambda_D_values: Sequence[float],
    uncertainty_replan_bonus_values: Sequence[float] = (0.0,),
) -> list[CostSweepSpec]:
    return [
        CostSweepSpec(
            lambda_C=float(lambda_C),
            lambda_D=float(lambda_D),
            uncertainty_replan_bonus=float(uncertainty_replan_bonus),
        )
        for lambda_C in lambda_C_values
        for lambda_D in lambda_D_values
        for uncertainty_replan_bonus in uncertainty_replan_bonus_values
    ]


def parse_training_output_dir(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("Wrote PPO training run to "):
            return Path(line.removeprefix("Wrote PPO training run to ").strip())
    raise RuntimeError("train_replan_ppo output did not include an output directory")


def load_train_summary(run_dir: str | Path) -> dict[str, Any]:
    with (Path(run_dir) / "train_summary.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def row_from_summary(
    spec: CostSweepSpec,
    run_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    train = summary.get("train_summary", {})
    eval_summary = summary.get("eval_summary", {})
    last_update = summary.get("last_update", {})
    return {
        "label": spec.label,
        "lambda_C": spec.lambda_C,
        "lambda_D": spec.lambda_D,
        "uncertainty_replan_bonus": spec.uncertainty_replan_bonus,
        "run_dir": str(run_dir),
        "env": summary.get("env", ""),
        "seed": summary.get("seed", ""),
        "actual_env_steps": summary.get("actual_env_steps", ""),
        "updates": summary.get("updates", ""),
        "train_episodes": train.get("episodes", ""),
        "train_success_rate": train.get("success_rate", ""),
        "train_mean_episode_return": train.get("mean_episode_return", ""),
        "train_mean_high_level_return": train.get("mean_high_level_return", ""),
        "train_mean_nfe_per_action": train.get("mean_nfe_per_action", ""),
        "train_replan_rate": train.get("replan_rate", ""),
        "train_forced_replan_rate": train.get("forced_replan_rate", ""),
        "train_learned_replan_rate": train.get("learned_replan_rate", ""),
        "train_discarded_actions_per_episode": train.get(
            "mean_discarded_actions_per_episode", ""
        ),
        "eval_episodes": eval_summary.get("episodes", ""),
        "eval_success_rate": eval_summary.get("success_rate", ""),
        "eval_mean_episode_return": eval_summary.get("mean_episode_return", ""),
        "eval_mean_high_level_return": eval_summary.get("mean_high_level_return", ""),
        "eval_mean_nfe_per_action": eval_summary.get("mean_nfe_per_action", ""),
        "eval_replan_rate": eval_summary.get("replan_rate", ""),
        "eval_forced_replan_rate": eval_summary.get("forced_replan_rate", ""),
        "eval_learned_replan_rate": eval_summary.get("learned_replan_rate", ""),
        "eval_discarded_actions_per_episode": eval_summary.get(
            "mean_discarded_actions_per_episode", ""
        ),
        "last_update_continue_actions": last_update.get("continue_actions", ""),
        "last_update_replan_actions": last_update.get("replan_actions", ""),
        "last_update_forced_replans": last_update.get("forced_replans", ""),
        "last_update_loss": last_update.get("loss", ""),
        "last_update_entropy": last_update.get("entropy", ""),
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
            numeric_value(row, "eval_success_rate"),
            numeric_value(row, "eval_mean_episode_return"),
            numeric_value(row, "eval_mean_high_level_return"),
            -numeric_value(row, "eval_mean_nfe_per_action", default=float("inf")),
            -numeric_value(row, "eval_replan_rate", default=float("inf")),
        ),
    )


def write_markdown(
    path: Path,
    rows: Sequence[dict[str, Any]],
    best: dict[str, Any] | None,
) -> None:
    lines = [
        "# PPO Replan Cost Sweep",
        "",
        "This file is generated by `scripts/run_ppo_cost_sweep.py`.",
        "",
    ]
    if best is not None:
        lines.extend(
            [
                "## Best Debug Row",
                "",
                (
                    "Selected by eval success, eval return, eval high-level return, "
                    f"lower NFE/action, then lower replan rate: `{best['label']}`."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Results",
            "",
            "| Label | lambda_C | lambda_D | Bonus | Eval Success | Eval Return | Eval HL Return | Eval NFE/Action | Eval Replan | Train Replan | Last Continue | Last Replan |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {label} | {lambda_C} | {lambda_D} | {bonus} | {success} | {ret} | {hl} | "
            "{nfe} | {eval_replan} | {train_replan} | {cont} | {replan} |".format(
                label=row["label"],
                lambda_C=row["lambda_C"],
                lambda_D=row["lambda_D"],
                bonus=row.get("uncertainty_replan_bonus", 0.0),
                success=row["eval_success_rate"],
                ret=row["eval_mean_episode_return"],
                hl=row["eval_mean_high_level_return"],
                nfe=row["eval_mean_nfe_per_action"],
                eval_replan=row["eval_replan_rate"],
                train_replan=row["train_replan_rate"],
                cont=row["last_update_continue_actions"],
                replan=row["last_update_replan_actions"],
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_sweep_outputs(
    output_dir: str | Path,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
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


def build_run_command(
    args: argparse.Namespace,
    spec: CostSweepSpec,
    runs_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_REPLAN_PPO),
        "--env",
        args.env,
        "--seed",
        str(args.seed),
        "--total_env_steps",
        str(args.total_env_steps),
        "--rollout_steps",
        str(args.rollout_steps),
        "--eval_episodes",
        str(args.eval_episodes),
        "--chunk_horizon",
        str(args.chunk_horizon),
        "--denoise_steps",
        str(args.denoise_steps),
        "--hidden_dims",
        args.hidden_dims,
        "--learning_rate",
        str(args.learning_rate),
        "--minibatch_size",
        str(args.minibatch_size),
        "--update_epochs",
        str(args.update_epochs),
        "--lambda_C",
        str(spec.lambda_C),
        "--lambda_D",
        str(spec.lambda_D),
        "--uncertainty_replan_bonus",
        str(spec.uncertainty_replan_bonus),
        "--output_root",
        str(runs_root),
    ]
    if args.max_episode_steps is not None:
        command.extend(["--max_episode_steps", str(args.max_episode_steps)])
    if args.td_critic_checkpoint is not None:
        command.extend(["--td_critic_checkpoint", str(args.td_critic_checkpoint)])
    if args.td_gamma is not None:
        command.extend(["--td_gamma", str(args.td_gamma)])
    if args.device is not None:
        command.extend(["--device", args.device])
    return command


def run_one_spec(
    args: argparse.Namespace,
    spec: CostSweepSpec,
    runs_root: Path,
) -> dict[str, Any]:
    command = build_run_command(args, spec, runs_root)
    print(f"Running {spec.label}: {' '.join(command)}", flush=True)
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"run failed for {spec.label} with exit code {result.returncode}"
        )
    run_dir = parse_training_output_dir(result.stdout)
    summary = load_train_summary(run_dir)
    return row_from_summary(spec, run_dir, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        choices=["fake", "robomimic_lift", "robomimic_can", "robomimic_square"],
        default="robomimic_lift",
    )
    parser.add_argument(
        "--lambda_C_values",
        type=parse_float_list,
        default=(0.0, 0.001, 0.003, 0.01),
    )
    parser.add_argument(
        "--lambda_D_values",
        type=parse_float_list,
        default=(0.0, 0.01, 0.03, 0.1),
    )
    parser.add_argument(
        "--uncertainty_replan_bonus_values",
        type=parse_float_list,
        default=(0.0,),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_env_steps", type=int, default=64)
    parser.add_argument("--rollout_steps", type=int, default=32)
    parser.add_argument("--max_episode_steps", type=int, default=None)
    parser.add_argument("--eval_episodes", type=int, default=2)
    parser.add_argument("--chunk_horizon", type=int, default=4)
    parser.add_argument("--denoise_steps", type=int, default=20)
    parser.add_argument("--hidden_dims", type=str, default="256,256")
    parser.add_argument("--learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--update_epochs", type=int, default=5)
    parser.add_argument("--td_critic_checkpoint", type=Path, default=None)
    parser.add_argument("--td_gamma", type=float, default=0.99)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=REPO_ROOT / "outputs" / "evals",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = args.output_root / f"{timestamp}_{args.env}_ppo_cost_sweep_seed{args.seed}"
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
        args.lambda_C_values,
        args.lambda_D_values,
        args.uncertainty_replan_bonus_values,
    )
    rows = [run_one_spec(args, spec, runs_root) for spec in specs]
    summary = write_sweep_outputs(sweep_dir, rows)
    print(f"Wrote PPO cost sweep outputs to {sweep_dir}")
    print(OmegaConf.to_yaml(OmegaConf.create(summary), resolve=True))


if __name__ == "__main__":
    main()
