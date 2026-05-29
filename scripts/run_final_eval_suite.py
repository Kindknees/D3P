#!/usr/bin/env python3
"""Run final-style adaptive replanning evaluations and generate analysis artifacts."""

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
EVAL_POLICY = REPO_ROOT / "scripts" / "eval_policy.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptive_diffusion.plots import (  # noqa: E402
    LabeledRun,
    load_rows,
    write_analysis_outputs,
    write_timeline_outputs,
)

SUPPORTED_BASELINE_ENVS = ("robomimic_lift", "robomimic_can", "robomimic_square")
SUPPORTED_PPO_ENVS = ("fake", "robomimic_lift", "robomimic_can", "robomimic_square")
BASELINE_METHODS = (
    "fixed_chunk",
    "high_compute",
    "low_compute",
    "td_error_replan",
    "denoise_only",
)


@dataclass(frozen=True)
class PPOCheckpointSpec:
    label: str
    checkpoint: Path
    env: str | None = None


@dataclass(frozen=True)
class TDArtifactSpec:
    env: str
    critic_checkpoint: Path
    calibration: Path


@dataclass(frozen=True)
class RuntimeArtifactSpec:
    env: str
    base_policy_checkpoint: Path
    normalization_path: Path


@dataclass(frozen=True)
class EvalJob:
    label: str
    env: str
    seed: int
    kind: str
    command: tuple[str, ...]


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_int_csv(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one integer")
    return values


def parse_float_csv(value: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one float")
    return values


def parse_ppo_checkpoint(value: str) -> PPOCheckpointSpec:
    if "=" not in value:
        path = Path(value)
        return PPOCheckpointSpec(label=path.stem, checkpoint=path)
    label_part, path = value.split("=", 1)
    env = None
    label = label_part.strip()
    if ":" in label:
        env_part, label = label.split(":", 1)
        env = env_part.strip() or None
        label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("checkpoint label cannot be empty")
    return PPOCheckpointSpec(label=label, checkpoint=Path(path), env=env)


def parse_td_artifact(value: str) -> TDArtifactSpec:
    if ":" not in value or "=" not in value:
        raise argparse.ArgumentTypeError(
            "TD artifact must use ENV:critic_checkpoint=calibration_json"
        )
    env, remainder = value.split(":", 1)
    critic, calibration = remainder.split("=", 1)
    env = env.strip()
    if not env:
        raise argparse.ArgumentTypeError("TD artifact env cannot be empty")
    return TDArtifactSpec(
        env=env,
        critic_checkpoint=Path(critic),
        calibration=Path(calibration),
    )


def td_artifact_for_env(args: argparse.Namespace, env: str) -> tuple[Path | None, Path | None]:
    for artifact in getattr(args, "td_artifacts", ()):
        if artifact.env == env:
            return artifact.critic_checkpoint, artifact.calibration
    return args.td_critic_checkpoint, args.td_calibration


def parse_runtime_artifact(value: str) -> RuntimeArtifactSpec:
    if ":" not in value or "=" not in value:
        raise argparse.ArgumentTypeError(
            "runtime artifact must use ENV:base_policy_checkpoint=normalization_npz"
        )
    env, remainder = value.split(":", 1)
    checkpoint, normalization = remainder.split("=", 1)
    env = env.strip()
    if not env:
        raise argparse.ArgumentTypeError("runtime artifact env cannot be empty")
    return RuntimeArtifactSpec(
        env=env,
        base_policy_checkpoint=Path(checkpoint),
        normalization_path=Path(normalization),
    )


def runtime_artifact_for_env(
    args: argparse.Namespace,
    env: str,
) -> tuple[Path | None, Path | None, Path | None]:
    for artifact in getattr(args, "runtime_artifacts", ()):
        if artifact.env == env:
            return (
                artifact.base_policy_checkpoint,
                artifact.normalization_path,
                getattr(args, "env_meta_path", None),
            )
    return (
        getattr(args, "base_policy_checkpoint", None),
        getattr(args, "normalization_path", None),
        getattr(args, "env_meta_path", None),
    )


def format_percentile(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace(".", "p")


def parse_output_dir(stdout: str) -> Path:
    prefixes = ("Wrote metrics to ", "Wrote eval to ")
    for line in stdout.splitlines():
        for prefix in prefixes:
            if line.startswith(prefix):
                return Path(line.removeprefix(prefix).strip())
    raise RuntimeError("evaluation output did not include a run directory")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_args(args: argparse.Namespace) -> None:
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.timeline_episodes <= 0:
        raise ValueError("--timeline_episodes must be positive")
    if args.max_jobs is not None and args.max_jobs <= 0:
        raise ValueError("--max_jobs must be positive when provided")
    if not args.baseline_methods and not args.ppo_checkpoints:
        raise ValueError("provide at least one baseline method or --ppo_checkpoint")
    invalid_methods = set(args.baseline_methods) - set(BASELINE_METHODS)
    if invalid_methods:
        raise ValueError("unsupported baseline methods: " + ",".join(sorted(invalid_methods)))
    for env in args.envs:
        if env not in SUPPORTED_PPO_ENVS:
            raise ValueError(
                f"unsupported env {env}; current local runners support "
                f"{','.join(SUPPORTED_PPO_ENVS)}."
            )
        if args.baseline_methods and env not in SUPPORTED_BASELINE_ENVS:
            raise ValueError(f"baseline methods are not supported for env {env}")
    needs_td_critic = {"td_error_replan", "denoise_only"} & set(args.baseline_methods)
    if needs_td_critic:
        for env in args.envs:
            critic_checkpoint, calibration = td_artifact_for_env(args, env)
            if critic_checkpoint is None:
                methods = ",".join(sorted(needs_td_critic))
                raise ValueError(
                    f"{methods} for {env} requires --td_critic_checkpoint or --td_artifact"
                )
            if not critic_checkpoint.exists():
                raise FileNotFoundError(f"TD critic checkpoint not found: {critic_checkpoint}")
            if "td_error_replan" in args.baseline_methods:
                if calibration is None:
                    raise ValueError(
                        f"td_error_replan for {env} requires --td_calibration or --td_artifact"
                    )
                if not calibration.exists():
                    raise FileNotFoundError(f"TD calibration not found: {calibration}")
    for artifact in getattr(args, "runtime_artifacts", ()):
        if artifact.env not in SUPPORTED_PPO_ENVS:
            raise ValueError(f"unsupported runtime artifact env: {artifact.env}")
        if not artifact.base_policy_checkpoint.exists():
            raise FileNotFoundError(
                f"base policy checkpoint not found: {artifact.base_policy_checkpoint}"
            )
        if not artifact.normalization_path.exists():
            raise FileNotFoundError(
                f"normalization file not found: {artifact.normalization_path}"
            )
    if (
        getattr(args, "base_policy_checkpoint", None) is not None
        and not args.base_policy_checkpoint.exists()
    ):
        raise FileNotFoundError(f"base policy checkpoint not found: {args.base_policy_checkpoint}")
    if (
        getattr(args, "normalization_path", None) is not None
        and not args.normalization_path.exists()
    ):
        raise FileNotFoundError(f"normalization file not found: {args.normalization_path}")
    if getattr(args, "env_meta_path", None) is not None and not args.env_meta_path.exists():
        raise FileNotFoundError(f"env meta file not found: {args.env_meta_path}")
    for spec in args.ppo_checkpoints:
        if spec.env is not None and spec.env not in SUPPORTED_PPO_ENVS:
            raise ValueError(f"unsupported PPO checkpoint env: {spec.env}")
        if not spec.checkpoint.exists():
            raise FileNotFoundError(f"PPO checkpoint not found: {spec.checkpoint}")


def build_baseline_command(
    args: argparse.Namespace,
    method: str,
    env: str,
    seed: int,
    runs_root: Path,
    python_executable: str,
) -> tuple[str, ...]:
    command = [
        python_executable,
        str(RUN_BASELINE),
        "--env",
        env,
        "--method",
        method,
        "--episodes",
        str(args.episodes),
        "--seed",
        str(seed),
        "--output_root",
        str(runs_root),
    ]
    if args.max_episode_steps is not None:
        command.extend(["--max_episode_steps", str(args.max_episode_steps)])
    base_policy_checkpoint, normalization_path, env_meta_path = runtime_artifact_for_env(args, env)
    if base_policy_checkpoint is not None:
        command.extend(["--base_policy_checkpoint", str(base_policy_checkpoint)])
    if normalization_path is not None:
        command.extend(["--normalization_path", str(normalization_path)])
    if env_meta_path is not None:
        command.extend(["--env_meta_path", str(env_meta_path)])
    if method in {"td_error_replan", "denoise_only"}:
        critic_checkpoint, calibration = td_artifact_for_env(args, env)
        command.extend(
            [
                "--td_critic_checkpoint",
                str(critic_checkpoint),
                "--td_gamma",
                str(args.td_gamma),
            ]
        )
        if method == "td_error_replan":
            command.extend(
                [
                    "--td_threshold_mode",
                    "percentile",
                    "--td_threshold_percentile",
                    str(args.td_threshold_percentile),
                    "--td_calibration",
                    str(calibration),
                ]
            )
        if method == "denoise_only":
            candidates = ",".join(
                str(value)
                for value in getattr(args, "denoise_candidates", (4, 8, 12, 20))
            )
            thresholds = ",".join(
                str(value)
                for value in getattr(
                    args, "denoise_uncertainty_thresholds", (0.5, 1.0, 1.5)
                )
            )
            command.extend(
                [
                    "--denoise_candidates",
                    candidates,
                    "--denoise_uncertainty_thresholds",
                    thresholds,
                ]
            )
    return tuple(command)


def build_ppo_command(
    args: argparse.Namespace,
    spec: PPOCheckpointSpec,
    env: str,
    seed: int,
    runs_root: Path,
    python_executable: str,
) -> tuple[str, ...]:
    command = [
        python_executable,
        str(EVAL_POLICY),
        "--env",
        env,
        "--checkpoint",
        str(spec.checkpoint),
        "--episodes",
        str(args.episodes),
        "--seed",
        str(seed),
        "--deterministic",
        str(args.deterministic).lower(),
        "--output_root",
        str(runs_root),
    ]
    if args.max_episode_steps is not None:
        command.extend(["--max_episode_steps", str(args.max_episode_steps)])
    base_policy_checkpoint, normalization_path, env_meta_path = runtime_artifact_for_env(args, env)
    if base_policy_checkpoint is not None:
        command.extend(["--base_policy_checkpoint", str(base_policy_checkpoint)])
    if normalization_path is not None:
        command.extend(["--normalization_path", str(normalization_path)])
    if env_meta_path is not None:
        command.extend(["--env_meta_path", str(env_meta_path)])
    if args.td_critic_checkpoint is not None:
        command.extend(["--td_critic_checkpoint", str(args.td_critic_checkpoint)])
    if args.td_gamma is not None:
        command.extend(["--td_gamma", str(args.td_gamma)])
    if args.device is not None:
        command.extend(["--device", args.device])
    return tuple(command)


def build_eval_jobs(
    args: argparse.Namespace,
    runs_root: Path,
    python_executable: str | None = None,
) -> list[EvalJob]:
    executable = python_executable or sys.executable
    jobs: list[EvalJob] = []
    for env in args.envs:
        for seed in args.seeds:
            for method in args.baseline_methods:
                label = method
                if method == "td_error_replan":
                    label = f"td_error_p{format_percentile(args.td_threshold_percentile)}"
                jobs.append(
                    EvalJob(
                        label=label,
                        env=env,
                        seed=int(seed),
                        kind="baseline",
                        command=build_baseline_command(args, method, env, seed, runs_root, executable),
                    )
                )
            for spec in args.ppo_checkpoints:
                if spec.env is not None and spec.env != env:
                    continue
                jobs.append(
                    EvalJob(
                        label=spec.label,
                        env=env,
                        seed=int(seed),
                        kind="ppo",
                        command=build_ppo_command(args, spec, env, seed, runs_root, executable),
                    )
                )
    return jobs


def run_job(job: EvalJob) -> tuple[Path, str]:
    print(f"Running {job.label} env={job.env} seed={job.seed}: {' '.join(job.command)}", flush=True)
    result = subprocess.run(job.command, check=False, capture_output=True, text=True)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"job failed label={job.label} env={job.env} seed={job.seed} "
            f"with exit code {result.returncode}"
        )
    return parse_output_dir(result.stdout), result.stdout


def command_row(job: EvalJob) -> dict[str, Any]:
    return {
        "label": job.label,
        "env": job.env,
        "seed": job.seed,
        "kind": job.kind,
        "command": " ".join(job.command),
    }


def result_row(job: EvalJob, run_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": job.label,
        "env": job.env,
        "seed": job.seed,
        "kind": job.kind,
        "run_dir": str(run_dir),
        "success_rate": row.get("success_rate", ""),
        "mean_episode_return": row.get("mean_episode_return", ""),
        "mean_high_level_return": row.get("mean_high_level_return", ""),
        "mean_nfe_per_action": row.get("mean_nfe_per_action", ""),
        "replan_rate": row.get("replan_rate", ""),
        "learned_replan_rate": row.get("learned_replan_rate", ""),
        "mean_discarded_actions_per_episode": row.get("mean_discarded_actions_per_episode", ""),
        "source_path": row.get("source_path", ""),
    }


def job_key(job: EvalJob) -> tuple[str, str, int, str]:
    return (job.label, job.env, int(job.seed), job.kind)


def row_key(row: dict[str, Any]) -> tuple[str, str, int, str] | None:
    seed = str(row.get("seed", ""))
    if not seed.lstrip("-").isdigit():
        return None
    label = str(row.get("label", ""))
    env = str(row.get("env", ""))
    kind = str(row.get("kind", ""))
    if not label or not env or not kind:
        return None
    return (label, env, int(seed), kind)


def row_has_valid_run(row: dict[str, Any]) -> bool:
    run_dir = Path(str(row.get("run_dir", "")))
    source_path = Path(str(row.get("source_path") or run_dir / "eval_summary.json"))
    return run_dir.is_dir() and source_path.exists() and source_path.stat().st_size > 0


def completed_rows_for_jobs(jobs: Sequence[EvalJob], results_csv: Path) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    wanted = {job_key(job) for job in jobs}
    completed: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for row in read_csv_rows(results_csv):
        key = row_key(row)
        if key is None or key not in wanted:
            continue
        if not row_has_valid_run(row):
            continue
        completed[key] = dict(row)
    return completed


def pending_jobs(jobs: Sequence[EvalJob], completed: dict[tuple[str, str, int, str], dict[str, Any]]) -> list[EvalJob]:
    return [job for job in jobs if job_key(job) not in completed]


def labeled_run_from_row(row: dict[str, Any]) -> LabeledRun:
    return LabeledRun(str(row["label"]), Path(str(row["run_dir"])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=parse_csv, default=("robomimic_lift",))
    parser.add_argument("--seeds", type=parse_int_csv, default=(0, 1, 2))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_episode_steps", type=int, default=None)
    parser.add_argument(
        "--baseline_methods",
        type=parse_csv,
        default=("fixed_chunk", "td_error_replan", "denoise_only"),
        help=(
            "Comma-separated baseline methods. Defaults to final evidence baselines; "
            "use an empty string to disable."
        ),
    )
    parser.add_argument(
        "--ppo_checkpoint",
        action="append",
        type=parse_ppo_checkpoint,
        dest="ppo_checkpoints",
        default=[],
        help="LABEL=checkpoint.pt or ENV:LABEL=checkpoint.pt PPO eval target. Can be repeated.",
    )
    parser.add_argument("--td_critic_checkpoint", type=Path, default=None)
    parser.add_argument("--td_calibration", type=Path, default=None)
    parser.add_argument(
        "--td_artifact",
        action="append",
        type=parse_td_artifact,
        dest="td_artifacts",
        default=[],
        help="ENV:critic_checkpoint=calibration_json TD-error artifacts. Can be repeated.",
    )
    parser.add_argument("--td_threshold_percentile", type=float, default=95.0)
    parser.add_argument("--td_gamma", type=float, default=0.99)
    parser.add_argument(
        "--denoise_candidates",
        type=parse_int_csv,
        default=(4, 8, 12, 20),
    )
    parser.add_argument(
        "--denoise_uncertainty_thresholds",
        type=parse_float_csv,
        default=(0.5, 1.0, 1.5),
    )
    parser.add_argument("--base_policy_checkpoint", type=Path, default=None)
    parser.add_argument("--normalization_path", type=Path, default=None)
    parser.add_argument("--env_meta_path", type=Path, default=None)
    parser.add_argument(
        "--runtime_artifact",
        action="append",
        type=parse_runtime_artifact,
        dest="runtime_artifacts",
        default=[],
        help="ENV:base_policy_checkpoint=normalization_npz runtime artifacts. Can be repeated.",
    )
    parser.add_argument("--deterministic", type=lambda value: str(value).lower() not in {"0", "false", "no", "n"}, default=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--timeline_episodes", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--suite_dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_jobs", type=int, default=None)
    parser.add_argument("--python_executable", type=str, default=sys.executable)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=REPO_ROOT / "outputs" / "evals",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_label = "_".join(args.envs)
    suite_dir = args.suite_dir or args.output_root / f"{timestamp}_{env_label}_final_eval_suite"
    runs_root = suite_dir / "runs"
    plots_dir = suite_dir / "plots"
    if suite_dir.exists() and not args.resume:
        raise FileExistsError(f"suite_dir already exists; use --resume to continue: {suite_dir}")
    runs_root.mkdir(parents=True, exist_ok=args.resume)

    jobs = build_eval_jobs(args, runs_root, python_executable=args.python_executable)
    if not jobs:
        raise ValueError("final eval suite has no jobs after env-specific filtering")
    config = {
        key: [str(item) for item in value]
        if isinstance(value, tuple)
        else [
            f"{spec.env + ':' if spec.env else ''}{spec.label}={spec.checkpoint}"
            for spec in value
        ]
        if key == "ppo_checkpoints"
        else [
            f"{artifact.env}:{artifact.critic_checkpoint}={artifact.calibration}"
            for artifact in value
        ]
        if key == "td_artifacts"
        else [
            f"{artifact.env}:{artifact.base_policy_checkpoint}={artifact.normalization_path}"
            for artifact in value
        ]
        if key == "runtime_artifacts"
        else str(value)
        if isinstance(value, Path)
        else value
        for key, value in vars(args).items()
    }
    config["suite_dir"] = str(suite_dir)
    config["runs_root"] = str(runs_root)
    OmegaConf.save(OmegaConf.create(config), suite_dir / "suite_config.yaml")
    write_csv(suite_dir / "suite_commands.csv", [command_row(job) for job in jobs])

    completed = completed_rows_for_jobs(jobs, suite_dir / "suite_results.csv") if args.resume else {}
    jobs_to_run = pending_jobs(jobs, completed)
    if args.max_jobs is not None:
        jobs_to_run = jobs_to_run[: args.max_jobs]

    labeled_runs: list[LabeledRun] = []
    result_rows: list[dict[str, Any]] = []
    if not args.dry_run:
        result_rows.extend(completed[job_key(job)] for job in jobs if job_key(job) in completed)
        for job in jobs_to_run:
            run_dir, _stdout = run_job(job)
            row = load_rows([LabeledRun(job.label, run_dir)])[0]
            result_rows.append(result_row(job, run_dir, row))
        write_csv(suite_dir / "suite_results.csv", result_rows)
        labeled_runs = [labeled_run_from_row(row) for row in result_rows]
        if labeled_runs:
            rows = load_rows(labeled_runs)
            analysis_summary = write_analysis_outputs(plots_dir, rows)
            timeline_paths = write_timeline_outputs(
                plots_dir,
                labeled_runs,
                max_episodes_per_run=args.timeline_episodes,
            )
            analysis_summary["outputs"]["replan_timeline_plots"] = timeline_paths
            analysis_summary["outputs"]["timeline_manifest"] = str(plots_dir / "timeline_manifest.csv")
            with (plots_dir / "analysis_summary.json").open("w", encoding="utf-8") as f:
                json.dump(analysis_summary, f, indent=2)
        else:
            analysis_summary = {}
    else:
        write_csv(suite_dir / "suite_results.csv", [])
        analysis_summary = {}

    summary = {
        "suite_dir": str(suite_dir),
        "dry_run": bool(args.dry_run),
        "num_jobs": len(jobs),
        "num_completed": len(result_rows),
        "num_pending": len(jobs) - len(result_rows),
        "num_ran_this_invocation": 0 if args.dry_run else len(jobs_to_run),
        "suite_complete": len(result_rows) == len(jobs),
        "resumed": bool(args.resume),
        "max_jobs": args.max_jobs,
        "commands_csv": str(suite_dir / "suite_commands.csv"),
        "results_csv": str(suite_dir / "suite_results.csv"),
        "plots_dir": str(plots_dir) if result_rows else None,
        "analysis_summary": analysis_summary,
    }
    with (suite_dir / "suite_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote final eval suite to {suite_dir}")
    print(OmegaConf.to_yaml(OmegaConf.create(summary), resolve=True))


if __name__ == "__main__":
    main()
