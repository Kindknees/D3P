#!/usr/bin/env python3
"""Check adaptive replanning final-evaluation readiness and write a run plan."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_baseline import TASKS  # noqa: E402
from scripts.run_final_eval_suite import (  # noqa: E402
    PPOCheckpointSpec,
    RuntimeArtifactSpec,
    TDArtifactSpec,
    parse_ppo_checkpoint,
    parse_runtime_artifact,
    parse_td_artifact,
)


DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_ENVS = ("robomimic_lift", "robomimic_can", "robomimic_square")


@dataclass(frozen=True)
class RuntimePaths:
    base_policy_checkpoint: Path
    normalization_path: Path
    env_meta_path: Path
    uses_runtime_artifact: bool = False


@dataclass(frozen=True)
class ArtifactCheck:
    label: str
    path: Path | None
    exists: bool
    valid: bool
    size_bytes: int | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "path": str(self.path) if self.path is not None else None,
            "exists": self.exists,
            "valid": self.valid,
            "size_bytes": self.size_bytes,
            "reason": self.reason,
        }


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_int_csv(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one integer")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=parse_csv, default=DEFAULT_ENVS)
    parser.add_argument("--seeds", type=parse_int_csv, default=DEFAULT_SEEDS)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--td_bootstrap_episodes", type=int, default=3)
    parser.add_argument("--td_bootstrap_max_episode_steps", type=int, default=None)
    parser.add_argument("--td_artifact", action="append", type=parse_td_artifact, default=[])
    parser.add_argument("--ppo_checkpoint", action="append", type=parse_ppo_checkpoint, default=[])
    parser.add_argument(
        "--runtime_artifact",
        action="append",
        type=parse_runtime_artifact,
        default=[],
        help="ENV:base_policy_checkpoint=normalization_npz runtime artifacts. Can be repeated.",
    )
    parser.add_argument("--env_meta_path", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs" / "readiness")
    parser.add_argument("--square_work_root", type=Path, default=Path("/tmp/d3p_square_final"))
    parser.add_argument("--python_executable", type=str, default="python")
    return parser.parse_args()


def default_runtime_paths(env: str, env_meta_override: Path | None = None) -> RuntimePaths:
    if env not in TASKS:
        raise ValueError(f"unsupported env: {env}")
    task = TASKS[env]
    data_dir = Path(os.environ.get("DPPO_DATA_DIR", REPO_ROOT / "data"))
    log_dir = Path(os.environ.get("DPPO_LOG_DIR", REPO_ROOT / "log"))
    return RuntimePaths(
        base_policy_checkpoint=log_dir / task["checkpoint"],
        normalization_path=data_dir / "robomimic" / task["env_name"] / "normalization.npz",
        env_meta_path=env_meta_override
        or REPO_ROOT / "cfg" / "robomimic" / "env_meta" / f"{task['env_name']}.json",
        uses_runtime_artifact=False,
    )


def runtime_paths_for_env(args: argparse.Namespace, env: str) -> RuntimePaths:
    default = default_runtime_paths(env, args.env_meta_path)
    for artifact in args.runtime_artifact:
        if artifact.env == env:
            return RuntimePaths(
                base_policy_checkpoint=artifact.base_policy_checkpoint,
                normalization_path=artifact.normalization_path,
                env_meta_path=default.env_meta_path,
                uses_runtime_artifact=True,
            )
    return default


def td_artifact_for_env(
    artifacts: Sequence[TDArtifactSpec], env: str
) -> TDArtifactSpec | None:
    for artifact in artifacts:
        if artifact.env == env:
            return artifact
    return None


def ppo_checkpoint_for_env(
    checkpoints: Sequence[PPOCheckpointSpec], env: str
) -> PPOCheckpointSpec | None:
    for checkpoint in checkpoints:
        if checkpoint.env == env:
            return checkpoint
    for checkpoint in checkpoints:
        if checkpoint.env is None:
            return checkpoint
    return None


def check_artifact(label: str, path: Path | None) -> ArtifactCheck:
    if path is None:
        return ArtifactCheck(label, path, exists=False, valid=False, size_bytes=None, reason="not_provided")
    if not path.exists():
        return ArtifactCheck(label, path, exists=False, valid=False, size_bytes=None, reason="missing")
    if not path.is_file():
        return ArtifactCheck(label, path, exists=True, valid=False, size_bytes=None, reason="not_file")
    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        return ArtifactCheck(label, path, exists=True, valid=False, size_bytes=size_bytes, reason="empty_file")
    return ArtifactCheck(label, path, exists=True, valid=True, size_bytes=size_bytes, reason="ok")


def shell_join(parts: Sequence[str | Path]) -> str:
    return shlex.join(str(part) for part in parts)


def build_runtime_artifact_arg(env: str, paths: RuntimePaths) -> str:
    return f"{env}:{paths.base_policy_checkpoint}={paths.normalization_path}"


def format_td_artifact(spec: TDArtifactSpec) -> str:
    return f"{spec.env}:{spec.critic_checkpoint}={spec.calibration}"


def format_ppo_checkpoint(spec: PPOCheckpointSpec) -> str:
    prefix = f"{spec.env}:" if spec.env is not None else ""
    return f"{prefix}{spec.label}={spec.checkpoint}"


def format_runtime_artifact(spec: RuntimeArtifactSpec) -> str:
    return f"{spec.env}:{spec.base_policy_checkpoint}={spec.normalization_path}"


def has_td_artifact(specs: Sequence[TDArtifactSpec], env: str) -> bool:
    return any(spec.env == env for spec in specs)


def has_ppo_checkpoint(specs: Sequence[PPOCheckpointSpec], env: str) -> bool:
    return any(spec.env == env for spec in specs)


def has_runtime_artifact(specs: Sequence[RuntimeArtifactSpec], env: str) -> bool:
    return any(spec.env == env for spec in specs)


def build_square_commands(
    args: argparse.Namespace,
    env: str,
    paths: RuntimePaths,
    td_artifact: TDArtifactSpec | None,
    ppo_checkpoint: PPOCheckpointSpec | None,
) -> dict[str, str]:
    task = TASKS[env]
    max_steps = int(args.td_bootstrap_max_episode_steps or task["max_episode_steps"])
    fixed_root = args.square_work_root / "td_data"
    fixed_placeholder = fixed_root / "<fixed_run_dir>"
    train_root = args.square_work_root / "ppo_train"
    suite_root = args.square_work_root / "final_suite"
    runtime_arg = build_runtime_artifact_arg(env, paths)

    collect_fixed = [
        "conda",
        "run",
        "-n",
        "d3p",
        args.python_executable,
        "scripts/run_baseline.py",
        "--env",
        env,
        "--method",
        "fixed_chunk",
        "--episodes",
        str(args.td_bootstrap_episodes),
        "--max_episode_steps",
        str(max_steps),
        "--save_transitions",
        "true",
        "--base_policy_checkpoint",
        paths.base_policy_checkpoint,
        "--normalization_path",
        paths.normalization_path,
        "--env_meta_path",
        paths.env_meta_path,
        "--output_root",
        fixed_root,
    ]
    train_td = [
        "conda",
        "run",
        "-n",
        "d3p",
        args.python_executable,
        "scripts/train_td_critic.py",
        "--dataset",
        fixed_placeholder / "transitions.npz",
        "--output_dir",
        fixed_placeholder / "td_critic",
        "--epochs",
        "10",
        "--batch_size",
        "64",
        "--hidden_dims",
        "64",
        "--device",
        "cpu",
    ]
    calibrate_td = [
        "conda",
        "run",
        "-n",
        "d3p",
        args.python_executable,
        "scripts/calibrate_td_threshold.py",
        "--dataset",
        fixed_placeholder / "transitions.npz",
        "--critic_checkpoint",
        fixed_placeholder / "td_critic" / "td_critic.pt",
        "--output_json",
        fixed_placeholder / "td_threshold_calibration.json",
        "--percentiles",
        "70,80,90,95",
        "--z_thresholds",
        "0.5,1.0,1.5,2.0",
        "--device",
        "cpu",
    ]
    train_ppo = [
        "conda",
        "run",
        "-n",
        "d3p",
        args.python_executable,
        "scripts/train_replan_ppo.py",
        "--env",
        env,
        "--seed",
        "0",
        "--total_env_steps",
        "720",
        "--rollout_steps",
        "80",
        "--max_episode_steps",
        str(task["max_episode_steps"]),
        "--eval_episodes",
        "3",
        "--hidden_dims",
        "64",
        "--minibatch_size",
        "40",
        "--update_epochs",
        "2",
        "--lambda_C",
        "0.003",
        "--lambda_D",
        "0.03",
        "--td_critic_checkpoint",
        td_artifact.critic_checkpoint if td_artifact else fixed_placeholder / "td_critic" / "td_critic.pt",
        "--uncertainty_replan_bonus",
        "0.3",
        "--bc_teacher",
        "td_error",
        "--bc_warmstart_steps",
        "720",
        "--bc_epochs",
        "10",
        "--bc_minibatch_size",
        "80",
        "--bc_learning_rate",
        "0.001",
        "--bc_td_calibration",
        td_artifact.calibration if td_artifact else fixed_placeholder / "td_threshold_calibration.json",
        "--bc_td_threshold_percentile",
        "90",
        "--bc_regularizer_coef",
        "0.2",
        "--base_policy_checkpoint",
        paths.base_policy_checkpoint,
        "--normalization_path",
        paths.normalization_path,
        "--env_meta_path",
        paths.env_meta_path,
        "--output_root",
        train_root,
    ]
    square_td_arg = (
        format_td_artifact(td_artifact)
        if td_artifact is not None
        else f"{env}:{fixed_placeholder / 'td_critic' / 'td_critic.pt'}={fixed_placeholder / 'td_threshold_calibration.json'}"
    )
    square_ppo_arg = (
        format_ppo_checkpoint(ppo_checkpoint)
        if ppo_checkpoint is not None
        else f"{env}:square_bc_p90_reg02=<square_ppo_checkpoint.pt>"
    )
    square_suite_placeholder = suite_root / "<square_final_eval_suite_dir>"

    final_suite = [
        "conda",
        "run",
        "-n",
        "d3p",
        args.python_executable,
        "scripts/run_final_eval_suite.py",
        "--envs",
        env,
        "--seeds",
        ",".join(str(seed) for seed in args.seeds),
        "--episodes",
        str(args.episodes),
        "--max_episode_steps",
        str(task["max_episode_steps"]),
        "--baseline_methods",
        "fixed_chunk,td_error_replan,denoise_only",
        "--td_artifact",
        square_td_arg,
        "--td_threshold_percentile",
        "95",
        "--runtime_artifact",
        runtime_arg,
        "--env_meta_path",
        paths.env_meta_path,
        "--timeline_episodes",
        "1",
        "--output_root",
        suite_root,
        "--ppo_checkpoint",
        square_ppo_arg,
    ]

    final_pipeline = [
        "conda",
        "run",
        "-n",
        "d3p",
        args.python_executable,
        "scripts/run_final_evidence_pipeline.py",
        "--envs",
        ",".join(args.envs),
        "--seeds",
        ",".join(str(seed) for seed in args.seeds),
        "--min_episodes",
        str(args.episodes),
    ]
    for suite_env in args.envs:
        if suite_env == env:
            suite_value = f"{suite_env}={square_suite_placeholder}"
        else:
            suite_value = f"{suite_env}=<{suite_env}_final_eval_suite_dir>"
        final_pipeline.extend(["--suite", suite_value])
    final_pipeline.append("--generate_readiness")
    for spec in args.td_artifact:
        final_pipeline.extend(["--td_artifact", format_td_artifact(spec)])
    if not has_td_artifact(args.td_artifact, env):
        final_pipeline.extend(["--td_artifact", square_td_arg])
    for spec in args.ppo_checkpoint:
        final_pipeline.extend(["--ppo_checkpoint", format_ppo_checkpoint(spec)])
    if not has_ppo_checkpoint(args.ppo_checkpoint, env):
        final_pipeline.extend(["--ppo_checkpoint", square_ppo_arg])
    for spec in args.runtime_artifact:
        final_pipeline.extend(["--runtime_artifact", format_runtime_artifact(spec)])
    if not has_runtime_artifact(args.runtime_artifact, env):
        final_pipeline.extend(["--runtime_artifact", runtime_arg])
    if args.env_meta_path is not None:
        final_pipeline.extend(["--env_meta_path", args.env_meta_path])
    final_pipeline.extend([
        "--square_work_root",
        args.square_work_root,
        "--output_dir",
        args.square_work_root / "final_pipeline",
        "--require_complete",
    ])

    return {
        "collect_fixed_transitions": shell_join(collect_fixed),
        "train_td_critic": shell_join(train_td),
        "calibrate_td_threshold": shell_join(calibrate_td),
        "train_ppo_replan": shell_join(train_ppo),
        "run_final_suite": shell_join(final_suite),
        "run_final_pipeline": shell_join(final_pipeline),
    }


def build_env_readiness(args: argparse.Namespace, env: str) -> dict[str, Any]:
    paths = runtime_paths_for_env(args, env)
    td_artifact = td_artifact_for_env(args.td_artifact, env)
    ppo_checkpoint = ppo_checkpoint_for_env(args.ppo_checkpoint, env)
    checks = [
        check_artifact("base_policy_checkpoint", paths.base_policy_checkpoint),
        check_artifact("normalization", paths.normalization_path),
        check_artifact("env_meta", paths.env_meta_path),
        check_artifact(
            "td_critic",
            td_artifact.critic_checkpoint if td_artifact else None,
        ),
        check_artifact(
            "td_calibration",
            td_artifact.calibration if td_artifact else None,
        ),
        check_artifact(
            "ppo_checkpoint",
            ppo_checkpoint.checkpoint if ppo_checkpoint else None,
        ),
    ]
    missing = [check.label for check in checks if not check.valid]
    runtime_ready = all(check.valid for check in checks[:3])
    final_eval_ready = all(check.valid for check in checks)
    report: dict[str, Any] = {
        "env": env,
        "task": TASKS[env]["env_name"],
        "max_episode_steps": int(TASKS[env]["max_episode_steps"]),
        "runtime_ready": runtime_ready,
        "final_eval_ready": final_eval_ready,
        "uses_runtime_artifact": paths.uses_runtime_artifact,
        "missing": missing,
        "checks": [check.as_dict() for check in checks],
    }
    if env == "robomimic_square":
        report["next_commands"] = build_square_commands(args, env, paths, td_artifact, ppo_checkpoint)
    return report


def build_readiness(args: argparse.Namespace) -> dict[str, Any]:
    env_reports = [build_env_readiness(args, env) for env in args.envs]
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "envs": env_reports,
        "all_final_eval_ready": all(item["final_eval_ready"] for item in env_reports),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Adaptive Replanning Final Readiness",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "| Env | Runtime Ready | Final Eval Ready | Missing |",
        "| --- | ---: | ---: | --- |",
    ]
    for env in report["envs"]:
        missing = ", ".join(env["missing"]) if env["missing"] else "none"
        lines.append(
            f"| {env['env']} | {str(env['runtime_ready']).lower()} | "
            f"{str(env['final_eval_ready']).lower()} | {missing} |"
        )
    lines.extend(["", "## Artifact Checks", ""])
    for env in report["envs"]:
        lines.extend([f"### {env['env']}", ""])
        for check in env["checks"]:
            if check["valid"]:
                status = "valid"
            elif check["exists"]:
                status = f"invalid ({check['reason']})"
            else:
                status = "missing"
            size = f", {check['size_bytes']} bytes" if check.get("size_bytes") is not None else ""
            lines.append(f"- `{check['label']}`: {status}{size} - `{check['path']}`")
        commands = env.get("next_commands") or {}
        if commands:
            lines.extend(["", "Suggested Square sequence:", ""])
            for label, command in commands.items():
                lines.extend([f"`{label}`:", "```bash", command, "```", ""])
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    for env in args.envs:
        if env not in TASKS:
            raise ValueError(f"unsupported env: {env}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"{timestamp}_final_readiness"
    output_dir.mkdir(parents=True, exist_ok=False)
    report = build_readiness(args)
    json_path = output_dir / "readiness.json"
    markdown_path = output_dir / "readiness.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_markdown(markdown_path, report)
    print(f"Wrote readiness report to {output_dir}")
    print(OmegaConf.to_yaml(OmegaConf.create({
        "output_dir": str(output_dir),
        "json": str(json_path),
        "markdown": str(markdown_path),
        "all_final_eval_ready": report["all_final_eval_ready"],
    }), resolve=True))


if __name__ == "__main__":
    main()
