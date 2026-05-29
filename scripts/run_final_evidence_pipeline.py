#!/usr/bin/env python3
"""Run the final evidence audit and report generation as one reproducible pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import audit_final_evidence as audit  # noqa: E402
from scripts import build_final_report as final_report  # noqa: E402
from scripts import check_final_readiness as readiness  # noqa: E402


FINAL_EXPECTED_LABELS_BY_ENV: dict[str, tuple[str, ...]] = {
    "robomimic_lift": ("fixed_chunk", "td_error_p95", "lift_bc_p90_reg02", "denoise_only"),
    "robomimic_can": ("fixed_chunk", "td_error_p95", "can_bc_p90_reg02", "denoise_only"),
    "robomimic_square": ("fixed_chunk", "td_error_p95", "square_bc_p90_reg02", "denoise_only"),
}


def expected_labels_from_preset(preset: str, envs: tuple[str, ...]) -> tuple[audit.LabelSpec, ...]:
    if preset == "none":
        return ()
    if preset != "final_full":
        raise ValueError(f"unknown expected-label preset: {preset}")
    labels: list[audit.LabelSpec] = []
    for env in envs:
        for label in FINAL_EXPECTED_LABELS_BY_ENV.get(env, ()):
            labels.append(audit.LabelSpec(env, label))
    return tuple(labels)


def merge_expected_labels(
    explicit: tuple[audit.LabelSpec, ...],
    preset: tuple[audit.LabelSpec, ...],
) -> tuple[audit.LabelSpec, ...]:
    merged: list[audit.LabelSpec] = []
    seen: set[tuple[str, str]] = set()
    for label in (*explicit, *preset):
        key = (label.env, label.label)
        if key in seen:
            continue
        seen.add(key)
        merged.append(label)
    return tuple(merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=audit.parse_csv, default=audit.DEFAULT_ENVS)
    parser.add_argument("--seeds", type=audit.parse_int_csv, default=audit.DEFAULT_SEEDS)
    parser.add_argument("--min_episodes", type=int, default=100)
    parser.add_argument("--suite", action="append", type=audit.parse_suite, default=[])
    parser.add_argument("--expected_label", action="append", type=audit.parse_label, default=[])
    parser.add_argument(
        "--expected_label_preset",
        choices=("none", "final_full"),
        default="final_full",
        help=(
            "Use a named final expected-label set. final_full requires fixed, TD-error, "
            "PPO, and denoise-only rows for each requested Robomimic env; use none for "
            "narrow diagnostics."
        ),
    )
    parser.add_argument("--readiness_json", type=Path, default=None)
    parser.add_argument(
        "--generate_readiness",
        action="store_true",
        help="Generate readiness.json inside the pipeline output before running the audit.",
    )
    parser.add_argument("--readiness_episodes", type=int, default=100)
    parser.add_argument("--td_bootstrap_episodes", type=int, default=3)
    parser.add_argument("--td_bootstrap_max_episode_steps", type=int, default=None)
    parser.add_argument("--td_artifact", action="append", type=readiness.parse_td_artifact, default=[])
    parser.add_argument("--ppo_checkpoint", action="append", type=readiness.parse_ppo_checkpoint, default=[])
    parser.add_argument("--runtime_artifact", action="append", type=readiness.parse_runtime_artifact, default=[])
    parser.add_argument("--env_meta_path", type=Path, default=None)
    parser.add_argument("--square_work_root", type=Path, default=Path("/tmp/d3p_square_final"))
    parser.add_argument("--python_executable", type=str, default="python")
    parser.add_argument(
        "--experiment_markdown",
        type=Path,
        default=REPO_ROOT / "experiments" / "adaptive_replanning.md",
    )
    parser.add_argument(
        "--markdown_token",
        action="append",
        default=list(audit.DEFAULT_MARKDOWN_TOKENS),
        help="Required text token in the experiment markdown. Can be repeated.",
    )
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs" / "final_pipeline")
    parser.add_argument(
        "--require_complete",
        action="store_true",
        help="Exit with status 1 after writing outputs if final evidence is incomplete.",
    )
    return parser.parse_args()


def readiness_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        envs=args.envs,
        seeds=args.seeds,
        episodes=args.readiness_episodes,
        td_bootstrap_episodes=args.td_bootstrap_episodes,
        td_bootstrap_max_episode_steps=args.td_bootstrap_max_episode_steps,
        td_artifact=args.td_artifact,
        ppo_checkpoint=args.ppo_checkpoint,
        runtime_artifact=args.runtime_artifact,
        env_meta_path=args.env_meta_path,
        square_work_root=args.square_work_root,
        python_executable=args.python_executable,
    )


def audit_namespace(args: argparse.Namespace) -> argparse.Namespace:
    preset_labels = expected_labels_from_preset(
        getattr(args, "expected_label_preset", "none"),
        tuple(args.envs),
    )
    return argparse.Namespace(
        envs=args.envs,
        seeds=args.seeds,
        min_episodes=args.min_episodes,
        suite=args.suite,
        expected_label=merge_expected_labels(tuple(args.expected_label), preset_labels),
        readiness_json=args.readiness_json,
        experiment_markdown=args.experiment_markdown,
        markdown_token=args.markdown_token,
        output_dir=args.output_dir,
    )


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def square_td_artifact_placeholder(args: argparse.Namespace, env: str) -> str:
    fixed_run = args.square_work_root / "td_data" / "<fixed_run_dir>"
    critic = fixed_run / "td_critic" / "td_critic.pt"
    calibration = fixed_run / "td_threshold_calibration.json"
    return f"{env}:{critic}={calibration}"


def square_ppo_checkpoint_placeholder(env: str) -> str:
    return f"{env}:square_bc_p90_reg02=<square_ppo_checkpoint.pt>"


def square_suite_placeholder(args: argparse.Namespace, env: str) -> str:
    return f"{env}={args.square_work_root / 'final_suite' / '<square_final_eval_suite_dir>'}"


def final_suite_placeholder(env: str) -> str:
    return "<square_final_eval_suite_dir>" if env == "robomimic_square" else f"<{env}_final_eval_suite_dir>"


def build_final_closeout_placeholders(args: argparse.Namespace) -> list[dict[str, str]]:
    placeholders: list[dict[str, str]] = []
    suite_envs = {suite.env for suite in args.suite}
    for env in args.envs:
        if env in suite_envs:
            continue
        placeholders.append(
            {
                "token": final_suite_placeholder(env),
                "kind": "final_suite",
                "env": env,
                "argument": "--suite",
                "replacement_hint": f"Replace with the completed final-eval suite directory for {env}.",
            }
        )

    if "robomimic_square" in args.envs and not readiness.has_td_artifact(args.td_artifact, "robomimic_square"):
        placeholders.append(
            {
                "token": "<fixed_run_dir>",
                "kind": "square_td_fixed_run_dir",
                "env": "robomimic_square",
                "argument": "--td_artifact",
                "replacement_hint": "Replace with the Square fixed-chunk run directory containing transitions.npz and td_critic outputs.",
            }
        )
    if "robomimic_square" in args.envs and not readiness.has_ppo_checkpoint(args.ppo_checkpoint, "robomimic_square"):
        placeholders.append(
            {
                "token": "<square_ppo_checkpoint.pt>",
                "kind": "square_ppo_checkpoint",
                "env": "robomimic_square",
                "argument": "--ppo_checkpoint",
                "replacement_hint": "Replace with the trained Square PPO replan checkpoint path.",
            }
        )
    return placeholders


def build_final_closeout_command(args: argparse.Namespace) -> str:
    parts: list[str | Path] = [
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
        str(args.min_episodes),
        "--expected_label_preset",
        getattr(args, "expected_label_preset", "final_full"),
    ]
    for label in args.expected_label:
        parts.extend(["--expected_label", f"{label.env}:{label.label}"])

    suite_envs: set[str] = set()
    for suite in args.suite:
        suite_envs.add(suite.env)
        parts.extend(["--suite", f"{suite.env}={suite.path}"])
    for env in args.envs:
        if env in suite_envs:
            continue
        if env == "robomimic_square":
            parts.extend(["--suite", square_suite_placeholder(args, env)])
        else:
            parts.extend(["--suite", f"{env}=<{env}_final_eval_suite_dir>"])

    if args.generate_readiness:
        parts.append("--generate_readiness")
    else:
        parts.extend(["--readiness_json", args.readiness_json])

    for spec in args.td_artifact:
        parts.extend(["--td_artifact", readiness.format_td_artifact(spec)])
    if "robomimic_square" in args.envs and not readiness.has_td_artifact(args.td_artifact, "robomimic_square"):
        parts.extend(["--td_artifact", square_td_artifact_placeholder(args, "robomimic_square")])

    for spec in args.ppo_checkpoint:
        parts.extend(["--ppo_checkpoint", readiness.format_ppo_checkpoint(spec)])
    if "robomimic_square" in args.envs and not readiness.has_ppo_checkpoint(args.ppo_checkpoint, "robomimic_square"):
        parts.extend(["--ppo_checkpoint", square_ppo_checkpoint_placeholder("robomimic_square")])

    for spec in args.runtime_artifact:
        parts.extend(["--runtime_artifact", readiness.format_runtime_artifact(spec)])
    if "robomimic_square" in args.envs and not readiness.has_runtime_artifact(args.runtime_artifact, "robomimic_square"):
        square_paths = readiness.runtime_paths_for_env(readiness_namespace(args), "robomimic_square")
        parts.extend(["--runtime_artifact", readiness.build_runtime_artifact_arg("robomimic_square", square_paths)])

    if args.env_meta_path is not None:
        parts.extend(["--env_meta_path", args.env_meta_path])
    parts.extend([
        "--square_work_root",
        args.square_work_root,
        "--output_dir",
        args.square_work_root / "final_pipeline",
        "--require_complete",
    ])
    return readiness.shell_join(parts)


def write_pipeline_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Adaptive Replanning Final Evidence Pipeline",
        "",
        f"Created: `{summary['created_at']}`",
        "",
        f"All goal evidence ready: `{str(summary['all_goal_evidence_ready']).lower()}`",
        f"All suites ready: `{str(summary['all_suites_ready']).lower()}`",
        f"Readiness ready: `{str(summary['readiness_ready']).lower()}`",
        f"Markdown ready: `{str(summary['markdown_ready']).lower()}`",
        "",
        "## Outputs",
        "",
        f"- Readiness JSON: `{summary['readiness_json']}`",
        f"- Readiness Markdown: `{summary['readiness_markdown']}`",
        f"- Audit JSON: `{summary['audit_json']}`",
        f"- Audit Markdown: `{summary['audit_markdown']}`",
        f"- Final Report JSON: `{summary['final_report_json']}`",
        f"- Final Results Matrix: `{summary['final_results_csv']}`",
        f"- Final Environment Summary: `{summary['final_env_summary_csv']}`",
        f"- Final Report Markdown: `{summary['final_report_markdown']}`",
    ]
    if summary.get("final_closeout_command"):
        lines.extend([
            "",
            "## Final Close-Out Command",
            "",
            f"Ready: `{str(summary.get('final_closeout_ready', False)).lower()}`",
            "",
            "```bash",
            summary["final_closeout_command"],
            "```",
        ])
        placeholders = list(summary.get("final_closeout_placeholders") or [])
        if placeholders:
            lines.extend([
                "",
                "### Close-Out Placeholders",
                "",
                "| Token | Env | Argument | Replacement Hint |",
                "| --- | --- | --- | --- |",
            ])
            for placeholder in placeholders:
                lines.append(
                    f"| `{placeholder.get('token', '')}` | {placeholder.get('env', '')} | "
                    f"`{placeholder.get('argument', '')}` | {placeholder.get('replacement_hint', '')} |"
                )
    lines.extend([
        "",
        "## Environment Status",
        "",
        "| Env | Suite Ready | Readiness Missing | Missing Rows | Failed Checks |",
        "| --- | ---: | --- | --- | --- |",
    ])
    for env in summary["envs"]:
        readiness_missing = ", ".join(env.get("readiness_missing") or []) or "none"
        missing_rows = ", ".join(env.get("missing_rows") or []) or "none"
        failed = ", ".join(env.get("failed_checks") or []) or "none"
        lines.append(
            f"| {env['env']} | {str(env['suite_ready']).lower()} | "
            f"{readiness_missing} | {missing_rows} | {failed} |"
        )

    if any(env.get("readiness_checks") or env.get("next_commands") for env in summary["envs"]):
        lines.extend(["", "## Readiness Artifact Details", ""])
        for env in summary["envs"]:
            checks = list(env.get("readiness_checks") or [])
            commands = env.get("next_commands") or {}
            if not checks and not commands:
                continue
            lines.extend([f"### {env['env']}", ""])
            for check in checks:
                status = "valid" if check.get("valid") else check.get("reason", "invalid")
                path_value = check.get("path") or "not_provided"
                lines.append(f"- `{check.get('label')}`: {status} - `{path_value}`")
            if commands:
                lines.extend(["", "Suggested Square sequence:", ""])
                for label, command in commands.items():
                    lines.extend([f"`{label}`:", "```bash", command, "```", ""])
            lines.append("")
    else:
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def resolve_readiness_json(args: argparse.Namespace, pipeline_dir: Path) -> tuple[Path, Path | None]:
    if args.generate_readiness and args.readiness_json is not None:
        raise ValueError("use either --generate_readiness or --readiness_json, not both")
    if not args.generate_readiness:
        if args.readiness_json is None:
            raise ValueError("provide --readiness_json or use --generate_readiness")
        return args.readiness_json, None

    readiness_dir = pipeline_dir / "readiness"
    readiness_dir.mkdir(parents=True, exist_ok=False)
    readiness_report = readiness.build_readiness(readiness_namespace(args))
    readiness_json = readiness_dir / "readiness.json"
    readiness_markdown = readiness_dir / "readiness.md"
    write_json(readiness_json, readiness_report)
    readiness.write_markdown(readiness_markdown, readiness_report)
    return readiness_json, readiness_markdown


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    if args.min_episodes <= 0:
        raise ValueError("--min_episodes must be positive")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_dir = args.output_dir / f"{timestamp}_final_evidence_pipeline"
    audit_dir = pipeline_dir / "audit"
    report_dir = pipeline_dir / "report"
    audit_dir.mkdir(parents=True, exist_ok=False)
    readiness_json, readiness_markdown = resolve_readiness_json(args, pipeline_dir)
    args.readiness_json = readiness_json

    audit_report = audit.build_audit(audit_namespace(args))
    audit_json = audit_dir / "final_evidence_audit.json"
    audit_markdown = audit_dir / "final_evidence_audit.md"
    audit_report["source_path"] = str(audit_json)
    write_json(audit_json, audit_report)
    audit.write_markdown(audit_markdown, audit_report)

    report = final_report.build_report(audit_report)
    report_outputs = final_report.write_report(report_dir, report)
    readiness_envs = {env.get("env"): env for env in audit_report["readiness"].get("envs", [])}
    readiness_missing_envs = set(audit_report["readiness"].get("missing") or [])

    env_summaries = []
    for env_report in audit_report["envs"]:
        env_name = env_report["env"]
        readiness_env = readiness_envs.get(env_name, {})
        readiness_missing = list(readiness_env.get("missing") or [])
        if not readiness_missing and env_name in readiness_missing_envs:
            readiness_missing = ["final_eval_ready"]
        env_summaries.append(
            {
                "env": env_name,
                "suite_ready": bool(env_report["suite_ready"]),
                "suite_paths": list(env_report.get("suite_paths") or []),
                "missing_rows": list(env_report.get("missing_rows") or []),
                "readiness_missing": readiness_missing,
                "readiness_checks": list(readiness_env.get("checks") or []),
                "next_commands": dict(readiness_env.get("next_commands") or {}),
                "failed_checks": [
                    check["label"] for check in env_report.get("checks", []) if not check.get("passed")
                ],
            }
        )

    final_closeout_placeholders = build_final_closeout_placeholders(args)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pipeline_dir": str(pipeline_dir),
        "readiness_json": str(readiness_json),
        "readiness_markdown": str(readiness_markdown) if readiness_markdown is not None else None,
        "generated_readiness": bool(args.generate_readiness),
        "audit_json": str(audit_json),
        "audit_markdown": str(audit_markdown),
        "final_report_json": report_outputs["json"],
        "final_results_csv": report_outputs["csv"],
        "final_env_summary_csv": report_outputs["env_summary_csv"],
        "final_report_markdown": report_outputs["markdown"],
        "final_closeout_command": build_final_closeout_command(args),
        "final_closeout_ready": not final_closeout_placeholders,
        "final_closeout_placeholders": final_closeout_placeholders,
        "all_goal_evidence_ready": bool(audit_report["all_goal_evidence_ready"]),
        "all_suites_ready": bool(audit_report["all_suites_ready"]),
        "readiness_ready": bool(audit_report["readiness"]["ready"]),
        "markdown_ready": bool(audit_report["markdown"]["ready"]),
        "require_complete": bool(getattr(args, "require_complete", False)),
        "envs": env_summaries,
    }
    summary_json = pipeline_dir / "pipeline_summary.json"
    summary_markdown = pipeline_dir / "pipeline_summary.md"
    write_json(summary_json, summary)
    write_pipeline_markdown(summary_markdown, summary)
    summary["pipeline_summary_json"] = str(summary_json)
    summary["pipeline_summary_markdown"] = str(summary_markdown)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_pipeline(args)
    print(f"Wrote final evidence pipeline to {summary['pipeline_dir']}")
    print(OmegaConf.to_yaml(OmegaConf.create(summary), resolve=True))
    if args.require_complete and not summary["all_goal_evidence_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
