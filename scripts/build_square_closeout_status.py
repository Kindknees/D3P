#!/usr/bin/env python3
"""Build a single Square close-out status artifact from final pipeline outputs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discover_square_closeout_replacements import discover_replacements, resolver_command  # noqa: E402
from scripts.resolve_final_closeout import parse_replacement, resolve_command  # noqa: E402

SQUARE_ENV = "robomimic_square"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline_summary", type=Path, required=True)
    parser.add_argument("--square_work_root", type=Path, default=Path("/tmp/d3p_square_final"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--python_executable", type=str, default="python")
    parser.add_argument(
        "--replace",
        action="append",
        type=parse_replacement,
        default=[],
        help="Manual placeholder replacement TOKEN=VALUE. Overrides discovered replacements.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def square_env_summary(summary: dict[str, Any]) -> dict[str, Any] | None:
    for env in summary.get("envs") or []:
        if env.get("env") == SQUARE_ENV:
            return dict(env)
    return None


def placeholder_tokens(summary: dict[str, Any]) -> list[str]:
    return [str(item.get("token")) for item in summary.get("final_closeout_placeholders") or [] if item.get("token")]


def replacement_map(discovery: dict[str, Any], manual_replacements: Sequence[tuple[str, str]]) -> dict[str, str]:
    replacements = dict(discovery.get("replacements") or {})
    replacements.update(dict(manual_replacements))
    return replacements


def try_resolve(
    summary: dict[str, Any],
    replacements: dict[str, str],
) -> tuple[dict[str, Any] | None, list[str]]:
    tokens = placeholder_tokens(summary)
    missing = [token for token in tokens if token not in replacements]
    if missing:
        return None, missing
    if not summary.get("final_closeout_command"):
        return None, []
    return resolve_command(summary, replacements, validate_paths=True), []


def _add_readiness_blockers(blockers: list[dict[str, Any]], square_env: dict[str, Any]) -> None:
    for label in square_env.get("readiness_missing") or []:
        blockers.append({"kind": "readiness_missing", "label": label})


def _add_missing_row_blockers(blockers: list[dict[str, Any]], square_env: dict[str, Any]) -> None:
    for row in square_env.get("missing_rows") or []:
        blockers.append({"kind": "missing_row", "row": row})


def _add_discovery_blockers(blockers: list[dict[str, Any]], discovery: dict[str, Any]) -> None:
    for check in discovery.get("checks") or []:
        if check.get("valid"):
            continue
        blockers.append(
            {
                "kind": "discovery",
                "token": check.get("token"),
                "reason": check.get("reason"),
                "candidate_count": check.get("candidate_count"),
            }
        )


def _add_resolver_blockers(
    blockers: list[dict[str, Any]],
    resolution: dict[str, Any] | None,
    missing_replacements: Sequence[str],
) -> None:
    for token in missing_replacements:
        blockers.append({"kind": "resolver_missing_replacement", "token": token})
    if resolution is None:
        return
    for token in resolution.get("unresolved_tokens") or []:
        blockers.append({"kind": "resolver_unresolved_token", "token": token})
    for check in resolution.get("path_checks") or []:
        if check.get("valid"):
            continue
        blockers.append(
            {
                "kind": "path_check",
                "label": check.get("kind"),
                "reason": check.get("reason"),
                "path": check.get("path"),
            }
        )


def build_status(
    pipeline_summary: Path,
    square_work_root: Path,
    manual_replacements: Sequence[tuple[str, str]] = (),
    *,
    python_executable: str = "python",
) -> dict[str, Any]:
    summary = read_json(pipeline_summary)
    summary["pipeline_summary_json"] = str(pipeline_summary)
    square_env = square_env_summary(summary)
    discovery = discover_replacements(square_work_root)
    replacements = replacement_map(discovery, manual_replacements)
    resolution, missing_replacements = try_resolve(summary, replacements)

    blockers: list[dict[str, Any]] = []
    if square_env is None:
        blockers.append({"kind": "square_env_missing", "env": SQUARE_ENV})
        summary_readiness_ready = False
        summary_suite_ready = False
    else:
        _add_readiness_blockers(blockers, square_env)
        _add_missing_row_blockers(blockers, square_env)
        summary_readiness_ready = not bool(square_env.get("readiness_missing"))
        summary_suite_ready = bool(square_env.get("suite_ready")) and not bool(square_env.get("missing_rows"))

    _add_discovery_blockers(blockers, discovery)
    _add_resolver_blockers(blockers, resolution, missing_replacements)

    resolver_ready = None if resolution is None else bool(resolution.get("ready"))
    ready_for_final_pipeline = (
        summary_readiness_ready
        and summary_suite_ready
        and bool(discovery.get("ready"))
        and resolver_ready is True
        and not blockers
    )
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pipeline_summary": str(pipeline_summary),
        "square_work_root": str(square_work_root),
        "ready_for_final_pipeline": ready_for_final_pipeline,
        "summary_readiness_ready": summary_readiness_ready,
        "summary_suite_ready": summary_suite_ready,
        "discovery_ready": bool(discovery.get("ready")),
        "resolver_ready": resolver_ready,
        "manual_replacements": [f"{token}={value}" for token, value in manual_replacements],
        "combined_replacements": replacements,
        "missing_replacements": list(missing_replacements),
        "blockers": blockers,
        "square_env": square_env,
        "final_closeout_ready": bool(summary.get("final_closeout_ready")),
        "final_closeout_command": summary.get("final_closeout_command"),
        "final_closeout_placeholders": list(summary.get("final_closeout_placeholders") or []),
        "next_commands": dict((square_env or {}).get("next_commands") or {}),
        "discovery": discovery,
        "discovery_resolver_command": resolver_command(discovery, pipeline_summary, python_executable),
        "resolution": resolution,
    }


def _table_row(values: Sequence[Any]) -> str:
    return "| " + " | ".join(str(value) for value in values) + " |"


def markdown_for_status(status: dict[str, Any]) -> str:
    lines = [
        "# Square Close-Out Status",
        "",
        f"Created: `{status['created_at']}`",
        f"Ready for final pipeline: `{str(status['ready_for_final_pipeline']).lower()}`",
        f"Pipeline summary: `{status['pipeline_summary']}`",
        f"Square work root: `{status['square_work_root']}`",
        "",
        "## Gates",
        "",
        _table_row(["Gate", "Ready"]),
        _table_row(["---", "---:"]),
        _table_row(["summary_readiness", str(status["summary_readiness_ready"]).lower()]),
        _table_row(["summary_suite", str(status["summary_suite_ready"]).lower()]),
        _table_row(["discovery", str(status["discovery_ready"]).lower()]),
        _table_row(["resolver", "not_run" if status["resolver_ready"] is None else str(status["resolver_ready"]).lower()]),
        "",
        "## Blockers",
        "",
    ]
    blockers = list(status.get("blockers") or [])
    if blockers:
        lines.extend([
            _table_row(["Kind", "Label/Token", "Reason", "Path/Row"]),
            _table_row(["---", "---", "---", "---"]),
        ])
        for blocker in blockers:
            label = blocker.get("label") or blocker.get("token") or blocker.get("env") or ""
            path_or_row = blocker.get("path") or blocker.get("row") or ""
            lines.append(_table_row([blocker.get("kind", ""), label, blocker.get("reason", ""), path_or_row]))
    else:
        lines.append("none")

    lines.extend([
        "",
        "## Discovery Checks",
        "",
        _table_row(["Token", "Valid", "Reason", "Path"]),
        _table_row(["---", "---:", "---", "---"]),
    ])
    for check in status["discovery"].get("checks") or []:
        lines.append(
            _table_row([
                f"`{check.get('token', '')}`",
                str(check.get("valid", False)).lower(),
                check.get("reason", ""),
                f"`{check.get('path')}`" if check.get("path") else "",
            ])
        )

    if status.get("resolution") is not None:
        lines.extend([
            "",
            "## Resolver Path Checks",
            "",
            _table_row(["Kind", "Valid", "Reason", "Path"]),
            _table_row(["---", "---:", "---", "---"]),
        ])
        for check in status["resolution"].get("path_checks") or []:
            lines.append(
                _table_row([
                    check.get("kind", ""),
                    str(check.get("valid", False)).lower(),
                    check.get("reason", ""),
                    f"`{check.get('path')}`",
                ])
            )
    elif status.get("missing_replacements"):
        lines.extend([
            "",
            "## Missing Replacements",
            "",
        ])
        for token in status["missing_replacements"]:
            lines.append(f"- `{token}`")

    if status.get("final_closeout_command"):
        lines.extend([
            "",
            "## Final Close-Out Command",
            "",
            "```bash",
            status["final_closeout_command"],
            "```",
            "",
            "Discovery resolver command:",
            "",
            "```bash",
            status["discovery_resolver_command"],
            "```",
        ])

    if status.get("next_commands"):
        lines.extend(["", "## Next Commands", ""])
        for label, command in status["next_commands"].items():
            lines.extend([f"`{label}`:", "```bash", str(command), "```", ""])

    return "\n".join(lines).rstrip() + "\n"


def write_outputs(output_dir: Path, status: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "square_closeout_status.json"
    markdown_path = output_dir / "square_closeout_status.md"
    json_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_for_status(status), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def main() -> None:
    args = parse_args()
    status = build_status(
        args.pipeline_summary,
        args.square_work_root,
        args.replace,
        python_executable=args.python_executable,
    )
    output_dir = args.output_dir or args.pipeline_summary.parent / "square_closeout_status"
    status["outputs"] = write_outputs(output_dir, status)
    print(json.dumps(status, indent=2))
    if not status["ready_for_final_pipeline"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
