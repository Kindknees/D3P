#!/usr/bin/env python3
"""Resolve a generated final close-out command from a pipeline summary."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ANGLE_PLACEHOLDER_RE = re.compile(r"<[^>]+>")


def parse_replacement(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("replacement must use TOKEN=VALUE")
    token, replacement = value.split("=", 1)
    token = token.strip()
    replacement = replacement.strip()
    if not token:
        raise argparse.ArgumentTypeError("replacement token cannot be empty")
    if not replacement:
        raise argparse.ArgumentTypeError("replacement value cannot be empty")
    return token, replacement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline_summary", type=Path, required=True)
    parser.add_argument(
        "--replace",
        action="append",
        type=parse_replacement,
        default=[],
        help="Placeholder replacement TOKEN=VALUE. Can be repeated.",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--validate_paths", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def placeholder_tokens(summary: dict[str, Any]) -> list[str]:
    return [str(item.get("token")) for item in summary.get("final_closeout_placeholders") or [] if item.get("token")]


def path_check(kind: str, argument: str, raw_path: str, *, expect_dir: bool = False) -> dict[str, Any]:
    path = Path(raw_path)
    exists = path.exists()
    is_file = path.is_file()
    is_dir = path.is_dir()
    size_bytes = path.stat().st_size if exists and is_file else None
    if not exists:
        valid = False
        reason = "missing"
    elif expect_dir and not is_dir:
        valid = False
        reason = "not_dir"
    elif not expect_dir and not is_file:
        valid = False
        reason = "not_file"
    elif not expect_dir and (size_bytes or 0) <= 0:
        valid = False
        reason = "empty_file"
    else:
        valid = True
        reason = "ok"
    return {
        "kind": kind,
        "argument": argument,
        "path": raw_path,
        "exists": exists,
        "is_file": is_file,
        "is_dir": is_dir,
        "size_bytes": size_bytes,
        "valid": valid,
        "reason": reason,
    }


def _path_after_env_prefix(value: str) -> str:
    return value.split(":", 1)[1] if ":" in value else value


def collect_path_checks(command: str) -> list[dict[str, Any]]:
    parts = shlex.split(command)
    checks: list[dict[str, Any]] = []
    for index, part in enumerate(parts[:-1]):
        value = parts[index + 1]
        if part == "--suite":
            if "=" not in value:
                continue
            checks.append(path_check("final_suite", part, value.split("=", 1)[1], expect_dir=True))
        elif part == "--td_artifact":
            artifact = _path_after_env_prefix(value)
            if "=" not in artifact:
                continue
            critic, calibration = artifact.split("=", 1)
            checks.append(path_check("td_critic", part, critic))
            checks.append(path_check("td_calibration", part, calibration))
        elif part == "--ppo_checkpoint":
            if "=" not in value:
                continue
            checks.append(path_check("ppo_checkpoint", part, value.split("=", 1)[1]))
        elif part == "--runtime_artifact":
            artifact = _path_after_env_prefix(value)
            if "=" not in artifact:
                continue
            checkpoint, normalization = artifact.split("=", 1)
            checks.append(path_check("base_policy_checkpoint", part, checkpoint))
            checks.append(path_check("normalization", part, normalization))
        elif part in ("--readiness_json", "--env_meta_path"):
            checks.append(path_check(part.lstrip("-"), part, value))
    return checks


def resolve_command(
    summary: dict[str, Any],
    replacements: dict[str, str],
    *,
    validate_paths: bool = False,
) -> dict[str, Any]:
    command = str(summary.get("final_closeout_command") or "")
    if not command:
        raise ValueError("pipeline summary does not contain final_closeout_command")

    tokens = placeholder_tokens(summary)
    missing = [token for token in tokens if token not in replacements]
    if missing:
        raise ValueError("missing replacements: " + ", ".join(missing))

    resolved = command
    applied: dict[str, str] = {}
    for token in tokens:
        replacement = replacements[token]
        resolved = resolved.replace(token, replacement)
        applied[token] = replacement

    unresolved_tokens = sorted(set(ANGLE_PLACEHOLDER_RE.findall(resolved)))
    path_checks = collect_path_checks(resolved) if validate_paths else []
    paths_ready = all(check["valid"] for check in path_checks) if validate_paths else None
    ready = not unresolved_tokens and (paths_ready is not False)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pipeline_summary": str(summary.get("pipeline_summary_json") or ""),
        "ready": ready,
        "validate_paths": validate_paths,
        "paths_ready": paths_ready,
        "source_command": command,
        "resolved_command": resolved,
        "applied_replacements": applied,
        "unresolved_tokens": unresolved_tokens,
        "path_checks": path_checks,
    }


def write_outputs(output_dir: Path, resolution: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "resolved_closeout.json"
    command_path = output_dir / "resolved_closeout_command.sh"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(resolution, f, indent=2)
    command_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + resolution["resolved_command"] + "\n",
        encoding="utf-8",
    )
    command_path.chmod(0o755)
    return {"json": str(json_path), "command": str(command_path)}


def resolve_from_file(
    pipeline_summary: Path,
    replacements: Sequence[tuple[str, str]],
    output_dir: Path | None = None,
    *,
    validate_paths: bool = False,
) -> dict[str, Any]:
    summary = read_json(pipeline_summary)
    summary["pipeline_summary_json"] = str(pipeline_summary)
    resolution = resolve_command(summary, dict(replacements), validate_paths=validate_paths)
    output_dir = output_dir or pipeline_summary.parent / "resolved_closeout"
    outputs = write_outputs(output_dir, resolution)
    return {**resolution, "outputs": outputs}


def main() -> None:
    args = parse_args()
    resolution = resolve_from_file(
        args.pipeline_summary, args.replace, args.output_dir, validate_paths=args.validate_paths
    )
    print(json.dumps(resolution, indent=2))
    if not resolution["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
