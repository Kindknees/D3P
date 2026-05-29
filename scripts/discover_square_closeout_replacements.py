#!/usr/bin/env python3
"""Discover Square close-out replacements under a square_work_root."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SQUARE_SUITE_TOKEN = "<square_final_eval_suite_dir>"
FIXED_RUN_TOKEN = "<fixed_run_dir>"
SQUARE_PPO_TOKEN = "<square_ppo_checkpoint.pt>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--square_work_root", type=Path, default=Path("/tmp/d3p_square_final"))
    parser.add_argument("--pipeline_summary", type=Path, default=None)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--python_executable", type=str, default="python")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def nonempty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def base_check(token: str, kind: str, path: Path | None, valid: bool, reason: str) -> dict[str, Any]:
    return {
        "token": token,
        "kind": kind,
        "path": str(path) if path is not None else None,
        "valid": bool(valid),
        "reason": reason,
    }


def validate_suite_dir(path: Path) -> dict[str, Any]:
    summary_path = path / "suite_summary.json"
    if not path.is_dir():
        return base_check(SQUARE_SUITE_TOKEN, "square_final_suite", path, False, "not_dir")
    if not nonempty_file(summary_path):
        return base_check(SQUARE_SUITE_TOKEN, "square_final_suite", path, False, "suite_summary_missing")
    try:
        summary = read_json(summary_path)
    except (OSError, json.JSONDecodeError):
        return base_check(SQUARE_SUITE_TOKEN, "square_final_suite", path, False, "suite_summary_invalid")
    if bool(summary.get("dry_run")):
        return base_check(SQUARE_SUITE_TOKEN, "square_final_suite", path, False, "dry_run")
    num_jobs = int(summary.get("num_jobs") or 0)
    num_completed = int(summary.get("num_completed") or 0)
    if num_jobs <= 0 or num_completed != num_jobs:
        return base_check(SQUARE_SUITE_TOKEN, "square_final_suite", path, False, "incomplete_jobs")
    return base_check(SQUARE_SUITE_TOKEN, "square_final_suite", path, True, "ok")


def validate_fixed_run_dir(path: Path) -> dict[str, Any]:
    required = [
        path / "transitions.npz",
        path / "td_critic" / "td_critic.pt",
        path / "td_threshold_calibration.json",
    ]
    missing = [item.name for item in required if not nonempty_file(item)]
    if missing:
        check = base_check(FIXED_RUN_TOKEN, "square_td_fixed_run_dir", path, False, "missing_required_files")
        check["missing"] = missing
        return check
    return base_check(FIXED_RUN_TOKEN, "square_td_fixed_run_dir", path, True, "ok")


def validate_ppo_checkpoint(path: Path) -> dict[str, Any]:
    if not nonempty_file(path):
        return base_check(SQUARE_PPO_TOKEN, "square_ppo_checkpoint", path, False, "missing_or_empty")
    return base_check(SQUARE_PPO_TOKEN, "square_ppo_checkpoint", path, True, "ok")


def newest_valid_candidate(
    candidates: list[Path],
    validator: Callable[[Path], dict[str, Any]],
    token: str,
    kind: str,
) -> dict[str, Any]:
    sorted_candidates = sorted(candidates, key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)
    inspected = []
    for candidate in sorted_candidates:
        check = validator(candidate)
        inspected.append(check)
        if check["valid"]:
            check["candidate_count"] = len(sorted_candidates)
            return check
    check = base_check(token, kind, None, False, "no_valid_candidate")
    check["candidate_count"] = len(sorted_candidates)
    check["inspected"] = inspected[:5]
    return check


def discover_replacements(square_work_root: Path) -> dict[str, Any]:
    suite_root = square_work_root / "final_suite"
    td_root = square_work_root / "td_data"
    ppo_root = square_work_root / "ppo_train"

    suite_candidates = [path for path in suite_root.iterdir()] if suite_root.exists() else []
    td_candidates = [path for path in td_root.iterdir()] if td_root.exists() else []
    ppo_candidates = list(ppo_root.glob("**/checkpoints/best.pt")) if ppo_root.exists() else []

    checks = [
        newest_valid_candidate(suite_candidates, validate_suite_dir, SQUARE_SUITE_TOKEN, "square_final_suite"),
        newest_valid_candidate(td_candidates, validate_fixed_run_dir, FIXED_RUN_TOKEN, "square_td_fixed_run_dir"),
        newest_valid_candidate(ppo_candidates, validate_ppo_checkpoint, SQUARE_PPO_TOKEN, "square_ppo_checkpoint"),
    ]
    replacements: dict[str, str] = {}
    for check in checks:
        if not check["valid"]:
            continue
        token = str(check["token"])
        path = Path(str(check["path"]))
        replacements[token] = path.name if token in (SQUARE_SUITE_TOKEN, FIXED_RUN_TOKEN) else str(path)
    replacement_args = [f"{token}={value}" for token, value in replacements.items()]
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "square_work_root": str(square_work_root),
        "ready": len(replacements) == 3,
        "replacements": replacements,
        "replacement_args": replacement_args,
        "checks": checks,
    }


def resolver_command(discovery: dict[str, Any], pipeline_summary: Path, python_executable: str) -> str:
    parts: list[str | Path] = [
        "conda",
        "run",
        "-n",
        "d3p",
        python_executable,
        "scripts/resolve_final_closeout.py",
        "--pipeline_summary",
        pipeline_summary,
    ]
    for arg in discovery["replacement_args"]:
        parts.extend(["--replace", arg])
    parts.append("--validate_paths")
    return shlex.join(str(part) for part in parts)


def main() -> None:
    args = parse_args()
    discovery = discover_replacements(args.square_work_root)
    if args.pipeline_summary is not None:
        discovery["pipeline_summary"] = str(args.pipeline_summary)
        discovery["resolver_command"] = resolver_command(discovery, args.pipeline_summary, args.python_executable)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(discovery, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(discovery, indent=2))
    if not discovery["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
