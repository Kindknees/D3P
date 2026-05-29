import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _write_file(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "resolve_final_closeout.py"
spec = importlib.util.spec_from_file_location("resolve_final_closeout", SCRIPT_PATH)
resolver = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = resolver
spec.loader.exec_module(resolver)


def _write_summary(path: Path, root: Path | None = None) -> Path:
    root = root or Path("/tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "final_closeout_command": (
                    "conda run -n d3p python scripts/run_final_evidence_pipeline.py "
                    f"--suite robomimic_square={root}/final/<square_final_eval_suite_dir> "
                    f"--td_artifact robomimic_square:{root}/td/<fixed_run_dir>/td_critic/td_critic.pt="
                    f"{root}/td/<fixed_run_dir>/td_threshold_calibration.json "
                    "--ppo_checkpoint robomimic_square:square_bc_p90_reg02=<square_ppo_checkpoint.pt> "
                    f"--runtime_artifact robomimic_square:{root}/runtime/policy.pt={root}/runtime/normalization.npz "
                    "--require_complete"
                ),
                "final_closeout_placeholders": [
                    {
                        "token": "<square_final_eval_suite_dir>",
                        "kind": "final_suite",
                        "env": "robomimic_square",
                        "argument": "--suite",
                    },
                    {
                        "token": "<fixed_run_dir>",
                        "kind": "square_td_fixed_run_dir",
                        "env": "robomimic_square",
                        "argument": "--td_artifact",
                    },
                    {
                        "token": "<square_ppo_checkpoint.pt>",
                        "kind": "square_ppo_checkpoint",
                        "env": "robomimic_square",
                        "argument": "--ppo_checkpoint",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_resolve_from_file_writes_ready_command_and_json(tmp_path):
    summary = _write_summary(tmp_path / "pipeline_summary.json")

    resolution = resolver.resolve_from_file(
        summary,
        (
            ("<square_final_eval_suite_dir>", "20260529_square_final_eval_suite"),
            ("<fixed_run_dir>", "20260529_square_fixed_seed0"),
            ("<square_ppo_checkpoint.pt>", "/tmp/square/checkpoints/best.pt"),
        ),
        tmp_path / "resolved",
    )

    assert resolution["ready"] is True
    assert resolution["unresolved_tokens"] == []
    assert "<" not in resolution["resolved_command"]
    assert "20260529_square_final_eval_suite" in resolution["resolved_command"]
    assert "20260529_square_fixed_seed0" in resolution["resolved_command"]
    assert "/tmp/square/checkpoints/best.pt" in resolution["resolved_command"]
    json_path = Path(resolution["outputs"]["json"])
    command_path = Path(resolution["outputs"]["command"])
    assert json_path.exists()
    assert command_path.exists()
    assert command_path.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash")


def test_validate_paths_marks_existing_artifacts_ready(tmp_path):
    root = tmp_path / "square"
    (root / "final" / "suite_dir").mkdir(parents=True)
    _write_file(root / "td" / "fixed_run" / "td_critic" / "td_critic.pt")
    _write_file(root / "td" / "fixed_run" / "td_threshold_calibration.json")
    _write_file(root / "runtime" / "policy.pt")
    _write_file(root / "runtime" / "normalization.npz")
    ppo = _write_file(root / "ppo" / "best.pt")
    summary = _write_summary(tmp_path / "pipeline_summary.json", root=root)

    resolution = resolver.resolve_from_file(
        summary,
        (
            ("<square_final_eval_suite_dir>", "suite_dir"),
            ("<fixed_run_dir>", "fixed_run"),
            ("<square_ppo_checkpoint.pt>", str(ppo)),
        ),
        tmp_path / "resolved_with_checks",
        validate_paths=True,
    )

    assert resolution["ready"] is True
    assert resolution["paths_ready"] is True
    assert resolution["path_checks"]
    assert all(check["valid"] for check in resolution["path_checks"])
    kinds = {check["kind"] for check in resolution["path_checks"]}
    assert {
        "final_suite",
        "td_critic",
        "td_calibration",
        "ppo_checkpoint",
        "base_policy_checkpoint",
        "normalization",
    }.issubset(kinds)


def test_validate_paths_reports_missing_artifacts(tmp_path):
    root = tmp_path / "square"
    (root / "final" / "suite_dir").mkdir(parents=True)
    _write_file(root / "td" / "fixed_run" / "td_threshold_calibration.json")
    _write_file(root / "runtime" / "policy.pt")
    _write_file(root / "runtime" / "normalization.npz")
    ppo = _write_file(root / "ppo" / "best.pt")
    summary = _write_summary(tmp_path / "pipeline_summary.json", root=root)

    resolution = resolver.resolve_command(
        json.loads(summary.read_text(encoding="utf-8")),
        {
            "<square_final_eval_suite_dir>": "suite_dir",
            "<fixed_run_dir>": "fixed_run",
            "<square_ppo_checkpoint.pt>": str(ppo),
        },
        validate_paths=True,
    )

    assert resolution["ready"] is False
    assert resolution["paths_ready"] is False
    missing = [check for check in resolution["path_checks"] if not check["valid"]]
    assert len(missing) == 1
    assert missing[0]["kind"] == "td_critic"
    assert missing[0]["reason"] == "missing"


def test_resolve_command_rejects_missing_replacements(tmp_path):
    summary = json.loads(_write_summary(tmp_path / "pipeline_summary.json").read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="missing replacements"):
        resolver.resolve_command(summary, {"<fixed_run_dir>": "run"})


def test_resolve_command_reports_unresolved_unknown_placeholders(tmp_path):
    summary = json.loads(_write_summary(tmp_path / "pipeline_summary.json").read_text(encoding="utf-8"))
    summary["final_closeout_command"] += " --extra <manual_value>"

    resolution = resolver.resolve_command(
        summary,
        {
            "<square_final_eval_suite_dir>": "suite",
            "<fixed_run_dir>": "run",
            "<square_ppo_checkpoint.pt>": "ppo.pt",
        },
    )

    assert resolution["ready"] is False
    assert resolution["unresolved_tokens"] == ["<manual_value>"]
