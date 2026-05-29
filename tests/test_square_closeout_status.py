import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_square_closeout_status.py"
spec = importlib.util.spec_from_file_location("build_square_closeout_status", SCRIPT_PATH)
status_script = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = status_script
spec.loader.exec_module(status_script)


def _write_file(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_suite(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _write_file(path / "suite_summary.json", json.dumps({"dry_run": False, "num_jobs": 3, "num_completed": 3}))
    return path


def _write_fixed_run(path: Path) -> Path:
    _write_file(path / "transitions.npz")
    _write_file(path / "td_critic" / "td_critic.pt")
    _write_file(path / "td_threshold_calibration.json")
    return path


def _summary_command(root: Path) -> str:
    return (
        "conda run -n d3p python scripts/run_final_evidence_pipeline.py "
        f"--suite robomimic_square={root}/final_suite/<square_final_eval_suite_dir> "
        f"--td_artifact robomimic_square:{root}/td_data/<fixed_run_dir>/td_critic/td_critic.pt="
        f"{root}/td_data/<fixed_run_dir>/td_threshold_calibration.json "
        "--ppo_checkpoint robomimic_square:square_bc_p90_reg02=<square_ppo_checkpoint.pt> "
        f"--runtime_artifact robomimic_square:{root}/runtime/state.pt={root}/runtime/normalization.npz "
        "--require_complete"
    )


def _write_pipeline_summary(path: Path, root: Path, *, square_ready: bool) -> Path:
    if square_ready:
        square_env = {
            "env": "robomimic_square",
            "suite_ready": True,
            "suite_paths": [str(root / "final_suite" / "suite")],
            "missing_rows": [],
            "readiness_missing": [],
            "readiness_checks": [
                {"label": "base_policy_checkpoint", "valid": True, "path": str(root / "runtime" / "state.pt")},
                {"label": "normalization", "valid": True, "path": str(root / "runtime" / "normalization.npz")},
            ],
            "next_commands": {},
            "failed_checks": [],
        }
    else:
        square_env = {
            "env": "robomimic_square",
            "suite_ready": False,
            "suite_paths": [],
            "missing_rows": ["fixed_chunk:seed0"],
            "readiness_missing": ["base_policy_checkpoint"],
            "readiness_checks": [
                {
                    "label": "base_policy_checkpoint",
                    "valid": False,
                    "reason": "missing",
                    "path": str(root / "runtime" / "state.pt"),
                }
            ],
            "next_commands": {"collect_fixed_transitions": "python scripts/run_baseline.py --env robomimic_square"},
            "failed_checks": ["suite_provided"],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "created_at": "2026-05-29T00:00:00",
                "pipeline_dir": str(path.parent),
                "all_goal_evidence_ready": square_ready,
                "readiness_ready": square_ready,
                "all_suites_ready": square_ready,
                "final_closeout_ready": False,
                "final_closeout_command": _summary_command(root),
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
                "envs": [square_env],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_build_status_reports_square_blockers_when_outputs_are_missing(tmp_path):
    root = tmp_path / "square_work"
    summary = _write_pipeline_summary(tmp_path / "pipeline_summary.json", root, square_ready=False)

    status = status_script.build_status(summary, root)

    assert status["ready_for_final_pipeline"] is False
    assert status["summary_readiness_ready"] is False
    assert status["summary_suite_ready"] is False
    assert status["discovery_ready"] is False
    assert status["resolver_ready"] is None
    assert status["missing_replacements"] == [
        "<square_final_eval_suite_dir>",
        "<fixed_run_dir>",
        "<square_ppo_checkpoint.pt>",
    ]
    blockers = {(blocker["kind"], blocker.get("label") or blocker.get("token") or blocker.get("row")) for blocker in status["blockers"]}
    assert ("readiness_missing", "base_policy_checkpoint") in blockers
    assert ("missing_row", "fixed_chunk:seed0") in blockers
    assert ("discovery", "<square_final_eval_suite_dir>") in blockers
    assert ("resolver_missing_replacement", "<fixed_run_dir>") in blockers
    assert "collect_fixed_transitions" in status["next_commands"]


def test_build_status_resolves_and_validates_discovered_square_outputs(tmp_path):
    root = tmp_path / "square_work"
    summary = _write_pipeline_summary(tmp_path / "pipeline_summary.json", root, square_ready=True)
    _write_suite(root / "final_suite" / "suite")
    _write_fixed_run(root / "td_data" / "fixed")
    ppo = _write_file(root / "ppo_train" / "ppo" / "checkpoints" / "best.pt")
    _write_file(root / "runtime" / "state.pt")
    _write_file(root / "runtime" / "normalization.npz")

    status = status_script.build_status(summary, root)

    assert status["ready_for_final_pipeline"] is True
    assert status["discovery_ready"] is True
    assert status["resolver_ready"] is True
    assert status["blockers"] == []
    assert status["combined_replacements"] == {
        "<square_final_eval_suite_dir>": "suite",
        "<fixed_run_dir>": "fixed",
        "<square_ppo_checkpoint.pt>": str(ppo),
    }
    assert status["resolution"]["paths_ready"] is True


def test_write_outputs_creates_json_and_markdown(tmp_path):
    root = tmp_path / "square_work"
    summary = _write_pipeline_summary(tmp_path / "pipeline_summary.json", root, square_ready=False)
    status = status_script.build_status(summary, root)

    outputs = status_script.write_outputs(tmp_path / "status", status)

    assert Path(outputs["json"]).exists()
    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    assert "Square Close-Out Status" in markdown
    assert "Ready for final pipeline" in markdown
    assert "<square_final_eval_suite_dir>" in markdown
