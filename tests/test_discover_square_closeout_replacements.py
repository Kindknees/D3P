import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "discover_square_closeout_replacements.py"
spec = importlib.util.spec_from_file_location("discover_square_closeout_replacements", SCRIPT_PATH)
discovery = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = discovery
spec.loader.exec_module(discovery)


def _write_file(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_suite(path: Path, *, dry_run=False, num_jobs=3, num_completed=3) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _write_file(
        path / "suite_summary.json",
        json.dumps({"dry_run": dry_run, "num_jobs": num_jobs, "num_completed": num_completed}),
    )
    return path


def _write_fixed_run(path: Path) -> Path:
    _write_file(path / "transitions.npz")
    _write_file(path / "td_critic" / "td_critic.pt")
    _write_file(path / "td_threshold_calibration.json")
    return path


def test_discover_replacements_finds_ready_square_outputs(tmp_path):
    root = tmp_path / "square_work"
    suite = _write_suite(root / "final_suite" / "20260529_square_final_eval_suite")
    fixed = _write_fixed_run(root / "td_data" / "20260529_square_fixed_seed0")
    ppo = _write_file(root / "ppo_train" / "20260529_square_ppo" / "checkpoints" / "best.pt")

    result = discovery.discover_replacements(root)

    assert result["ready"] is True
    assert result["replacements"] == {
        "<square_final_eval_suite_dir>": suite.name,
        "<fixed_run_dir>": fixed.name,
        "<square_ppo_checkpoint.pt>": str(ppo),
    }
    assert all(check["valid"] for check in result["checks"])
    assert "<fixed_run_dir>=20260529_square_fixed_seed0" in result["replacement_args"]


def test_discover_replacements_reports_missing_and_invalid_outputs(tmp_path):
    root = tmp_path / "square_work"
    _write_suite(root / "final_suite" / "dry_run_suite", dry_run=True)
    _write_file(root / "td_data" / "partial_run" / "transitions.npz")

    result = discovery.discover_replacements(root)

    assert result["ready"] is False
    assert result["replacements"] == {}
    checks = {check["token"]: check for check in result["checks"]}
    assert checks["<square_final_eval_suite_dir>"]["reason"] == "no_valid_candidate"
    assert checks["<fixed_run_dir>"]["reason"] == "no_valid_candidate"
    assert checks["<square_ppo_checkpoint.pt>"]["reason"] == "no_valid_candidate"
    assert checks["<fixed_run_dir>"]["inspected"][0]["missing"] == ["td_critic.pt", "td_threshold_calibration.json"]


def test_resolver_command_uses_discovered_replacements(tmp_path):
    root = tmp_path / "square_work"
    _write_suite(root / "final_suite" / "suite")
    _write_fixed_run(root / "td_data" / "fixed")
    ppo = _write_file(root / "ppo_train" / "ppo" / "checkpoints" / "best.pt")
    result = discovery.discover_replacements(root)

    command = discovery.resolver_command(result, tmp_path / "pipeline_summary.json", "python")

    assert "scripts/resolve_final_closeout.py" in command
    assert "--validate_paths" in command
    assert "--replace '<square_final_eval_suite_dir>=suite'" in command
    assert "--replace '<fixed_run_dir>=fixed'" in command
    assert f"--replace '<square_ppo_checkpoint.pt>={ppo}'" in command
