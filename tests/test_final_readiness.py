import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_final_readiness.py"
spec = importlib.util.spec_from_file_location("check_final_readiness", SCRIPT_PATH)
readiness = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = readiness
spec.loader.exec_module(readiness)


def test_build_readiness_marks_env_ready_with_runtime_td_and_ppo(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    normalization = tmp_path / "normalization.npz"
    td_critic = tmp_path / "td_critic.pt"
    calibration = tmp_path / "td_threshold_calibration.json"
    ppo = tmp_path / "ppo.pt"
    for path in (checkpoint, normalization, td_critic, calibration, ppo):
        path.write_text("x", encoding="utf-8")

    env_meta = tmp_path / "square.json"
    env_meta.write_text("{}", encoding="utf-8")

    args = Namespace(
        envs=("robomimic_square",),
        seeds=(0, 1, 2),
        episodes=100,
        td_bootstrap_episodes=3,
        td_bootstrap_max_episode_steps=None,
        td_artifact=(
            readiness.TDArtifactSpec("robomimic_square", td_critic, calibration),
        ),
        ppo_checkpoint=(
            readiness.PPOCheckpointSpec("square_ppo", ppo, "robomimic_square"),
        ),
        runtime_artifact=(
            readiness.RuntimeArtifactSpec("robomimic_square", checkpoint, normalization),
        ),
        env_meta_path=env_meta,
        square_work_root=tmp_path / "work",
        python_executable="python",
    )

    report = readiness.build_readiness(args)
    env = report["envs"][0]

    assert env["runtime_ready"] is True
    assert env["final_eval_ready"] is True
    assert env["missing"] == []
    final_suite_command = env["next_commands"]["run_final_suite"]
    assert "--runtime_artifact" in final_suite_command
    assert "square_ppo" in final_suite_command
    assert "--baseline_methods fixed_chunk,td_error_replan,denoise_only" in final_suite_command
    assert "--env_meta_path" in final_suite_command
    assert str(env_meta) in final_suite_command
    final_pipeline_command = env["next_commands"]["run_final_pipeline"]
    assert "scripts/run_final_evidence_pipeline.py" in final_pipeline_command
    assert "--generate_readiness" in final_pipeline_command
    assert "--require_complete" in final_pipeline_command
    assert "--suite" in final_pipeline_command
    assert "robomimic_square=" in final_pipeline_command
    assert "--runtime_artifact robomimic_square:" in final_pipeline_command
    assert "--env_meta_path" in final_pipeline_command
    assert str(env_meta) in final_pipeline_command
    assert "square_ppo" in final_pipeline_command
    assert (
        "conda run -n d3p python scripts/train_td_critic.py"
        in env["next_commands"]["train_td_critic"]
    )
    assert "d3p d3p python" not in env["next_commands"]["train_td_critic"]


def test_zero_byte_runtime_artifacts_are_invalid(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    normalization = tmp_path / "normalization.npz"
    td_critic = tmp_path / "td_critic.pt"
    calibration = tmp_path / "td_threshold_calibration.json"
    ppo = tmp_path / "ppo.pt"
    checkpoint.touch()
    normalization.touch()
    for path in (td_critic, calibration, ppo):
        path.write_text("x", encoding="utf-8")

    args = Namespace(
        envs=("robomimic_square",),
        seeds=(0,),
        episodes=1,
        td_bootstrap_episodes=1,
        td_bootstrap_max_episode_steps=None,
        td_artifact=(
            readiness.TDArtifactSpec("robomimic_square", td_critic, calibration),
        ),
        ppo_checkpoint=(
            readiness.PPOCheckpointSpec("square_ppo", ppo, "robomimic_square"),
        ),
        runtime_artifact=(
            readiness.RuntimeArtifactSpec("robomimic_square", checkpoint, normalization),
        ),
        env_meta_path=None,
        square_work_root=tmp_path / "work",
        python_executable="python",
    )

    report = readiness.build_readiness(args)
    env = report["envs"][0]
    checks = {check["label"]: check for check in env["checks"]}

    assert env["runtime_ready"] is False
    assert env["final_eval_ready"] is False
    assert "base_policy_checkpoint" in env["missing"]
    assert "normalization" in env["missing"]
    assert checks["base_policy_checkpoint"]["reason"] == "empty_file"
    assert checks["normalization"]["size_bytes"] == 0


def test_build_readiness_reports_missing_square_runtime_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("DPPO_LOG_DIR", str(tmp_path / "missing_log"))
    monkeypatch.setenv("DPPO_DATA_DIR", str(tmp_path / "missing_data"))

    args = Namespace(
        envs=("robomimic_square",),
        seeds=(0,),
        episodes=1,
        td_bootstrap_episodes=1,
        td_bootstrap_max_episode_steps=None,
        td_artifact=(),
        ppo_checkpoint=(),
        runtime_artifact=(),
        env_meta_path=None,
        square_work_root=tmp_path / "work",
        python_executable="python",
    )

    report = readiness.build_readiness(args)
    env = report["envs"][0]

    assert env["runtime_ready"] is False
    assert env["final_eval_ready"] is False
    assert "base_policy_checkpoint" in env["missing"]
    assert "normalization" in env["missing"]
    assert "td_critic" in env["missing"]
    assert "ppo_checkpoint" in env["missing"]
