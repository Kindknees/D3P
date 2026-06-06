from pathlib import Path


def test_debug_agent_supports_fixed_schedule_trigger():
    text = Path("agent/eval/eval_adaptive_commitment_debug_agent.py").read_text()

    assert "fixed_schedule" in text
    assert "trigger_steps" in text
    assert "EvalAdaptiveCommitmentAgent" in text


def test_debug_config_targets_debug_agent():
    text = Path("cfg/robomimic/eval/lift/eval_adaptive_commitment_debug_mlp.yaml").read_text()

    assert "EvalAdaptiveCommitmentDebugAgent" in text
    assert "source: fixed_schedule" in text
