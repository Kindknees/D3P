from pathlib import Path


def test_adaptive_eval_bridge_config_points_to_agent():
    text = Path("cfg/robomimic/eval/lift/eval_adaptive_commitment_mlp.yaml").read_text()

    assert "_target_: agent.eval.eval_adaptive_commitment_agent.EvalAdaptiveCommitmentAgent" in text
    assert "act_steps: 1" in text
    assert "n_envs: 1" in text
    assert "horizon_steps: 4" in text
    assert "adaptive_commitment:" in text


def test_adaptive_commitment_summary_is_canonical_doc():
    docs = sorted(Path("docs").glob("*Adaptive*Commitment*.md"))
    assert docs == [Path("docs/Adaptive-Commitment-Experiment-Summary.md")]

    text = docs[0].read_text()
    assert "anti-reset / continuity preservation" in text
    assert "policy_disagreement" in text
