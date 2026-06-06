from pathlib import Path


def test_readiness_script_checks_core_dependencies_and_lift_files():
    text = Path("scripts/check_adaptive_commitment_readiness.py").read_text()

    for token in [
        "torch",
        "hydra",
        "omegaconf",
        "robomimic",
        "robosuite",
        "state_299.pt",
        "eval_adaptive_commitment_mlp.yaml",
    ]:
        assert token in text
