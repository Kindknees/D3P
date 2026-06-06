"""Check whether the adaptive commitment Robomimic smoke can run locally."""

from __future__ import annotations

import importlib.util
from pathlib import Path


REQUIRED_MODULES = [
    "torch",
    "hydra",
    "omegaconf",
    "robomimic",
    "robosuite",
    "numpy",
]

REQUIRED_FILES = {
    "lift_normalization": Path("data/robomimic/lift/normalization.npz"),
    "lift_state299_checkpoint": Path(
        "log/robomimic-finetune/lift_ft_diffusion_mlp_ta4_td20_tdf10/checkpoint/state_299.pt"
    ),
    "lift_adaptive_config": Path(
        "cfg/robomimic/eval/lift/eval_adaptive_commitment_mlp.yaml"
    ),
}


def main() -> None:
    missing_modules = [name for name in REQUIRED_MODULES if not _module_exists(name)]
    missing_files = [
        f"{name}: {path}" for name, path in REQUIRED_FILES.items() if not path.exists()
    ]

    print("Adaptive commitment readiness")
    print("Modules:")
    for name in REQUIRED_MODULES:
        print(f"  {name}: {'ok' if name not in missing_modules else 'missing'}")
    print("Files:")
    for name, path in REQUIRED_FILES.items():
        print(f"  {name}: {'ok' if path.exists() else 'missing'} ({path})")

    if missing_modules or missing_files:
        print("status: not_ready")
        if missing_modules:
            print("missing_modules: " + ", ".join(missing_modules))
        if missing_files:
            print("missing_files: " + "; ".join(missing_files))
        raise SystemExit(1)

    print("status: ready")
    print("lift_fixed_chunk_smoke_command:")
    print(
        "python3 script/run.py --config-name robomimic/eval/lift/eval_adaptive_commitment_mlp "
        "base_policy_path=log/robomimic-finetune/lift_ft_diffusion_mlp_ta4_td20_tdf10/checkpoint/state_299.pt "
        "normalization_path=data/robomimic/lift/normalization.npz "
        "adaptive_commitment.method=fixed_chunk adaptive_commitment.episodes=1"
    )


def _module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


if __name__ == "__main__":
    main()
