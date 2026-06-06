"""Run Matrix E benefit-predictor controller comparison."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


METHODS = [
    "fixed_chunk",
    "td_replan_anywhere_random_reset",
    "adaptive_commitment_with_repair",
    "benefit_predictor_controller",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/adaptive_commitment/matrix_E_benefit_predictor_v1")
    parser.add_argument("--tasks", nargs="+", default=["lift", "can", "square"])
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--branch-logs-path", default="outputs/adaptive_commitment/branch_benefit_dataset_v2/branch_logs.csv")
    parser.add_argument("--min-benefit", type=float, default=5.0)
    parser.add_argument("--max-interventions", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    repo = Path.cwd()
    command = [
        "conda",
        "run",
        "-n",
        "d3p",
        "python",
        str(repo / "scripts/run_policy_disagreement_matrix_a.py"),
        "--output-root",
        args.output_root,
        "--tasks",
        *args.tasks,
        "--methods",
        *args.methods,
        "--seeds",
        *(str(seed) for seed in args.seeds),
        "--device",
        args.device,
        "--branch-logs-path",
        args.branch_logs_path,
        "--benefit-min-benefit",
        str(args.min_benefit),
        "--benefit-max-interventions",
        str(args.max_interventions),
    ]
    print("RUN Matrix E", flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
