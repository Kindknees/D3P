"""Run H5 criticality-aware compute-allocation comparison."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


METHODS = [
    "fixed_chunk",
    "adaptive_commitment_with_repair",
    "criticality_aware_commitment_with_repair",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/adaptive_commitment/matrix_H5_criticality_compute_v1")
    parser.add_argument("--tasks", nargs="+", default=["lift", "can", "square"])
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--low-nfe", type=int, default=10)
    parser.add_argument("--high-nfe", type=int, default=20)
    parser.add_argument("--criticality-threshold", type=float, default=None)
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
        "--criticality-low-nfe",
        str(args.low_nfe),
        "--criticality-high-nfe",
        str(args.high_nfe),
    ]
    if args.criticality_threshold is not None:
        command.extend(["--criticality-threshold", str(args.criticality_threshold)])
    print("RUN H5 criticality-aware compute", flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
