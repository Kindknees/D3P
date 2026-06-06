"""Run a return-cost Pareto subset for adaptive commitment."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


METHODS = [
    "fixed_chunk",
    "td_replan_anywhere_random_reset",
    "adaptive_commitment_with_repair",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/adaptive_commitment/matrix_D_square_pareto_v1")
    parser.add_argument("--tasks", nargs="+", default=["square"])
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--denoising-steps", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--repair-nfe", nargs="+", type=int, default=[2, 5, 10])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    repo = Path.cwd()
    for denoise in args.denoising_steps:
        repair_sweep = args.repair_nfe if "adaptive_commitment_with_repair" in args.methods else [args.repair_nfe[0]]
        for repair_nfe in repair_sweep:
            output_root = Path(args.output_root) / f"reset_nfe_{denoise:02d}_repair_nfe_{repair_nfe:02d}"
            command = [
                "conda",
                "run",
                "-n",
                "d3p",
                "python",
                str(repo / "scripts/run_policy_disagreement_matrix_a.py"),
                "--output-root",
                str(output_root),
                "--tasks",
                *args.tasks,
                "--methods",
                *args.methods,
                "--seeds",
                *(str(seed) for seed in args.seeds),
                "--device",
                args.device,
                "--denoising-steps",
                str(denoise),
                "--repair-nfe",
                str(repair_nfe),
            ]
            print(
                f"RUN Matrix D denoising_steps={denoise} repair_nfe={repair_nfe} output={output_root}",
                flush=True,
            )
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
