"""Run Matrix B execution-disturbance stress tests."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


METHODS = [
    "fixed_chunk",
    "td_replan_anywhere_random_reset",
    "adaptive_commitment_heuristic",
    "adaptive_commitment_with_repair",
]
SIGMAS = [0.02, 0.05, 0.10]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/adaptive_commitment/matrix_B_execution_disturbance_v1")
    parser.add_argument("--tasks", nargs="+", default=["lift", "can", "square"])
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--sigmas", nargs="+", type=float, default=SIGMAS)
    parser.add_argument("--impulse-probability", type=float, default=0.10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--can-repair-accept-max-action-jump", type=float, default=None)
    parser.add_argument("--can-repair-reject-fallback", default="continue")
    args = parser.parse_args()

    repo = Path.cwd()
    for sigma in args.sigmas:
        sigma_name = f"sigma_{int(round(sigma * 1000)):03d}"
        output_root = Path(args.output_root) / sigma_name
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
            "--action-noise-sigma",
            str(sigma),
            "--action-noise-probability",
            str(args.impulse_probability),
        ]
        if args.can_repair_accept_max_action_jump is not None:
            command.extend(
                [
                    "--can-repair-accept-max-action-jump",
                    str(args.can_repair_accept_max_action_jump),
                    "--can-repair-reject-fallback",
                    args.can_repair_reject_fallback,
                ]
            )
        print(f"RUN Matrix B sigma={sigma} output={output_root}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
