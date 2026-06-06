"""Run paired confirmatory residual-buffer repair experiments."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


TASKS = {
    "lift": {
        "config_path": "cfg/robomimic/eval/lift",
        "checkpoint": "log/robomimic-finetune/lift_ft_diffusion_mlp_ta4_td20_tdf10/checkpoint/state_299.pt",
        "normalization": "data/robomimic/lift/normalization.npz",
        "steps": 300,
        "threshold": 0.15,
    },
    "can": {
        "config_path": "cfg/robomimic/eval/can",
        "checkpoint": "log/robomimic-finetune/can_ft_diffusion_mlp_ta4_td20_tdf10/2026-05-06_16-53-08_42/checkpoint/state_100.pt",
        "normalization": "data/robomimic/can/normalization.npz",
        "steps": 300,
        "threshold": 0.05,
    },
    "square": {
        "config_path": "cfg/robomimic/eval/square",
        "checkpoint": "log/robomimic-pretrain/square/square_pre_diffusion_mlp_ta4_td20/2024-07-10_01-46-16/checkpoint/state_8000.pt",
        "normalization": "data/robomimic/square/normalization.npz",
        "steps": 400,
        "threshold": 0.03,
    },
}

METHODS = [
    "fixed_chunk",
    "td_replan_anywhere_random_reset",
    "adaptive_commitment_with_repair",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/adaptive_commitment/confirmatory_repair_v1")
    parser.add_argument("--tasks", nargs="+", default=["can", "square"])
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--episodes-per-seed", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--can-repair-gate",
        choices=["gated", "pilot", "custom"],
        default="gated",
        help="Use the confirmatory Can gate, original pilot no-gate setup, or custom Can gate flags.",
    )
    parser.add_argument("--can-threshold", type=float, default=None)
    parser.add_argument("--task-threshold", type=float, default=None)
    parser.add_argument("--can-accept-min-action-jump", type=float, default=None)
    parser.add_argument("--can-accept-max-action-jump", type=float, default=None)
    parser.add_argument("--can-reject-fallback", choices=["continue", "reset"], default="continue")
    parser.add_argument("--lift-accept-min-action-jump", type=float, default=None)
    parser.add_argument("--lift-accept-max-action-jump", type=float, default=None)
    parser.add_argument("--lift-reject-fallback", choices=["continue", "reset"], default="continue")
    parser.add_argument("--square-accept-min-action-jump", type=float, default=0.05)
    parser.add_argument("--square-accept-max-action-jump", type=float, default=0.10)
    parser.add_argument("--square-reject-fallback", choices=["continue", "reset"], default="continue")
    parser.add_argument("--repair-noise-ratio", type=float, default=0.50)
    parser.add_argument("--repair-anchor-rho", type=float, default=0.50)
    parser.add_argument("--denoising-steps", type=int, default=None)
    parser.add_argument("--repair-nfe", type=int, default=10)
    args = parser.parse_args()

    repo = Path.cwd()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    env["LD_LIBRARY_PATH"] = "/home/koukanni/.mujoco/mujoco210/bin:/usr/lib/nvidia"

    for task in args.tasks:
        task_cfg = TASKS[task]
        if args.task_threshold is not None:
            threshold = args.task_threshold
        elif task == "can" and args.can_threshold is not None:
            threshold = args.can_threshold
        else:
            threshold = task_cfg["threshold"]
        steps = task_cfg["steps"]
        for method in args.methods:
            if method not in METHODS and method != "td_repair_residual_050":
                raise ValueError(f"Unsupported confirmatory method: {method}")
            for seed in args.seeds:
                logdir = Path(args.output_root) / task / method / f"seed_{seed}"
                summary_path = logdir / "adaptive_commitment_logs" / "episode_summary.csv"
                if args.skip_existing and summary_path.exists():
                    print(f"SKIP {task} {method} seed={seed}", flush=True)
                    continue
                logdir.mkdir(parents=True, exist_ok=True)
                command = [
                    "conda",
                    "run",
                    "-n",
                    "d3p",
                    "python",
                    "script/run.py",
                    "--config-path",
                    str(repo / task_cfg["config_path"]),
                    "--config-name",
                    "eval_adaptive_commitment_mlp",
                    f"base_policy_path={task_cfg['checkpoint']}",
                    f"normalization_path={task_cfg['normalization']}",
                    f"logdir={logdir}",
                    f"device={args.device}",
                    f"seed={seed}",
                    f"env.max_episode_steps={steps}",
                    f"n_steps={steps}",
                    f"adaptive_commitment.episodes={args.episodes_per_seed}",
                    f"adaptive_commitment.method={method}",
                    f"adaptive_commitment.commitment.high_td_threshold={threshold}",
                    f"adaptive_commitment.commitment.medium_td_threshold={threshold * 0.67}",
                    f"adaptive_commitment.reset.trigger_td_threshold={threshold}",
                    "adaptive_commitment.repair.enabled=true",
                    f"adaptive_commitment.repair.trigger_td_threshold={threshold}",
                    f"adaptive_commitment.repair.nfe={args.repair_nfe}",
                    f"adaptive_commitment.repair.noise_ratio={args.repair_noise_ratio:.2f}",
                    f"adaptive_commitment.repair.anchor_rho={args.repair_anchor_rho:.2f}",
                ]
                if args.denoising_steps is not None:
                    command.append(f"denoising_steps={args.denoising_steps}")
                if method == "fixed_chunk":
                    command.extend(
                        [
                            "adaptive_commitment.uncertainty.source=none",
                            "adaptive_commitment.uncertainty.num_samples=2",
                            f"adaptive_commitment.uncertainty.sample_interval={steps}",
                        ]
                    )
                else:
                    command.extend(
                        [
                            "adaptive_commitment.uncertainty.source=policy_disagreement",
                            "adaptive_commitment.uncertainty.num_samples=2",
                            f"adaptive_commitment.uncertainty.sample_interval={steps}",
                        ]
                    )
                if task == "square" and method in {"adaptive_commitment_with_repair", "td_repair_residual_050"}:
                    command.extend(
                        [
                            f"adaptive_commitment.repair.accept_min_action_jump={args.square_accept_min_action_jump}",
                            f"adaptive_commitment.repair.accept_max_action_jump={args.square_accept_max_action_jump}",
                            f"adaptive_commitment.repair.reject_fallback={args.square_reject_fallback}",
                        ]
                    )
                if task == "can" and method in {"adaptive_commitment_with_repair", "td_repair_residual_050"}:
                    if args.can_repair_gate == "gated":
                        command.extend(
                            [
                                "adaptive_commitment.repair.accept_max_action_jump=0.12",
                                "adaptive_commitment.repair.reject_fallback=continue",
                            ]
                        )
                    elif args.can_repair_gate == "custom":
                        if args.can_accept_min_action_jump is not None:
                            command.append(
                                f"adaptive_commitment.repair.accept_min_action_jump={args.can_accept_min_action_jump}"
                            )
                        if args.can_accept_max_action_jump is not None:
                            command.append(
                                f"adaptive_commitment.repair.accept_max_action_jump={args.can_accept_max_action_jump}"
                            )
                        command.append(f"adaptive_commitment.repair.reject_fallback={args.can_reject_fallback}")

                if task == "lift" and method in {"adaptive_commitment_with_repair", "td_repair_residual_050"}:
                    if args.lift_accept_min_action_jump is not None:
                        command.append(
                            f"adaptive_commitment.repair.accept_min_action_jump={args.lift_accept_min_action_jump}"
                        )
                    if args.lift_accept_max_action_jump is not None:
                        command.append(
                            f"adaptive_commitment.repair.accept_max_action_jump={args.lift_accept_max_action_jump}"
                        )
                    if args.lift_accept_min_action_jump is not None or args.lift_accept_max_action_jump is not None:
                        command.append(f"adaptive_commitment.repair.reject_fallback={args.lift_reject_fallback}")


                print(
                    f"RUN {task} {method} seed={seed} episodes={args.episodes_per_seed}",
                    flush=True,
                )
                with (logdir / "run.log").open("w") as log_file:
                    subprocess.run(
                        command,
                        check=True,
                        env=env,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                    )


if __name__ == "__main__":
    main()
