"""Run the real Robomimic policy-disagreement Matrix A experiment."""

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
    "td_replan_boundary_only",
    "phase_aware_hysteresis",
    "adaptive_commitment_heuristic",
    "td_repair_residual_050",
    "adaptive_commitment_with_repair",
]


def _repair_noise_ratio(method: str) -> float:
    if method.endswith("_025"):
        return 0.25
    if method.endswith("_075"):
        return 0.75
    return 0.50


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/adaptive_commitment/matrix_A_policy_disagreement_full_v1")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-noise-sigma", type=float, default=0.0)
    parser.add_argument("--action-noise-probability", type=float, default=0.0)
    parser.add_argument("--can-repair-accept-max-action-jump", type=float, default=None)
    parser.add_argument("--can-repair-reject-fallback", default="continue")
    parser.add_argument("--denoising-steps", type=int, default=None)
    parser.add_argument("--repair-nfe", type=int, default=10)
    parser.add_argument("--branch-logs-path", default="outputs/adaptive_commitment/branch_benefit_dataset_v2/branch_logs.csv")
    parser.add_argument("--benefit-min-benefit", type=float, default=5.0)
    parser.add_argument("--benefit-max-interventions", type=int, default=1)
    parser.add_argument("--criticality-low-nfe", type=int, default=10)
    parser.add_argument("--criticality-high-nfe", type=int, default=20)
    parser.add_argument("--criticality-threshold", type=float, default=None)
    args = parser.parse_args()

    repo = Path.cwd()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    env["LD_LIBRARY_PATH"] = "/home/koukanni/.mujoco/mujoco210/bin:/usr/lib/nvidia"

    for task in args.tasks:
        task_cfg = TASKS[task]
        threshold = task_cfg["threshold"]
        steps = task_cfg["steps"]
        for method in args.methods:
            for seed in args.seeds:
                logdir = Path(args.output_root) / task / method / f"seed_{seed}"
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
                    "adaptive_commitment.episodes=1",
                    f"adaptive_commitment.method={method}",
                ]
                if args.denoising_steps is not None:
                    command.append(f"denoising_steps={args.denoising_steps}")
                if method == "benefit_predictor_controller":
                    command.extend(
                        [
                            f"adaptive_commitment.benefit_predictor.branch_logs_path={args.branch_logs_path}",
                            f"adaptive_commitment.benefit_predictor.min_benefit={args.benefit_min_benefit}",
                            f"adaptive_commitment.benefit_predictor.max_interventions={args.benefit_max_interventions}",
                        ]
                    )
                if method == "criticality_aware_commitment_with_repair":
                    criticality_threshold = (
                        threshold if args.criticality_threshold is None else args.criticality_threshold
                    )
                    command.extend(
                        [
                            f"adaptive_commitment.criticality.low_nfe={args.criticality_low_nfe}",
                            f"adaptive_commitment.criticality.high_nfe={args.criticality_high_nfe}",
                            f"adaptive_commitment.criticality.threshold={criticality_threshold}",
                        ]
                    )
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
                repair_noise_ratio = _repair_noise_ratio(method)
                if args.action_noise_sigma > 0 and args.action_noise_probability > 0:
                    command.extend(
                        [
                            "adaptive_commitment.action_noise.enabled=true",
                            "adaptive_commitment.action_noise.type=impulse",
                            f"adaptive_commitment.action_noise.sigma={args.action_noise_sigma}",
                            f"adaptive_commitment.action_noise.probability={args.action_noise_probability}",
                        ]
                    )
                command.extend(
                    [
                    f"adaptive_commitment.commitment.high_td_threshold={threshold}",
                    f"adaptive_commitment.commitment.medium_td_threshold={threshold * 0.67}",
                    f"adaptive_commitment.reset.trigger_td_threshold={threshold}",
                    "adaptive_commitment.repair.enabled=true",
                    f"adaptive_commitment.repair.trigger_td_threshold={threshold}",
                    f"adaptive_commitment.repair.nfe={args.repair_nfe}",
                    f"adaptive_commitment.repair.noise_ratio={repair_noise_ratio:.2f}",
                    "adaptive_commitment.repair.anchor_rho=0.50",
                    f"adaptive_commitment.phase_hysteresis.phase_thresholds=[{threshold},{threshold},{threshold},{threshold}]",
                    "adaptive_commitment.phase_hysteresis.min_consecutive=1",
                    ]
                )
                if task == "square" and method in {
                    "adaptive_commitment_with_repair",
                    "benefit_predictor_controller",
                    "criticality_aware_commitment_with_repair",
                }:
                    command.extend(
                        [
                            "adaptive_commitment.repair.accept_min_action_jump=0.05",
                            "adaptive_commitment.repair.accept_max_action_jump=0.10",
                            "adaptive_commitment.repair.reject_fallback=continue",
                        ]
                    )
                if (
                    task == "can"
                    and method == "adaptive_commitment_with_repair"
                    and args.can_repair_accept_max_action_jump is not None
                ):
                    command.extend(
                        [
                            f"adaptive_commitment.repair.accept_max_action_jump={args.can_repair_accept_max_action_jump}",
                            f"adaptive_commitment.repair.reject_fallback={args.can_repair_reject_fallback}",
                        ]
                    )
                print(f"RUN {task} {method} seed={seed}", flush=True)
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
