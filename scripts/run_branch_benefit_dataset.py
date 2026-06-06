"""Run branch-benefit collection across tasks, seeds, and event steps."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path


TASKS = {
    "lift": {
        "config_path": "cfg/robomimic/eval/lift",
        "checkpoint": "log/robomimic-finetune/lift_ft_diffusion_mlp_ta4_td20_tdf10/checkpoint/state_299.pt",
        "normalization": "data/robomimic/lift/normalization.npz",
        "horizon": 300,
    },
    "can": {
        "config_path": "cfg/robomimic/eval/can",
        "checkpoint": "log/robomimic-finetune/can_ft_diffusion_mlp_ta4_td20_tdf10/2026-05-06_16-53-08_42/checkpoint/state_100.pt",
        "normalization": "data/robomimic/can/normalization.npz",
        "horizon": 300,
    },
    "square": {
        "config_path": "cfg/robomimic/eval/square",
        "checkpoint": "log/robomimic-pretrain/square/square_pre_diffusion_mlp_ta4_td20/2024-07-10_01-46-16/checkpoint/state_8000.pt",
        "normalization": "data/robomimic/square/normalization.npz",
        "horizon": 400,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/adaptive_commitment/branch_benefit_dataset_v1")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--event-steps", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    repo = Path.cwd()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    env["LD_LIBRARY_PATH"] = "/home/koukanni/.mujoco/mujoco210/bin:/usr/lib/nvidia"

    output_root = Path(args.output_root)
    rows = []
    for task in args.tasks:
        task_cfg = TASKS[task]
        for seed in args.seeds:
            for event_step in args.event_steps:
                run_dir = output_root / "runs" / task / f"seed_{seed}" / f"event_{event_step}"
                run_dir.mkdir(parents=True, exist_ok=True)
                command = [
                    "conda",
                    "run",
                    "-n",
                    "d3p",
                    "python",
                    "scripts/collect_branch_benefit_smoke.py",
                    "--config-path",
                    task_cfg["config_path"],
                    "--config-name",
                    "eval_adaptive_commitment_mlp",
                    "--base-policy-path",
                    task_cfg["checkpoint"],
                    "--normalization-path",
                    task_cfg["normalization"],
                    "--output-dir",
                    str(run_dir),
                    "--seed",
                    str(seed),
                    "--branch-horizon",
                    str(task_cfg["horizon"]),
                    "--event-step",
                    str(event_step),
                    "--device",
                    args.device,
                ]
                print(f"RUN {task} seed={seed} event_step={event_step}", flush=True)
                with (run_dir / "run.log").open("w") as log_file:
                    subprocess.run(
                        command,
                        check=True,
                        env=env,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                    )
                rows.extend(_read_rows(run_dir / "branch_logs.csv"))

    merged = output_root / "branch_logs.csv"
    merged.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with merged.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote merged branch logs to {merged}")
    print(f"num_events={len(rows)}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file_obj:
        return list(csv.DictReader(file_obj))


if __name__ == "__main__":
    main()
