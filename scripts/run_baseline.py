#!/usr/bin/env python3
"""Run adaptive diffusion baseline evaluations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptive_diffusion import (  # noqa: E402
    AdaptiveDiffusionExecutor,
    DenoiseOnlyController,
    FixedChunkController,
    TDCriticValueFunction,
    TDErrorHeuristicController,
    TDErrorSignal,
    TorchDiffusionPolicyAdapter,
    UncertaintyRuleDenoiseAdaptor,
    resolve_baseline_reference_config,
)
from adaptive_diffusion.td_critic import format_percentile_key  # noqa: E402
from env.gym_utils import make_async  # noqa: E402
from model.diffusion.diffusion import DiffusionModel  # noqa: E402
from model.diffusion.mlp_diffusion import DiffusionMLP  # noqa: E402


TASKS = {
    "robomimic_lift": {
        "env_name": "lift",
        "obs_dim": 19,
        "action_dim": 7,
        "max_episode_steps": 300,
        "policy_horizon": 4,
        "checkpoint": (
            "robomimic-pretrain/lift/"
            "lift_pre_diffusion_mlp_ta4_td20/"
            "2024-06-28_14-47-58/checkpoint/state_5000.pt"
        ),
    },
    "robomimic_can": {
        "env_name": "can",
        "obs_dim": 23,
        "action_dim": 7,
        "max_episode_steps": 300,
        "policy_horizon": 4,
        "checkpoint": (
            "robomimic-pretrain/can/"
            "can_pre_diffusion_mlp_ta4_td20/"
            "2024-06-28_13-29-54/checkpoint/state_5000.pt"
        ),
    },
    "robomimic_square": {
        "env_name": "square",
        "obs_dim": 23,
        "action_dim": 7,
        "max_episode_steps": 400,
        "policy_horizon": 4,
        "checkpoint": (
            "robomimic-pretrain/square/"
            "square_pre_diffusion_mlp_ta4_td20/"
            "2024-07-10_01-46-16/checkpoint/state_8000.pt"
        ),
        "network": {
            "time_dim": 32,
            "mlp_dims": [1024, 1024, 1024],
            "cond_mlp_dims": [512, 64],
        },
    },
}


def diffusion_mlp_kwargs(task: dict) -> dict:
    network = task.get("network") or {}
    cond_mlp_dims = network.get("cond_mlp_dims")
    return {
        "time_dim": int(network.get("time_dim", 16)),
        "mlp_dims": list(network.get("mlp_dims", [512, 512, 512])),
        "cond_mlp_dims": list(cond_mlp_dims) if cond_mlp_dims is not None else None,
        "residual_style": bool(network.get("residual_style", True)),
        "cond_dim": int(task["obs_dim"]),
        "horizon_steps": int(task["policy_horizon"]),
        "action_dim": int(task["action_dim"]),
    }


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


def parse_int_csv(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one integer")
    return values


def parse_float_csv(value: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one float")
    return values


def ensure_mujoco_build_paths() -> None:
    """Add local MuJoCo/OpenGL paths needed by mujoco_py builds."""

    conda_prefix = Path(os.environ.get("CONDA_PREFIX", ""))
    include_candidates = [
        conda_prefix / "x86_64-conda-linux-gnu" / "sysroot" / "usr" / "include",
    ]
    library_candidates = [
        Path.home() / ".mujoco" / "mujoco210" / "bin",
        Path("/usr/lib/nvidia"),
    ]
    _append_env_paths("CPATH", include_candidates)
    _append_env_paths("C_INCLUDE_PATH", include_candidates)
    _append_env_paths("LD_LIBRARY_PATH", library_candidates)


def _append_env_paths(name: str, candidates: list[Path]) -> None:
    current = os.environ.get(name, "")
    paths = [path for path in current.split(":") if path]
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in paths:
            paths.append(candidate_str)
    if paths:
        os.environ[name] = ":".join(paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=sorted(TASKS), default="robomimic_lift")
    parser.add_argument(
        "--method",
        choices=[
            "fixed_chunk",
            "high_compute",
            "low_compute",
            "td_error_replan",
            "denoise_only",
        ],
        default="fixed_chunk",
    )
    parser.add_argument("--chunk_horizon", type=int, default=None)
    parser.add_argument("--denoise_steps", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", type=str_to_bool, default=False)
    parser.add_argument("--max_episode_steps", type=int, default=None)
    parser.add_argument("--save_transitions", type=str_to_bool, default=False)
    parser.add_argument("--td_critic_checkpoint", type=Path, default=None)
    parser.add_argument(
        "--td_threshold_mode", choices=["z_score", "percentile"], default="z_score"
    )
    parser.add_argument("--td_threshold", type=float, default=1.0)
    parser.add_argument("--td_threshold_percentile", type=float, default=90.0)
    parser.add_argument("--td_calibration", type=Path, default=None)
    parser.add_argument("--td_gamma", type=float, default=0.99)
    parser.add_argument(
        "--denoise_candidates",
        type=parse_int_csv,
        default=(4, 8, 12, 20),
    )
    parser.add_argument(
        "--denoise_uncertainty_thresholds",
        type=parse_float_csv,
        default=(0.5, 1.0, 1.5),
    )
    parser.add_argument("--base_policy_checkpoint", type=Path, default=None)
    parser.add_argument("--normalization_path", type=Path, default=None)
    parser.add_argument("--env_meta_path", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "outputs" / "runs")
    return parser.parse_args()


def load_calibrated_threshold(calibration_path: Path, percentile: float) -> float:
    with calibration_path.open("r", encoding="utf-8") as f:
        calibration = json.load(f)
    thresholds = calibration.get("percentile_thresholds", {})
    key = format_percentile_key(percentile)
    if key not in thresholds:
        available = ", ".join(sorted(thresholds)) or "none"
        raise KeyError(
            f"percentile {key} not found in {calibration_path}; "
            f"available percentiles: {available}"
        )
    return float(thresholds[key])


def build_controller_and_uncertainty(args: argparse.Namespace, device: str):
    if args.method in {"fixed_chunk", "high_compute", "low_compute"}:
        return FixedChunkController(), None, None

    if args.method == "denoise_only":
        controller = DenoiseOnlyController(
            UncertaintyRuleDenoiseAdaptor(
                candidate_steps=args.denoise_candidates,
                uncertainty_thresholds=args.denoise_uncertainty_thresholds,
            )
        )
        uncertainty_signal = None
        if args.td_critic_checkpoint is not None:
            if not args.td_critic_checkpoint.exists():
                raise FileNotFoundError(
                    f"TD critic checkpoint not found: {args.td_critic_checkpoint}"
                )
            value_fn = TDCriticValueFunction(args.td_critic_checkpoint, device=device)
            uncertainty_signal = TDErrorSignal(value_fn=value_fn, gamma=args.td_gamma)
        return controller, uncertainty_signal, None

    if args.td_critic_checkpoint is None:
        raise ValueError("--td_critic_checkpoint is required for td_error_replan")
    if not args.td_critic_checkpoint.exists():
        raise FileNotFoundError(f"TD critic checkpoint not found: {args.td_critic_checkpoint}")

    calibrated_threshold = None
    if args.td_threshold_mode == "percentile":
        if args.td_calibration is None:
            raise ValueError("--td_calibration is required for percentile threshold mode")
        if not args.td_calibration.exists():
            raise FileNotFoundError(f"TD calibration file not found: {args.td_calibration}")
        calibrated_threshold = load_calibrated_threshold(
            args.td_calibration, args.td_threshold_percentile
        )

    controller = TDErrorHeuristicController(
        threshold_mode=args.td_threshold_mode,
        threshold=args.td_threshold,
        threshold_percentile=args.td_threshold_percentile,
        calibrated_threshold=calibrated_threshold,
    )
    value_fn = TDCriticValueFunction(args.td_critic_checkpoint, device=device)
    uncertainty_signal = TDErrorSignal(value_fn=value_fn, gamma=args.td_gamma)
    return controller, uncertainty_signal, calibrated_threshold


def main() -> None:
    args = parse_args()
    ensure_mujoco_build_paths()
    task = TASKS[args.env]
    data_dir = Path(os.environ.get("DPPO_DATA_DIR", REPO_ROOT / "data"))
    log_dir = Path(os.environ.get("DPPO_LOG_DIR", REPO_ROOT / "log"))
    checkpoint_path = args.base_policy_checkpoint or log_dir / task["checkpoint"]
    normalization_path = (
        args.normalization_path
        or data_dir / "robomimic" / task["env_name"] / "normalization.npz"
    )
    env_meta_path = (
        args.env_meta_path
        or REPO_ROOT / "cfg" / "robomimic" / "env_meta" / f"{task['env_name']}.json"
    )

    for required_path in (checkpoint_path, normalization_path, env_meta_path):
        if not required_path.exists():
            raise FileNotFoundError(f"required file not found: {required_path}")

    reference_config = resolve_baseline_reference_config(
        method=args.method,
        chunk_horizon=args.chunk_horizon,
        denoise_steps=args.denoise_steps,
    )
    if (
        args.method == "denoise_only"
        and max(args.denoise_candidates) > reference_config.denoise_steps
    ):
        raise ValueError(
            "denoise_only candidate steps cannot exceed --denoise_steps; "
            f"got max candidate {max(args.denoise_candidates)} and "
            f"denoise_steps={reference_config.denoise_steps}"
        )

    policy_horizon = int(task["policy_horizon"])
    if reference_config.chunk_horizon > policy_horizon:
        raise ValueError(
            f"chunk_horizon={reference_config.chunk_horizon} cannot exceed "
            f"policy_horizon={policy_horizon}"
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    controller, uncertainty_signal, td_calibrated_threshold = (
        build_controller_and_uncertainty(args, device)
    )
    max_episode_steps = (
        int(args.max_episode_steps)
        if args.max_episode_steps is not None
        else 20
        if args.debug
        else int(task["max_episode_steps"])
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_root
        / f"{timestamp}_{args.env}_{args.method}_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    resolved_config = {
        "env": args.env,
        "env_name": task["env_name"],
        "method": args.method,
        "chunk_horizon": reference_config.chunk_horizon,
        "denoise_steps": reference_config.denoise_steps,
        "policy_horizon": policy_horizon,
        "episodes": args.episodes,
        "seed": args.seed,
        "debug": args.debug,
        "requested_max_episode_steps": args.max_episode_steps,
        "save_transitions": args.save_transitions,
        "td_critic_checkpoint": str(args.td_critic_checkpoint)
        if args.td_critic_checkpoint
        else None,
        "td_threshold_mode": args.td_threshold_mode,
        "td_threshold": args.td_threshold,
        "td_threshold_percentile": args.td_threshold_percentile,
        "td_calibration": str(args.td_calibration) if args.td_calibration else None,
        "td_calibrated_threshold": td_calibrated_threshold,
        "td_gamma": args.td_gamma,
        "denoise_candidates": list(args.denoise_candidates),
        "denoise_uncertainty_thresholds": list(args.denoise_uncertainty_thresholds),
        "device": device,
        "max_episode_steps": max_episode_steps,
        "checkpoint_path": str(checkpoint_path),
        "normalization_path": str(normalization_path),
        "env_meta_path": str(env_meta_path),
        "output_dir": str(output_dir),
    }
    OmegaConf.save(OmegaConf.create(resolved_config), output_dir / "config.yaml")

    wrappers = OmegaConf.create(
        {
            "robomimic_lowdim": {
                "normalization_path": str(normalization_path),
                "low_dim_keys": [
                    "robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_gripper_qpos",
                    "object",
                ],
            },
            "multi_step": {
                "n_obs_steps": 1,
                "n_action_steps": 1,
                "max_episode_steps": max_episode_steps,
                "reset_within_step": True,
            },
        }
    )
    env = make_async(
        task["env_name"],
        num_envs=1,
        asynchronous=False,
        max_episode_steps=max_episode_steps,
        wrappers=wrappers,
        robomimic_env_cfg_path=str(env_meta_path),
        use_image_obs=False,
        render=False,
        render_offscreen=False,
        obs_dim=int(task["obs_dim"]),
        action_dim=int(task["action_dim"]),
    )
    env.seed([args.seed])

    network = DiffusionMLP(**diffusion_mlp_kwargs(task))
    model = DiffusionModel(
        network=network,
        network_path=str(checkpoint_path),
        horizon_steps=policy_horizon,
        obs_dim=int(task["obs_dim"]),
        action_dim=int(task["action_dim"]),
        denoising_steps=reference_config.denoise_steps,
        predict_epsilon=True,
        denoised_clip_value=1.0,
        randn_clip_value=3,
        device=device,
    )
    policy = TorchDiffusionPolicyAdapter(
        model=model,
        action_dim=int(task["action_dim"]),
        chunk_horizon=reference_config.chunk_horizon,
        default_denoise_steps=reference_config.denoise_steps,
        device=device,
    )
    executor = AdaptiveDiffusionExecutor(
        env=env,
        diffusion_policy=policy,
        controller=controller,
        config={
            "method": args.method,
            "env_name": task["env_name"],
            "seed": args.seed,
            "max_episode_steps": max_episode_steps,
            "success_reward_threshold": 1.0,
            "save_transitions": args.save_transitions,
        },
        uncertainty_signal=uncertainty_signal,
    )

    try:
        for episode_id in range(args.episodes):
            executor.run_episode(
                episode_id=episode_id,
                deterministic=True,
                seed=args.seed + episode_id,
            )
        summary = executor.save_metrics(output_dir)
    finally:
        env.close()

    print(f"Wrote metrics to {output_dir}")
    print(OmegaConf.to_yaml(OmegaConf.create(summary), resolve=True))


if __name__ == "__main__":
    main()
