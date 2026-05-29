#!/usr/bin/env python3
"""Train a replan-only PPO controller."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptive_diffusion import (  # noqa: E402
    AdaptiveDiffusionExecutor,
    PPOBCRegularizerBatch,
    FixedChunkController,
    PPOReplanController,
    PPOUpdateConfig,
    ReplanActorCritic,
    ReplanFeatureConfig,
    ReplanRolloutBuffer,
    TDCriticValueFunction,
    TDErrorHeuristicController,
    TDErrorSignal,
    TorchDiffusionPolicyAdapter,
    ppo_config_to_dict,
    save_replan_actor_critic_checkpoint,
    update_replan_ppo,
)
from adaptive_diffusion.rl.collector import PPOExecutorRolloutCollector  # noqa: E402
from adaptive_diffusion.rl.features import (  # noqa: E402
    build_replan_features,
    replan_action_mask,
)


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


@dataclass(frozen=True)
class BCWarmstartExamples:
    obs_features: np.ndarray
    action_masks: np.ndarray
    actions: np.ndarray
    forced_replans: np.ndarray

    @property
    def size(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class BCWarmstartStats:
    examples: int
    epochs: int
    minibatch_size: int
    final_loss: float
    accuracy: float
    replan_fraction: float
    forced_replan_fraction: float
    learned_replan_fraction: float
    learned_replan_weight: float
    mean_sample_weight: float


class _Space:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class FakeDiffusionPolicy:
    """Small deterministic chunk sampler for PPO training smoke tests."""

    def __init__(
        self,
        action_dim: int,
        chunk_horizon: int,
        denoise_steps: int,
    ) -> None:
        self.action_dim = int(action_dim)
        self.chunk_horizon = int(chunk_horizon)
        self.default_denoise_steps = int(denoise_steps)
        self.sample_calls = 0
        self._base_action = np.linspace(
            0.25,
            -0.25,
            self.action_dim,
            dtype=np.float32,
        )

    def reset(self) -> None:
        pass

    def sample_action_chunk(
        self,
        obs: Any,
        denoise_steps: int | None = None,
        deterministic: bool = False,
        return_info: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.sample_calls += 1
        steps = int(denoise_steps or self.default_denoise_steps)
        state = np.asarray(obs.get("state", np.zeros(1)), dtype=np.float32).reshape(-1)
        offset = 0.05 * np.tanh(state[0]) if state.size else 0.0
        action = self._base_action + np.float32(offset)
        chunk = np.tile(action.reshape(1, -1), (self.chunk_horizon, 1)).astype(np.float32)
        return chunk, {
            "denoise_steps": steps,
            "nfe": steps,
            "wall_time_sec": 0.0,
            "sampler": "fake",
        }


class FakeVectorEnv:
    """Vector-env shaped fake task used to test the PPO loop quickly."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_episode_steps: int,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.max_episode_steps = int(max_episode_steps)
        self.single_action_space = _Space((1, self.action_dim))
        self.step_count = 0
        self._rng = np.random.default_rng(0)
        self._target_action = np.linspace(0.25, -0.25, self.action_dim, dtype=np.float32)

    def seed(self, seeds: list[int]) -> None:
        self._rng = np.random.default_rng(int(seeds[0]))

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self.step_count = 0
        return {"state": self._state()}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[
        dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        action_vec = np.asarray(action, dtype=np.float32).reshape(-1)[-self.action_dim :]
        self.step_count += 1
        distance = float(np.linalg.norm(action_vec - self._target_action))
        tracking_reward = max(0.0, 1.0 - distance)
        success = self.step_count >= self.max_episode_steps
        reward = 0.1 * tracking_reward + (1.0 if success else 0.0)
        obs = {"state": self._state()}
        return (
            obs,
            np.asarray([reward], dtype=np.float32),
            np.asarray([success], dtype=bool),
            np.asarray([False], dtype=bool),
            [{"is_success": np.asarray([{"task": success}], dtype=object)}],
        )

    def close(self) -> None:
        pass

    def _state(self) -> np.ndarray:
        state = np.zeros((1, 1, self.obs_dim), dtype=np.float32)
        state[0, 0, 0] = self.step_count / max(1, self.max_episode_steps)
        if self.obs_dim > 1:
            state[0, 0, 1] = np.sin(float(self.step_count))
        if self.obs_dim > 2:
            state[0, 0, 2] = np.cos(float(self.step_count))
        return state


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


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        choices=["fake", *sorted(TASKS)],
        default="fake",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_env_steps", type=int, default=64)
    parser.add_argument("--rollout_steps", type=int, default=32)
    parser.add_argument("--max_episode_steps", type=int, default=None)
    parser.add_argument("--eval_episodes", type=int, default=2)
    parser.add_argument("--chunk_horizon", type=int, default=4)
    parser.add_argument("--denoise_steps", type=int, default=20)
    parser.add_argument("--fake_obs_dim", type=int, default=3)
    parser.add_argument("--fake_action_dim", type=int, default=2)
    parser.add_argument("--hidden_dims", type=str, default="256,256")
    parser.add_argument("--activation", choices=["tanh", "relu"], default="tanh")
    parser.add_argument("--learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--update_epochs", type=int, default=5)
    parser.add_argument("--normalize_advantages", type=str_to_bool, default=True)
    parser.add_argument("--lambda_C", type=float, default=0.003)
    parser.add_argument("--lambda_D", type=float, default=0.03)
    parser.add_argument("--replan_cost_c0", type=float, default=0.0)
    parser.add_argument("--replan_cost_c1", type=float, default=1.0)
    parser.add_argument("--uncertainty_replan_bonus", type=float, default=0.0)
    parser.add_argument("--td_critic_checkpoint", type=Path, default=None)
    parser.add_argument("--td_gamma", type=float, default=0.99)
    parser.add_argument(
        "--bc_teacher",
        choices=["fixed", "td_error"],
        default="td_error",
    )
    parser.add_argument("--bc_warmstart_steps", type=int, default=0)
    parser.add_argument("--bc_epochs", type=int, default=3)
    parser.add_argument("--bc_minibatch_size", type=int, default=256)
    parser.add_argument("--bc_learning_rate", type=float, default=None)
    parser.add_argument("--bc_learned_replan_weight", type=float, default=1.0)
    parser.add_argument("--bc_regularizer_coef", type=float, default=0.0)
    parser.add_argument("--bc_regularizer_minibatch_size", type=int, default=None)
    parser.add_argument("--bc_td_calibration", type=Path, default=None)
    parser.add_argument("--bc_td_threshold_percentile", type=float, default=95.0)
    parser.add_argument("--base_policy_checkpoint", type=Path, default=None)
    parser.add_argument("--normalization_path", type=Path, default=None)
    parser.add_argument("--env_meta_path", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=REPO_ROOT / "outputs" / "runs",
    )
    return parser.parse_args()


def ensure_mujoco_build_paths() -> None:
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


def main() -> None:
    args = parse_args()
    if args.total_env_steps <= 0:
        raise ValueError("--total_env_steps must be positive")
    if args.rollout_steps <= 0:
        raise ValueError("--rollout_steps must be positive")
    if args.bc_warmstart_steps < 0:
        raise ValueError("--bc_warmstart_steps must be non-negative")
    if args.bc_epochs <= 0:
        raise ValueError("--bc_epochs must be positive")
    if args.bc_minibatch_size <= 0:
        raise ValueError("--bc_minibatch_size must be positive")
    if args.bc_learned_replan_weight <= 0.0:
        raise ValueError("--bc_learned_replan_weight must be positive")
    if args.bc_regularizer_coef < 0.0:
        raise ValueError("--bc_regularizer_coef must be non-negative")
    if (
        args.bc_regularizer_minibatch_size is not None
        and args.bc_regularizer_minibatch_size <= 0
    ):
        raise ValueError("--bc_regularizer_minibatch_size must be positive")
    if args.bc_regularizer_coef > 0.0 and args.bc_warmstart_steps <= 0:
        raise ValueError("--bc_regularizer_coef requires --bc_warmstart_steps > 0")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    max_episode_steps = resolve_max_episode_steps(args)
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    feature_config = resolve_feature_config(args)
    model = ReplanActorCritic(
        input_dim=feature_config.input_dim,
        hidden_dims=hidden_dims,
        activation=args.activation,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    ppo_config = PPOUpdateConfig(
        clip_ratio=args.clip_coef,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        max_grad_norm=args.max_grad_norm,
        normalize_advantages=args.normalize_advantages,
        bc_regularizer_coef=args.bc_regularizer_coef,
        bc_regularizer_minibatch_size=args.bc_regularizer_minibatch_size,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"{timestamp}_{args.env}_ppo_replan_only_seed{args.seed}"
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = build_resolved_config(
        args=args,
        feature_config=feature_config,
        hidden_dims=hidden_dims,
        ppo_config=ppo_config,
        max_episode_steps=max_episode_steps,
        device=device,
        output_dir=output_dir,
    )
    OmegaConf.save(OmegaConf.create(resolved_config), output_dir / "config.yaml")

    bc_warmstart_summary = {}
    bc_regularizer_batch = None
    if args.bc_warmstart_steps > 0:
        bc_stats, bc_examples = run_bc_warmstart(
            args=args,
            model=model,
            feature_config=feature_config,
            max_episode_steps=max_episode_steps,
            device=device,
            output_dir=output_dir,
        )
        bc_warmstart_summary = asdict(bc_stats)
        if args.bc_regularizer_coef > 0.0:
            bc_regularizer_batch = build_bc_regularizer_batch(
                bc_examples,
                learned_replan_weight=args.bc_learned_replan_weight,
            )

    executor, train_env = build_executor(
        args=args,
        model=model,
        feature_config=feature_config,
        deterministic=False,
        max_episode_steps=max_episode_steps,
        device=device,
    )
    update_rows = []
    env_steps = 0
    episode_id = 0
    update_id = 0
    try:
        while env_steps < args.total_env_steps:
            rollout = ReplanRolloutBuffer(
                capacity=args.rollout_steps + max_episode_steps,
                feature_dim=feature_config.input_dim,
            )
            episode_id, collected_steps = collect_rollout(
                executor=executor,
                rollout=rollout,
                start_episode_id=episode_id,
                target_steps=args.rollout_steps,
                seed=args.seed,
            )
            rollout.compute_returns_and_advantages(
                next_value=0.0,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )
            batch = rollout.as_batch(normalize_advantages=False)
            stats = update_replan_ppo(
                model=model,
                optimizer=optimizer,
                batch=batch,
                config=ppo_config,
                device=device,
                seed=args.seed + update_id,
                bc_regularizer_batch=bc_regularizer_batch,
            )
            env_steps += collected_steps
            update_row = build_update_row(
                update_id=update_id,
                env_steps=env_steps,
                rollout=rollout,
                stats=stats,
            )
            update_rows.append(update_row)
            save_replan_actor_critic_checkpoint(
                checkpoints_dir / "latest.pt",
                model,
                metadata={
                    "update_id": update_id,
                    "env_steps": env_steps,
                    "config": resolved_config,
                    "ppo": ppo_config_to_dict(ppo_config),
                    "bc_warmstart": bc_warmstart_summary,
                    "update_stats": asdict(stats),
                },
            )
            update_id += 1
    finally:
        train_summary = executor.save_metrics(output_dir)
        train_env.close()

    write_csv(output_dir / "training_updates.csv", update_rows)
    eval_summary = run_eval(
        args=args,
        model=model,
        feature_config=feature_config,
        max_episode_steps=max_episode_steps,
        device=device,
        output_dir=output_dir / "eval",
    )
    save_replan_actor_critic_checkpoint(
        checkpoints_dir / "best.pt",
        model,
        metadata={
            "update_id": update_id - 1,
            "env_steps": env_steps,
            "config": resolved_config,
            "ppo": ppo_config_to_dict(ppo_config),
            "bc_warmstart": bc_warmstart_summary,
            "train_summary": train_summary,
            "eval_summary": eval_summary,
        },
    )
    summary = {
        "env": args.env,
        "method": "ppo_replan_only",
        "seed": args.seed,
        "requested_total_env_steps": args.total_env_steps,
        "actual_env_steps": env_steps,
        "updates": update_id,
        "episodes": len(executor.episode_metrics),
        "bc_warmstart": bc_warmstart_summary,
        "latest_checkpoint": str(checkpoints_dir / "latest.pt"),
        "best_checkpoint": str(checkpoints_dir / "best.pt"),
        "train_summary": train_summary,
        "eval_summary": eval_summary,
        "last_update": update_rows[-1] if update_rows else {},
    }
    with (output_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote PPO training run to {output_dir}")
    print(OmegaConf.to_yaml(OmegaConf.create(summary), resolve=True))


def resolve_feature_config(args: argparse.Namespace) -> ReplanFeatureConfig:
    if args.env == "fake":
        return ReplanFeatureConfig(
            obs_dim=args.fake_obs_dim,
            action_dim=args.fake_action_dim,
            horizon=args.chunk_horizon,
        )
    task = TASKS[args.env]
    return ReplanFeatureConfig(
        obs_dim=int(task["obs_dim"]),
        action_dim=int(task["action_dim"]),
        horizon=args.chunk_horizon,
    )


def resolve_max_episode_steps(args: argparse.Namespace) -> int:
    if args.max_episode_steps is not None:
        return int(args.max_episode_steps)
    if args.env == "fake":
        return 8
    return int(TASKS[args.env]["max_episode_steps"])


def build_executor(
    args: argparse.Namespace,
    model: ReplanActorCritic,
    feature_config: ReplanFeatureConfig,
    deterministic: bool,
    max_episode_steps: int,
    device: str,
) -> tuple[AdaptiveDiffusionExecutor, Any]:
    controller = PPOReplanController(
        model=model,
        feature_config=feature_config,
        deterministic=deterministic,
        device=device,
    )
    if args.env == "fake":
        env = FakeVectorEnv(
            obs_dim=feature_config.obs_dim,
            action_dim=feature_config.action_dim,
            max_episode_steps=max_episode_steps,
        )
        policy = FakeDiffusionPolicy(
            action_dim=feature_config.action_dim,
            chunk_horizon=args.chunk_horizon,
            denoise_steps=args.denoise_steps,
        )
        uncertainty_signal = None
        env_name = "fake"
    else:
        env, policy = build_robomimic_env_and_policy(
            args=args,
            max_episode_steps=max_episode_steps,
            device=device,
        )
        uncertainty_signal = build_uncertainty_signal(args=args, device=device)
        env_name = TASKS[args.env]["env_name"]
    executor = AdaptiveDiffusionExecutor(
        env=env,
        diffusion_policy=policy,
        controller=controller,
        config={
            "method": "ppo_replan_only",
            "env_name": env_name,
            "seed": args.seed,
            "max_episode_steps": max_episode_steps,
            "success_reward_threshold": 1.0,
            "lambda_C": args.lambda_C,
            "lambda_D": args.lambda_D,
            "replan_cost_c0": args.replan_cost_c0,
            "replan_cost_c1": args.replan_cost_c1,
            "uncertainty_replan_bonus": args.uncertainty_replan_bonus,
        },
        uncertainty_signal=uncertainty_signal,
    )
    return executor, env


def build_robomimic_env_and_policy(
    args: argparse.Namespace,
    max_episode_steps: int,
    device: str,
) -> tuple[Any, TorchDiffusionPolicyAdapter]:
    ensure_mujoco_build_paths()
    from env.gym_utils import make_async
    from model.diffusion.diffusion import DiffusionModel
    from model.diffusion.mlp_diffusion import DiffusionMLP

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

    policy_horizon = int(task["policy_horizon"])
    if args.chunk_horizon > policy_horizon:
        raise ValueError(
            f"chunk_horizon={args.chunk_horizon} cannot exceed policy_horizon={policy_horizon}"
        )
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
    diffusion_model = DiffusionModel(
        network=network,
        network_path=str(checkpoint_path),
        horizon_steps=policy_horizon,
        obs_dim=int(task["obs_dim"]),
        action_dim=int(task["action_dim"]),
        denoising_steps=args.denoise_steps,
        predict_epsilon=True,
        denoised_clip_value=1.0,
        randn_clip_value=3,
        device=device,
    )
    policy = TorchDiffusionPolicyAdapter(
        model=diffusion_model,
        action_dim=int(task["action_dim"]),
        chunk_horizon=args.chunk_horizon,
        default_denoise_steps=args.denoise_steps,
        device=device,
    )
    return env, policy


def build_uncertainty_signal(args: argparse.Namespace, device: str):
    if args.td_critic_checkpoint is None:
        return None
    if not args.td_critic_checkpoint.exists():
        raise FileNotFoundError(f"TD critic checkpoint not found: {args.td_critic_checkpoint}")
    value_fn = TDCriticValueFunction(args.td_critic_checkpoint, device=device)
    return TDErrorSignal(value_fn=value_fn, gamma=args.td_gamma)


def load_td_percentile_threshold(path: Path, percentile: float) -> float:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    thresholds = data.get("percentile_thresholds", {})
    key_candidates = [f"{float(percentile):g}"]
    if float(percentile).is_integer():
        key_candidates.append(str(int(percentile)))
    for key in key_candidates:
        if key in thresholds:
            return float(thresholds[key])
    raise KeyError(
        f"percentile {percentile:g} not found in calibration file {path}"
    )


def build_bc_teacher_controller(args: argparse.Namespace):
    if args.bc_teacher == "fixed":
        return FixedChunkController()
    if args.bc_teacher != "td_error":
        raise ValueError(f"unsupported BC teacher: {args.bc_teacher}")
    if args.td_critic_checkpoint is None:
        raise ValueError("--bc_teacher td_error requires --td_critic_checkpoint")
    if args.bc_td_calibration is None:
        raise ValueError("--bc_teacher td_error requires --bc_td_calibration")
    if not args.bc_td_calibration.exists():
        raise FileNotFoundError(f"BC TD calibration not found: {args.bc_td_calibration}")
    controller = TDErrorHeuristicController(
        threshold_mode="percentile",
        threshold_percentile=args.bc_td_threshold_percentile,
    )
    controller.calibrated_threshold = load_td_percentile_threshold(
        args.bc_td_calibration,
        args.bc_td_threshold_percentile,
    )
    return controller


def build_bc_teacher_executor(
    args: argparse.Namespace,
    feature_config: ReplanFeatureConfig,
    max_episode_steps: int,
    device: str,
) -> tuple[AdaptiveDiffusionExecutor, Any]:
    teacher = build_bc_teacher_controller(args)
    if args.env == "fake":
        env = FakeVectorEnv(
            obs_dim=feature_config.obs_dim,
            action_dim=feature_config.action_dim,
            max_episode_steps=max_episode_steps,
        )
        policy = FakeDiffusionPolicy(
            action_dim=feature_config.action_dim,
            chunk_horizon=args.chunk_horizon,
            denoise_steps=args.denoise_steps,
        )
        uncertainty_signal = build_uncertainty_signal(args, device=device)
        env_name = "fake"
    else:
        env, policy = build_robomimic_env_and_policy(
            args=args,
            max_episode_steps=max_episode_steps,
            device=device,
        )
        uncertainty_signal = build_uncertainty_signal(args, device=device)
        env_name = TASKS[args.env]["env_name"]
    executor = AdaptiveDiffusionExecutor(
        env=env,
        diffusion_policy=policy,
        controller=teacher,
        config={
            "method": f"bc_{teacher.name}",
            "env_name": env_name,
            "seed": args.seed,
            "max_episode_steps": max_episode_steps,
            "success_reward_threshold": 1.0,
        },
        uncertainty_signal=uncertainty_signal,
    )
    return executor, env


def collect_bc_warmstart_examples(
    args: argparse.Namespace,
    feature_config: ReplanFeatureConfig,
    max_episode_steps: int,
    device: str,
    seed: int,
) -> BCWarmstartExamples:
    executor, env = build_bc_teacher_executor(
        args=args,
        feature_config=feature_config,
        max_episode_steps=max_episode_steps,
        device=device,
    )
    features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []
    forced_replans: list[bool] = []
    episode_id = 0
    try:
        while len(actions) < args.bc_warmstart_steps:
            executor.reset(seed=seed + 50_000 + episode_id)
            for t in range(max_episode_steps):
                current_obs = executor._single_obs(executor.obs)
                uncertainty = executor.uncertainty_signal.before_action(
                    current_obs,
                    executor.buffer,
                )
                action_mask = replan_action_mask(executor.buffer)
                decision = executor.controller.decide(
                    executor.buffer,
                    obs=current_obs,
                    uncertainty=uncertainty,
                )
                features.append(
                    build_replan_features(
                        obs=current_obs,
                        buffer=executor.buffer,
                        config=feature_config,
                        uncertainty=uncertainty,
                    )
                )
                masks.append(action_mask)
                actions.append(int(decision.action))
                forced_replans.append(bool(decision.forced_replan))
                row = executor.step(episode_id=episode_id, t=t, deterministic=True)
                if row["done"] or len(actions) >= args.bc_warmstart_steps:
                    break
            episode_id += 1
    finally:
        env.close()
    if not actions:
        raise RuntimeError("BC warm-start collection produced no examples")
    return BCWarmstartExamples(
        obs_features=np.stack(features).astype(np.float32),
        action_masks=np.stack(masks).astype(bool),
        actions=np.asarray(actions, dtype=np.int64),
        forced_replans=np.asarray(forced_replans, dtype=bool),
    )


def compute_bc_sample_weights(
    examples: BCWarmstartExamples,
    learned_replan_weight: float,
) -> np.ndarray:
    if learned_replan_weight <= 0.0:
        raise ValueError("learned_replan_weight must be positive")
    weights = np.ones(examples.size, dtype=np.float32)
    learned_replans = (examples.actions == 1) & (~examples.forced_replans)
    weights[learned_replans] = np.float32(learned_replan_weight)
    return weights


def pretrain_replan_actor_bc(
    model: ReplanActorCritic,
    examples: BCWarmstartExamples,
    epochs: int,
    minibatch_size: int,
    learning_rate: float,
    device: str,
    seed: int,
    learned_replan_weight: float = 1.0,
) -> BCWarmstartStats:
    torch_device = torch.device(device)
    model.to(torch_device)
    model.train()
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=float(learning_rate))
    features = torch.from_numpy(examples.obs_features).float().to(torch_device)
    masks = torch.from_numpy(examples.action_masks).bool().to(torch_device)
    actions = torch.from_numpy(examples.actions).long().to(torch_device)
    sample_weights_np = compute_bc_sample_weights(
        examples,
        learned_replan_weight=learned_replan_weight,
    )
    sample_weights = torch.from_numpy(sample_weights_np).float().to(torch_device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    final_loss = 0.0
    for _epoch in range(int(epochs)):
        permutation = torch.randperm(examples.size, generator=generator)
        for start in range(0, examples.size, int(minibatch_size)):
            batch_idx = permutation[start : start + int(minibatch_size)].to(torch_device)
            logits = model.actor(features[batch_idx])
            masked_logits = logits.masked_fill(
                ~masks[batch_idx],
                torch.finfo(logits.dtype).min,
            )
            per_example_loss = F.cross_entropy(
                masked_logits,
                actions[batch_idx],
                reduction="none",
            )
            batch_weights = sample_weights[batch_idx]
            weighted_loss = (per_example_loss * batch_weights).sum()
            normalizer = batch_weights.sum().clamp_min(1.0e-8)
            loss = weighted_loss / normalizer
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())
    model.eval()
    with torch.no_grad():
        logits = model.actor(features)
        masked_logits = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
        predictions = torch.argmax(masked_logits, dim=-1)
        accuracy = float((predictions == actions).float().mean().cpu().item())
    return BCWarmstartStats(
        examples=examples.size,
        epochs=int(epochs),
        minibatch_size=int(minibatch_size),
        final_loss=float(final_loss),
        accuracy=accuracy,
        replan_fraction=float(np.mean(examples.actions == 1)),
        forced_replan_fraction=float(np.mean(examples.forced_replans)),
        learned_replan_fraction=float(
            np.mean((examples.actions == 1) & (~examples.forced_replans))
        ),
        learned_replan_weight=float(learned_replan_weight),
        mean_sample_weight=float(np.mean(sample_weights_np)),
    )


def build_bc_regularizer_batch(
    examples: BCWarmstartExamples,
    learned_replan_weight: float,
) -> PPOBCRegularizerBatch:
    return PPOBCRegularizerBatch(
        obs_features=examples.obs_features,
        action_masks=examples.action_masks,
        actions=examples.actions,
        sample_weights=compute_bc_sample_weights(
            examples,
            learned_replan_weight=learned_replan_weight,
        ),
    )


def run_bc_warmstart(
    args: argparse.Namespace,
    model: ReplanActorCritic,
    feature_config: ReplanFeatureConfig,
    max_episode_steps: int,
    device: str,
    output_dir: Path,
) -> tuple[BCWarmstartStats, BCWarmstartExamples]:
    examples = collect_bc_warmstart_examples(
        args=args,
        feature_config=feature_config,
        max_episode_steps=max_episode_steps,
        device=device,
        seed=args.seed,
    )
    examples_path = output_dir / "bc_warmstart_examples.npz"
    np.savez_compressed(
        examples_path,
        obs_features=examples.obs_features,
        action_masks=examples.action_masks,
        actions=examples.actions,
        forced_replans=examples.forced_replans,
    )
    stats = pretrain_replan_actor_bc(
        model=model,
        examples=examples,
        epochs=args.bc_epochs,
        minibatch_size=args.bc_minibatch_size,
        learning_rate=args.bc_learning_rate
        if args.bc_learning_rate is not None
        else args.learning_rate,
        device=device,
        seed=args.seed,
        learned_replan_weight=args.bc_learned_replan_weight,
    )
    with (output_dir / "bc_warmstart_summary.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)
    return stats, examples


def collect_rollout(
    executor: AdaptiveDiffusionExecutor,
    rollout: ReplanRolloutBuffer,
    start_episode_id: int,
    target_steps: int,
    seed: int,
) -> tuple[int, int]:
    collector = PPOExecutorRolloutCollector(executor, rollout)
    episode_id = int(start_episode_id)
    start_size = rollout.size
    while rollout.size < target_steps:
        rows = collector.run_episode(
            episode_id=episode_id,
            deterministic=False,
            seed=seed + episode_id,
        )
        append_episode_metrics(executor, rows)
        episode_id += 1
    return episode_id, rollout.size - start_size


def append_episode_metrics(
    executor: AdaptiveDiffusionExecutor,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty episode")
    episode_length = len(rows)
    total_nfe = int(sum(row["nfe"] for row in rows))
    total_inference_time = float(sum(row["inference_wall_time_sec"] for row in rows))
    total_controller_time = float(sum(row["controller_wall_time_sec"] for row in rows))
    episode_return = float(sum(row["env_reward"] for row in rows))
    high_level_return = float(sum(row["hl_reward"] for row in rows))
    success = bool(
        any(row["success"] for row in rows)
        or max((row["env_reward"] for row in rows), default=0.0)
        >= executor.success_reward_threshold
    )
    replans = int(sum(1 for row in rows if row["replanned"]))
    forced_replans = int(sum(1 for row in rows if row["forced_replan"]))
    episode_row = {
        "episode_id": int(rows[0]["episode_id"]),
        "env_name": executor.env_name,
        "method": executor.method,
        "seed": executor.seed,
        "episode_return": episode_return,
        "high_level_return": high_level_return,
        "episode_length": episode_length,
        "success": success,
        "total_nfe": total_nfe,
        "mean_nfe_per_action": total_nfe / episode_length if episode_length else 0.0,
        "replans": replans,
        "forced_replans": forced_replans,
        "replan_rate": replans / episode_length if episode_length else 0.0,
        "mean_inference_wall_time_per_action": total_inference_time / episode_length
        if episode_length
        else 0.0,
        "mean_controller_wall_time_per_action": total_controller_time / episode_length
        if episode_length
        else 0.0,
        "mean_total_wall_time_per_action": (
            total_inference_time + total_controller_time
        )
        / episode_length
        if episode_length
        else 0.0,
        "discarded_actions": int(sum(row["discarded_actions"] for row in rows)),
    }
    executor.episode_metrics.append(episode_row)
    return episode_row


def build_update_row(
    update_id: int,
    env_steps: int,
    rollout: ReplanRolloutBuffer,
    stats: Any,
) -> dict[str, Any]:
    batch = rollout.as_batch(normalize_advantages=False)
    actions = batch.actions.tolist()
    return {
        "update_id": int(update_id),
        "env_steps": int(env_steps),
        "rollout_steps": int(rollout.size),
        "continue_actions": int(sum(1 for action in actions if action == 0)),
        "replan_actions": int(sum(1 for action in actions if action == 1)),
        "forced_replans": int(
            sum(
                1
                for mask, action in zip(batch.action_masks, actions)
                if not mask[0] and action == 1
            )
        ),
        "mean_hl_reward": float(np.mean(batch.rewards_hl)) if rollout.size else 0.0,
        "mean_env_reward": float(np.mean(batch.rewards_env)) if rollout.size else 0.0,
        "mean_nfe_per_action": float(np.mean(batch.nfe)) if rollout.size else 0.0,
        **asdict(stats),
    }


def run_eval(
    args: argparse.Namespace,
    model: ReplanActorCritic,
    feature_config: ReplanFeatureConfig,
    max_episode_steps: int,
    device: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    executor, env = build_executor(
        args=args,
        model=model,
        feature_config=feature_config,
        deterministic=True,
        max_episode_steps=max_episode_steps,
        device=device,
    )
    try:
        for episode_id in range(args.eval_episodes):
            executor.run_episode(
                episode_id=episode_id,
                deterministic=True,
                seed=args.seed + 10_000 + episode_id,
            )
        summary = executor.save_metrics(output_dir)
    finally:
        env.close()
    copy_if_exists(output_dir / "metrics_step.csv", output_dir / "eval_metrics_step.csv")
    copy_if_exists(output_dir / "metrics_episode.csv", output_dir / "eval_metrics_episode.csv")
    return summary


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.write_bytes(src.read_bytes())


def build_resolved_config(
    args: argparse.Namespace,
    feature_config: ReplanFeatureConfig,
    hidden_dims: tuple[int, ...],
    ppo_config: PPOUpdateConfig,
    max_episode_steps: int,
    device: str,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "env": args.env,
        "method": "ppo_replan_only",
        "seed": args.seed,
        "config": str(args.config) if args.config else None,
        "total_env_steps": args.total_env_steps,
        "rollout_steps": args.rollout_steps,
        "max_episode_steps": max_episode_steps,
        "eval_episodes": args.eval_episodes,
        "chunk_horizon": args.chunk_horizon,
        "denoise_steps": args.denoise_steps,
        "feature_config": {
            "obs_dim": feature_config.obs_dim,
            "action_dim": feature_config.action_dim,
            "horizon": feature_config.horizon,
            "input_dim": feature_config.input_dim,
        },
        "network": {
            "hidden_dims": list(hidden_dims),
            "activation": args.activation,
        },
        "ppo": ppo_config_to_dict(ppo_config),
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "learning_rate": args.learning_rate,
        "reward": {
            "lambda_C": args.lambda_C,
            "lambda_D": args.lambda_D,
            "replan_cost_c0": args.replan_cost_c0,
            "replan_cost_c1": args.replan_cost_c1,
            "uncertainty_replan_bonus": args.uncertainty_replan_bonus,
        },
        "bc_warmstart": {
            "teacher": args.bc_teacher,
            "steps": args.bc_warmstart_steps,
            "epochs": args.bc_epochs,
            "minibatch_size": args.bc_minibatch_size,
            "learning_rate": args.bc_learning_rate
            if args.bc_learning_rate is not None
            else args.learning_rate,
            "learned_replan_weight": args.bc_learned_replan_weight,
            "regularizer_coef": args.bc_regularizer_coef,
            "regularizer_minibatch_size": args.bc_regularizer_minibatch_size
            if args.bc_regularizer_minibatch_size is not None
            else args.minibatch_size,
            "td_calibration": str(args.bc_td_calibration)
            if args.bc_td_calibration
            else None,
            "td_threshold_percentile": args.bc_td_threshold_percentile,
        },
        "td_critic_checkpoint": str(args.td_critic_checkpoint)
        if args.td_critic_checkpoint
        else None,
        "base_policy_checkpoint": str(args.base_policy_checkpoint)
        if args.base_policy_checkpoint
        else None,
        "normalization_path": str(args.normalization_path)
        if args.normalization_path
        else None,
        "env_meta_path": str(args.env_meta_path) if args.env_meta_path else None,
        "td_gamma": args.td_gamma,
        "device": device,
        "output_dir": str(output_dir),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
