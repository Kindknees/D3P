#!/usr/bin/env python3
"""Evaluate a saved adaptive diffusion policy checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptive_diffusion import (  # noqa: E402
    ReplanFeatureConfig,
    load_replan_actor_critic_checkpoint,
)
from scripts.train_replan_ppo import (  # noqa: E402
    TASKS,
    build_executor,
    copy_if_exists,
    resolve_max_episode_steps,
    str_to_bool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        choices=["fake", *sorted(TASKS)],
        default=None,
        help="Defaults to the env stored in the checkpoint config.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", type=str_to_bool, default=True)
    parser.add_argument("--max_episode_steps", type=int, default=None)
    parser.add_argument("--chunk_horizon", type=int, default=None)
    parser.add_argument("--denoise_steps", type=int, default=None)
    parser.add_argument("--lambda_C", type=float, default=None)
    parser.add_argument("--lambda_D", type=float, default=None)
    parser.add_argument("--uncertainty_replan_bonus", type=float, default=None)
    parser.add_argument("--td_critic_checkpoint", type=Path, default=None)
    parser.add_argument("--td_gamma", type=float, default=None)
    parser.add_argument("--base_policy_checkpoint", type=Path, default=None)
    parser.add_argument("--normalization_path", type=Path, default=None)
    parser.add_argument("--env_meta_path", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=REPO_ROOT / "outputs" / "evals",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    model, metadata = load_replan_actor_critic_checkpoint(args.checkpoint)
    checkpoint_config = dict(metadata.get("config", {}))
    eval_args = build_eval_args(args, checkpoint_config)
    feature_config = feature_config_from_checkpoint(checkpoint_config, eval_args)
    if model.input_dim != feature_config.input_dim:
        raise ValueError(
            f"checkpoint input_dim {model.input_dim} does not match "
            f"eval feature dim {feature_config.input_dim}"
        )
    max_episode_steps = resolve_max_episode_steps(eval_args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_root
        / f"{timestamp}_{eval_args.env}_ppo_replan_only_eval_seed{eval_args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    resolved_config = {
        "env": eval_args.env,
        "method": "ppo_replan_only",
        "checkpoint": str(args.checkpoint),
        "seed": eval_args.seed,
        "episodes": args.episodes,
        "deterministic": args.deterministic,
        "max_episode_steps": max_episode_steps,
        "chunk_horizon": eval_args.chunk_horizon,
        "denoise_steps": eval_args.denoise_steps,
        "feature_config": {
            "obs_dim": feature_config.obs_dim,
            "action_dim": feature_config.action_dim,
            "horizon": feature_config.horizon,
            "input_dim": feature_config.input_dim,
        },
        "reward": {
            "lambda_C": eval_args.lambda_C,
            "lambda_D": eval_args.lambda_D,
            "replan_cost_c0": eval_args.replan_cost_c0,
            "replan_cost_c1": eval_args.replan_cost_c1,
            "uncertainty_replan_bonus": eval_args.uncertainty_replan_bonus,
        },
        "td_critic_checkpoint": str(eval_args.td_critic_checkpoint)
        if eval_args.td_critic_checkpoint
        else None,
        "base_policy_checkpoint": str(eval_args.base_policy_checkpoint)
        if eval_args.base_policy_checkpoint
        else None,
        "normalization_path": str(eval_args.normalization_path)
        if eval_args.normalization_path
        else None,
        "env_meta_path": str(eval_args.env_meta_path)
        if eval_args.env_meta_path
        else None,
        "td_gamma": eval_args.td_gamma,
        "device": eval_args.device,
        "output_dir": str(output_dir),
    }
    OmegaConf.save(OmegaConf.create(resolved_config), output_dir / "config.yaml")

    executor, env = build_executor(
        args=eval_args,
        model=model,
        feature_config=feature_config,
        deterministic=args.deterministic,
        max_episode_steps=max_episode_steps,
        device=eval_args.device,
    )
    try:
        for episode_id in range(args.episodes):
            executor.run_episode(
                episode_id=episode_id,
                deterministic=args.deterministic,
                seed=eval_args.seed + episode_id,
            )
        summary = executor.save_metrics(output_dir)
    finally:
        env.close()

    copy_if_exists(output_dir / "metrics_step.csv", output_dir / "eval_metrics_step.csv")
    copy_if_exists(
        output_dir / "metrics_episode.csv",
        output_dir / "eval_metrics_episode.csv",
    )
    with (output_dir / "eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote eval to {output_dir}")
    print(OmegaConf.to_yaml(OmegaConf.create(summary), resolve=True))


def build_eval_args(
    args: argparse.Namespace,
    checkpoint_config: dict[str, Any],
) -> SimpleNamespace:
    env = args.env or str(checkpoint_config.get("env", "fake"))
    reward_config = dict(checkpoint_config.get("reward", {}))
    feature_config = dict(checkpoint_config.get("feature_config", {}))
    checkpoint_td_critic = checkpoint_config.get("td_critic_checkpoint")
    td_critic_checkpoint = args.td_critic_checkpoint
    if td_critic_checkpoint is None and checkpoint_td_critic:
        td_critic_checkpoint = Path(str(checkpoint_td_critic))
    base_policy_checkpoint = getattr(args, "base_policy_checkpoint", None)
    if base_policy_checkpoint is None and checkpoint_config.get("base_policy_checkpoint"):
        base_policy_checkpoint = Path(str(checkpoint_config["base_policy_checkpoint"]))
    normalization_path = getattr(args, "normalization_path", None)
    if normalization_path is None and checkpoint_config.get("normalization_path"):
        normalization_path = Path(str(checkpoint_config["normalization_path"]))
    env_meta_path = getattr(args, "env_meta_path", None)
    if env_meta_path is None and checkpoint_config.get("env_meta_path"):
        env_meta_path = Path(str(checkpoint_config["env_meta_path"]))
    return SimpleNamespace(
        env=env,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps
        if args.max_episode_steps is not None
        else checkpoint_config.get("max_episode_steps"),
        chunk_horizon=int(
            args.chunk_horizon
            if args.chunk_horizon is not None
            else checkpoint_config.get("chunk_horizon", feature_config.get("horizon", 4))
        ),
        denoise_steps=int(
            args.denoise_steps
            if args.denoise_steps is not None
            else checkpoint_config.get("denoise_steps", 20)
        ),
        fake_obs_dim=int(feature_config.get("obs_dim", 3)),
        fake_action_dim=int(feature_config.get("action_dim", 2)),
        lambda_C=float(
            args.lambda_C
            if args.lambda_C is not None
            else reward_config.get("lambda_C", 0.0)
        ),
        lambda_D=float(
            args.lambda_D
            if args.lambda_D is not None
            else reward_config.get("lambda_D", 0.0)
        ),
        replan_cost_c0=float(reward_config.get("replan_cost_c0", 0.0)),
        replan_cost_c1=float(reward_config.get("replan_cost_c1", 1.0)),
        uncertainty_replan_bonus=float(
            args.uncertainty_replan_bonus
            if args.uncertainty_replan_bonus is not None
            else reward_config.get("uncertainty_replan_bonus", 0.0)
        ),
        td_critic_checkpoint=td_critic_checkpoint,
        base_policy_checkpoint=base_policy_checkpoint,
        normalization_path=normalization_path,
        env_meta_path=env_meta_path,
        td_gamma=float(
            args.td_gamma
            if args.td_gamma is not None
            else checkpoint_config.get("td_gamma", 0.99)
        ),
        device=args.device or str(checkpoint_config.get("device", "cpu")),
    )


def feature_config_from_checkpoint(
    checkpoint_config: dict[str, Any],
    eval_args: SimpleNamespace,
) -> ReplanFeatureConfig:
    feature_config = dict(checkpoint_config.get("feature_config", {}))
    if eval_args.env == "fake":
        obs_dim = int(feature_config.get("obs_dim", eval_args.fake_obs_dim))
        action_dim = int(feature_config.get("action_dim", eval_args.fake_action_dim))
    else:
        task = TASKS[eval_args.env]
        obs_dim = int(task["obs_dim"])
        action_dim = int(task["action_dim"])
    return ReplanFeatureConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        horizon=eval_args.chunk_horizon,
    )


if __name__ == "__main__":
    main()
