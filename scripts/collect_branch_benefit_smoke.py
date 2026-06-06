"""Collect a small intervention-benefit branch log from a saved sim state."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from adaptive_diffusion.diffusion_policy_adapter import (
    DiffusionPolicyAdapter,
    DiffusionResidualRepairer,
)
from adaptive_diffusion.repair import RepairConfig
from adaptive_diffusion.vector_env_adapter import SingleVectorEnvAdapter
from env.gym_utils import make_async


OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
OmegaConf.register_new_resolver("round_down", math.floor, replace=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default="cfg/robomimic/eval/lift")
    parser.add_argument("--config-name", default="eval_adaptive_commitment_mlp")
    parser.add_argument("--base-policy-path", required=True)
    parser.add_argument("--normalization-path", required=True)
    parser.add_argument("--output-dir", default="outputs/adaptive_commitment/branch_benefit_smoke/lift_seed0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--branch-horizon", type=int, default=300)
    parser.add_argument("--event-step", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = _load_cfg(args)
    model = hydra.utils.instantiate(cfg.model)
    policy = DiffusionPolicyAdapter(model=model, device=cfg.device, deterministic=True)
    repairer = DiffusionResidualRepairer(policy)
    repair_cfg = RepairConfig(target_len=cfg.horizon_steps, noise_ratio=0.5, anchor_rho=0.5)

    venv = _make_env(cfg)
    env = SingleVectorEnvAdapter(venv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        event = _capture_event(env, policy, cfg, args)
        rows = [_evaluate_event_branches(env, policy, repairer, repair_cfg, event, cfg, args)]
    finally:
        venv.close()

    path = output_dir / "branch_logs.csv"
    fields = [
        "task",
        "seed",
        "event_step",
        "phase",
        "remaining_buffer_length",
        "uncertainty",
        "residual_first_l2",
        "residual_mean_l2",
        "repair_action_jump",
        "repair_buffer_jump",
        "G_continue",
        "G_repair_050",
        "G_reset",
        "benefit_repair_050",
        "benefit_reset",
        "best_action",
    ]
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote branch logs to {path}")
    print(rows[0])


def _load_cfg(args):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(
        config_dir=str(Path(args.config_path).resolve()),
        version_base=None,
    ):
        cfg = hydra.compose(
            config_name=args.config_name,
            overrides=[
                f"base_policy_path={args.base_policy_path}",
                f"normalization_path={args.normalization_path}",
                f"logdir={args.output_dir}/hydra",
                f"device={args.device}",
                f"seed={args.seed}",
                f"env.max_episode_steps={args.branch_horizon}",
                f"n_steps={args.branch_horizon}",
                "env.n_envs=1",
                "act_steps=1",
            ],
        )
    OmegaConf.resolve(cfg)
    return cfg


def _make_env(cfg):
    return make_async(
        cfg.env.name,
        env_type=cfg.env.get("env_type", None),
        num_envs=cfg.env.n_envs,
        asynchronous=True,
        max_episode_steps=cfg.env.max_episode_steps,
        wrappers=cfg.env.get("wrappers", None),
        robomimic_env_cfg_path=cfg.get("robomimic_env_cfg_path", None),
        shape_meta=cfg.get("shape_meta", None),
        use_image_obs=cfg.env.get("use_image_obs", False),
        render=cfg.env.get("render", False),
        render_offscreen=cfg.env.get("save_video", False),
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        **cfg.env.specific if "specific" in cfg.env else {},
    )


def _capture_event(env, policy, cfg, args):
    obs = env.reset(seed=args.seed)
    chunk = policy.sample_action_chunk(obs)[: cfg.horizon_steps]
    branch_rng = _capture_rng()
    reward_prefix = 0.0
    for step in range(args.event_step):
        obs, reward, done, _ = env.step(chunk[step])
        reward_prefix += float(reward)
        if done:
            break
    residual = chunk[args.event_step : cfg.horizon_steps]
    uncertainty = policy.action_disagreement(obs, num_samples=2)
    return {
        "state": env.get_sim_state(),
        "obs": obs,
        "residual": residual,
        "reward_prefix": reward_prefix,
        "phase": args.event_step % cfg.horizon_steps,
        "remaining_buffer_length": len(residual),
        "uncertainty": uncertainty,
        "residual_first_l2": float(np.linalg.norm(residual[0])) if len(residual) else 0.0,
        "residual_mean_l2": float(np.mean(np.linalg.norm(residual, axis=1))) if len(residual) else 0.0,
        "rng": branch_rng,
    }


def _evaluate_event_branches(env, policy, repairer, repair_cfg, event, cfg, args):
    returns = {}
    repair_result = repairer(event["obs"], event["residual"], repair_cfg)
    branches = {
        "continue": event["residual"],
        "repair_050": repair_result.repaired_actions[: len(event["residual"])],
        "reset": None,
    }
    for name, initial_actions in branches.items():
        env.reset_to_sim_state(event["state"])
        _restore_rng(event["rng"])
        if initial_actions is None:
            initial_actions = policy.sample_action_chunk(event["obs"])[: cfg.horizon_steps]
        returns[name] = event["reward_prefix"] + _rollout_from_actions(
            env,
            policy,
            initial_actions,
            max_steps=args.branch_horizon - args.event_step,
        )
    benefits = {
        "repair_050": returns["repair_050"] - returns["continue"],
        "reset": returns["reset"] - returns["continue"],
    }
    best_action = max(returns, key=returns.get)
    return {
        "task": cfg.env_name,
        "seed": args.seed,
        "event_step": args.event_step,
        "phase": event["phase"],
        "remaining_buffer_length": event["remaining_buffer_length"],
        "uncertainty": event["uncertainty"],
        "residual_first_l2": event["residual_first_l2"],
        "residual_mean_l2": event["residual_mean_l2"],
        "repair_action_jump": repair_result.action_jump,
        "repair_buffer_jump": repair_result.buffer_jump,
        "G_continue": returns["continue"],
        "G_repair_050": returns["repair_050"],
        "G_reset": returns["reset"],
        "benefit_repair_050": benefits["repair_050"],
        "benefit_reset": benefits["reset"],
        "best_action": best_action,
    }


def _rollout_from_actions(env, policy, initial_actions, max_steps: int) -> float:
    total = 0.0
    obs = None
    done = False
    steps = 0
    for action in initial_actions:
        obs, reward, done, _ = env.step(action)
        total += float(reward)
        steps += 1
        if done or steps >= max_steps:
            return total
    while steps < max_steps:
        chunk = policy.sample_action_chunk(obs)
        for action in chunk:
            obs, reward, done, _ = env.step(action)
            total += float(reward)
            steps += 1
            if done or steps >= max_steps:
                return total
    return total


def _capture_rng():
    return {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }


def _restore_rng(state) -> None:
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])


if __name__ == "__main__":
    main()
