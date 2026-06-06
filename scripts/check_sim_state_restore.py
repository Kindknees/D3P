"""Smoke-check simulator save/restore through the eval vector env wrappers."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf

from adaptive_diffusion.vector_env_adapter import SingleVectorEnvAdapter
from env.gym_utils import make_async

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
OmegaConf.register_new_resolver("round_down", math.floor, replace=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--normalization-path", required=True)
    parser.add_argument("--task", default="lift")
    args = parser.parse_args()

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(
        config_dir=str(Path(args.config_path).resolve()),
        version_base=None,
    ):
        cfg = hydra.compose(
            config_name=args.config_name,
            overrides=[
                f"normalization_path={args.normalization_path}",
                "logdir=/tmp/adaptive_commitment_state_restore_smoke",
                "env.n_envs=1",
                "env.max_episode_steps=20",
                "n_steps=20",
            ],
        )
    OmegaConf.resolve(cfg)

    venv = make_async(
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
    env = SingleVectorEnvAdapter(venv)
    try:
        obs0 = env.reset(seed=0)
        state = env.get_sim_state()
        action = np.zeros(cfg.action_dim, dtype=np.float32)
        obs1, _, _, _ = env.step(action)
        obs2 = env.reset_to_sim_state(state)
    finally:
        venv.close()

    diff_restore = float(np.max(np.abs(obs2["state"] - obs0["state"])))
    diff_step = float(np.max(np.abs(obs1["state"] - obs0["state"])))
    print(f"task={args.task}")
    print(f"diff_after_step={diff_step}")
    print(f"diff_after_restore={diff_restore}")
    print("status: ready" if diff_restore < 1e-5 else "status: not_ready")
    if diff_restore >= 1e-5:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
