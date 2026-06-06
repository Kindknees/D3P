"""Adapters from repository vector envs to single-env executor APIs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SingleVectorEnvAdapter:
    """Use env index 0 of a vector env as a single step-by-step environment."""

    venv: object

    def reset(self, seed: int | None = None):
        options = {}
        if seed is not None:
            options["seed"] = seed
        obs = self.venv.reset_arg(options_list=[options])
        return _select_env0(obs)

    def step(self, action: np.ndarray):
        action = np.asarray(action)
        if action.ndim != 1:
            raise ValueError("SingleVectorEnvAdapter expects one flat action.")
        action_venv = action[None, None, :]
        obs, reward, terminated, truncated, info = self.venv.step(action_venv)
        done = bool(terminated[0] or truncated[0])
        info0 = info[0] if isinstance(info, list) else info
        return _select_env0(obs), float(reward[0]), done, info0

    def get_sim_state(self):
        result, success = self.venv.call_sync("get_sim_state", indices=[0])[0]
        if not success:
            raise RuntimeError("get_sim_state failed in vector env worker.")
        return result

    def reset_to_sim_state(self, state):
        result, success = self.venv.call_sync(
            "reset_to_sim_state",
            indices=[0],
            state=state,
        )[0]
        if not success:
            raise RuntimeError("reset_to_sim_state failed in vector env worker.")
        return _select_env0([result])


def _select_env0(obs):
    if isinstance(obs, dict):
        return {key: value[0] for key, value in obs.items()}
    if isinstance(obs, list):
        return obs[0]
    return obs[0]
