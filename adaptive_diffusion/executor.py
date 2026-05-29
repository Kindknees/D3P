"""Single-step action-buffer executor for adaptive diffusion policies."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from adaptive_diffusion.buffers import ActionBuffer
from adaptive_diffusion.controllers.fixed import FixedChunkController
from adaptive_diffusion.diffusion_interface import DiffusionPolicyInterface
from adaptive_diffusion.uncertainty import NullUncertaintySignal, UncertaintySignal


class AdaptiveDiffusionExecutor:
    """Coordinates buffer, controller, policy, environment, and metrics."""

    def __init__(
        self,
        env: Any,
        diffusion_policy: DiffusionPolicyInterface,
        controller: FixedChunkController,
        config: Mapping[str, Any],
        logger: Any = None,
        uncertainty_signal: Optional[UncertaintySignal] = None,
    ) -> None:
        self.env = env
        self.diffusion_policy = diffusion_policy
        self.controller = controller
        self.uncertainty_signal = uncertainty_signal or NullUncertaintySignal()
        self.config = dict(config)
        self.logger = logger
        self.method = self.config.get("method", getattr(controller, "name", "unknown"))
        self.env_name = self.config.get("env_name", "unknown")
        self.seed = int(self.config.get("seed", 0))
        self.max_episode_steps = int(self.config.get("max_episode_steps", 300))
        self.success_reward_threshold = float(
            self.config.get("success_reward_threshold", 1.0)
        )
        self.lambda_inference_cost = float(self.config.get("lambda_C", 0.0))
        self.lambda_discard = float(self.config.get("lambda_D", 0.0))
        self.replan_cost_c0 = float(self.config.get("replan_cost_c0", 0.0))
        self.replan_cost_c1 = float(self.config.get("replan_cost_c1", 0.0))
        self.uncertainty_replan_bonus = float(
            self.config.get("uncertainty_replan_bonus", 0.0)
        )
        self.should_save_transitions = bool(self.config.get("save_transitions", False))

        self.buffer = ActionBuffer.empty(
            horizon=diffusion_policy.chunk_horizon,
            action_dim=diffusion_policy.action_dim,
        )
        self.obs = None
        self.step_metrics: list[dict[str, Any]] = []
        self.episode_metrics: list[dict[str, Any]] = []
        self.transition_records: list[dict[str, Any]] = []

    def reset(self, seed: Optional[int] = None) -> dict[str, np.ndarray]:
        """Reset environment and clear policy/buffer state."""

        self.diffusion_policy.reset()
        self.uncertainty_signal.reset()
        self.buffer = ActionBuffer.empty(
            horizon=self.diffusion_policy.chunk_horizon,
            action_dim=self.diffusion_policy.action_dim,
        )
        self.obs = self.env.reset(seed=seed)
        return self._single_obs(self.obs)

    def step(self, episode_id: int, t: int, deterministic: bool = False) -> dict[str, Any]:
        """Execute one environment action, replanning first if required."""

        if self.obs is None:
            self.reset(seed=self.seed)

        buffer_remaining_before = self.buffer.remaining
        plan_age_before = self.buffer.age
        current_obs = self._single_obs(self.obs)
        obs_norm = self._obs_norm(current_obs)
        uncertainty = self.uncertainty_signal.before_action(current_obs, self.buffer)

        controller_start = time.perf_counter()
        decision = self.controller.decide(
            self.buffer,
            obs=current_obs,
            uncertainty=uncertainty,
        )
        controller_wall_time_sec = time.perf_counter() - controller_start
        replanned = bool(decision.action == 1)
        discarded_actions = buffer_remaining_before if replanned else 0

        sample_info = {
            "denoise_steps": 0,
            "nfe": 0,
            "wall_time_sec": 0.0,
            "sampler": "none",
        }
        if replanned:
            requested_denoise_steps = (
                decision.denoise_steps
                if decision.denoise_steps is not None
                else self.diffusion_policy.default_denoise_steps
            )
            chunk, sample_info = self.diffusion_policy.sample_action_chunk(
                current_obs,
                denoise_steps=requested_denoise_steps,
                deterministic=deterministic,
                return_info=True,
            )
            if chunk.shape[1] != self.diffusion_policy.action_dim:
                raise ValueError(
                    f"sampled action_dim {chunk.shape[1]} does not match "
                    f"{self.diffusion_policy.action_dim}"
                )
            self.buffer.replace(
                chunk[: self.buffer.horizon],
                denoise_steps=int(sample_info["denoise_steps"]),
                source=self.method,
            )

        action = self.buffer.pop_first()
        env_action = self._format_env_action(action)
        obs, reward, terminated, truncated, infos = self.env.step(env_action)
        self.obs = obs

        reward_value = float(np.asarray(reward).reshape(-1)[0])
        done = bool(np.asarray(terminated).reshape(-1)[0]) or bool(
            np.asarray(truncated).reshape(-1)[0]
        )
        info = infos[0] if isinstance(infos, list) and infos else infos
        success = self._extract_success(info, reward_value)
        next_obs = self._single_obs(obs)
        self.uncertainty_signal.after_transition(
            current_obs,
            action,
            reward_value,
            next_obs,
            done,
            info,
        )
        hl_reward = self._high_level_reward(
            reward=reward_value,
            replanned=replanned,
            buffer_remaining_before=buffer_remaining_before,
            nfe=int(sample_info["nfe"]),
            uncertainty=uncertainty.uncertainty,
        )

        row = {
            "episode_id": int(episode_id),
            "t": int(t),
            "env_name": self.env_name,
            "method": self.method,
            "seed": self.seed,
            "obs_norm": obs_norm,
            "controller_action": int(decision.action),
            "forced_replan": bool(decision.forced_replan),
            "replanned": replanned,
            "buffer_remaining_before": int(buffer_remaining_before),
            "buffer_remaining_after": int(self.buffer.remaining),
            "plan_age_before": int(plan_age_before),
            "discarded_actions": int(discarded_actions),
            "denoise_steps": int(sample_info["denoise_steps"]),
            "nfe": int(sample_info["nfe"]),
            "inference_wall_time_sec": float(sample_info["wall_time_sec"]),
            "controller_wall_time_sec": float(controller_wall_time_sec),
            "uncertainty": float(uncertainty.uncertainty),
            "td_error": float(uncertainty.td_error),
            "env_reward": reward_value,
            "hl_reward": float(hl_reward),
            "done": done,
            "success": bool(success),
        }
        self.step_metrics.append(row)
        if self.should_save_transitions:
            self._record_transition(
                current_obs=current_obs,
                next_obs=next_obs,
                action=action,
                reward=reward_value,
                done=done,
                row=row,
            )
        return row

    def run_episode(
        self,
        episode_id: int,
        deterministic: bool = False,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        """Run one episode or debug-length rollout and return episode metrics."""

        self.reset(seed=seed)
        rows = []
        for t in range(self.max_episode_steps):
            row = self.step(
                episode_id=episode_id,
                t=t,
                deterministic=deterministic,
            )
            rows.append(row)
            if row["done"]:
                break

        episode_length = len(rows)
        total_nfe = int(sum(row["nfe"] for row in rows))
        total_inference_time = float(
            sum(row["inference_wall_time_sec"] for row in rows)
        )
        total_controller_time = float(
            sum(row["controller_wall_time_sec"] for row in rows)
        )
        episode_return = float(sum(row["env_reward"] for row in rows))
        high_level_return = float(sum(row["hl_reward"] for row in rows))
        success = bool(
            any(row["success"] for row in rows)
            or max((row["env_reward"] for row in rows), default=0.0)
            >= self.success_reward_threshold
        )
        replans = int(sum(1 for row in rows if row["replanned"]))
        forced_replans = int(sum(1 for row in rows if row["forced_replan"]))

        episode_row = {
            "episode_id": int(episode_id),
            "env_name": self.env_name,
            "method": self.method,
            "seed": self.seed,
            "episode_return": episode_return,
            "high_level_return": high_level_return,
            "episode_length": episode_length,
            "success": success,
            "total_nfe": total_nfe,
            "mean_nfe_per_action": total_nfe / episode_length
            if episode_length
            else 0.0,
            "replans": replans,
            "forced_replans": forced_replans,
            "replan_rate": replans / episode_length if episode_length else 0.0,
            "mean_inference_wall_time_per_action": total_inference_time
            / episode_length
            if episode_length
            else 0.0,
            "mean_controller_wall_time_per_action": total_controller_time
            / episode_length
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
        self.episode_metrics.append(episode_row)
        return episode_row

    def save_metrics(self, output_dir: str | Path) -> dict[str, Any]:
        """Write step metrics, episode metrics, and aggregate summary."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self._write_csv(output_path / "metrics_step.csv", self.step_metrics)
        self._write_csv(output_path / "metrics_episode.csv", self.episode_metrics)
        summary = self.build_eval_summary()
        with (output_path / "eval_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if self.should_save_transitions:
            self._write_transition_npz(output_path / "transitions.npz")
        return summary

    def build_eval_summary(self) -> dict[str, Any]:
        """Aggregate episode metrics into a compact summary."""

        episodes = self.episode_metrics
        total_steps = int(sum(row["episode_length"] for row in episodes))
        total_nfe = int(sum(row["total_nfe"] for row in episodes))
        total_replans = int(sum(row["replans"] for row in episodes))
        total_forced = int(sum(row["forced_replans"] for row in episodes))
        total_discarded = int(sum(row["discarded_actions"] for row in episodes))
        total_time = float(
            sum(
                row["mean_inference_wall_time_per_action"] * row["episode_length"]
                for row in episodes
            )
        )
        total_controller_time = float(
            sum(
                row["mean_controller_wall_time_per_action"] * row["episode_length"]
                for row in episodes
            )
        )
        replan_rows = [row for row in self.step_metrics if row["replanned"]]
        learned_replans = int(
            sum(
                1
                for row in self.step_metrics
                if row["replanned"] and not row["forced_replan"]
            )
        )

        return {
            "env": self.env_name,
            "method": self.method,
            "seed": self.seed,
            "episodes": len(episodes),
            "success_rate": float(np.mean([row["success"] for row in episodes]))
            if episodes
            else 0.0,
            "mean_episode_return": float(
                np.mean([row["episode_return"] for row in episodes])
            )
            if episodes
            else 0.0,
            "median_episode_return": float(
                np.median([row["episode_return"] for row in episodes])
            )
            if episodes
            else 0.0,
            "mean_high_level_return": float(
                np.mean([row["high_level_return"] for row in episodes])
            )
            if episodes
            else 0.0,
            "mean_episode_length": float(
                np.mean([row["episode_length"] for row in episodes])
            )
            if episodes
            else 0.0,
            "mean_nfe_per_action": total_nfe / total_steps if total_steps else 0.0,
            "mean_inference_wall_time_per_action": total_time / total_steps
            if total_steps
            else 0.0,
            "mean_controller_wall_time_per_action": total_controller_time / total_steps
            if total_steps
            else 0.0,
            "mean_total_wall_time_per_action": (
                total_time + total_controller_time
            )
            / total_steps
            if total_steps
            else 0.0,
            "replan_rate": total_replans / total_steps if total_steps else 0.0,
            "forced_replan_rate": total_forced / total_steps if total_steps else 0.0,
            "learned_replan_rate": learned_replans / total_steps if total_steps else 0.0,
            "mean_plan_age_at_replan": float(
                np.mean([row["plan_age_before"] for row in replan_rows])
            )
            if replan_rows
            else 0.0,
            "mean_discarded_actions_per_episode": total_discarded / len(episodes)
            if episodes
            else 0.0,
            "mean_buffer_remaining_when_replanned": float(
                np.mean([row["buffer_remaining_before"] for row in replan_rows])
            )
            if replan_rows
            else 0.0,
        }

    def _format_env_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.diffusion_policy.action_dim,):
            raise ValueError(
                f"action shape {action.shape} does not match "
                f"({self.diffusion_policy.action_dim},)"
            )

        single_shape = getattr(self.env, "single_action_space", None).shape
        if len(single_shape) == 2 and single_shape[0] == 1:
            return action.reshape(1, 1, self.diffusion_policy.action_dim)
        if len(single_shape) == 1:
            return action.reshape(1, self.diffusion_policy.action_dim)
        raise ValueError(f"unsupported single_action_space shape {single_shape}")

    def _single_obs(self, obs: Any) -> dict[str, np.ndarray]:
        if not isinstance(obs, dict):
            raise TypeError("executor currently supports dict observations only")
        return {key: np.asarray(value)[0] for key, value in obs.items()}

    def _obs_norm(self, obs: Mapping[str, np.ndarray]) -> float:
        arrays = []
        for value in obs.values():
            try:
                array = np.asarray(value, dtype=np.float32)
            except (TypeError, ValueError):
                continue
            if array.size:
                arrays.append(array.reshape(-1))
        if not arrays:
            return 0.0
        return float(np.linalg.norm(np.concatenate(arrays)))

    def _high_level_reward(
        self,
        reward: float,
        replanned: bool,
        buffer_remaining_before: int,
        nfe: int,
        uncertainty: float = 0.0,
    ) -> float:
        replan_cost = self.replan_cost_c0 + self.replan_cost_c1 * float(nfe)
        inference_penalty = self.lambda_inference_cost * replan_cost if replanned else 0.0
        discard_penalty = (
            self.lambda_discard if replanned and buffer_remaining_before > 0 else 0.0
        )
        uncertainty_bonus = (
            self.uncertainty_replan_bonus * max(0.0, float(uncertainty))
            if replanned
            else 0.0
        )
        return float(reward) + uncertainty_bonus - inference_penalty - discard_penalty

    def _extract_success(self, info: Any, reward: float) -> bool:
        if isinstance(info, Mapping):
            for key in ("success", "is_success"):
                if key in info:
                    parsed = self._success_from_value(info[key])
                    if parsed is not None:
                        return parsed
        return reward >= self.success_reward_threshold

    def _success_from_value(self, value: Any) -> Optional[bool]:
        if isinstance(value, Mapping):
            for key in ("task", "success", "is_success"):
                if key in value:
                    return self._success_from_value(value[key])
            return None

        array = np.asarray(value)
        if array.size == 0:
            return None
        if array.dtype == object:
            return self._success_from_value(array.reshape(-1)[-1])
        return bool(array.reshape(-1)[-1])


    def _record_transition(
        self,
        current_obs: Mapping[str, np.ndarray],
        next_obs: Mapping[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        done: bool,
        row: Mapping[str, Any],
    ) -> None:
        self.transition_records.append(
            {
                "obs": self._state_vector(current_obs),
                "next_obs": self._state_vector(next_obs),
                "action": np.asarray(action, dtype=np.float32).reshape(-1).copy(),
                "reward": float(reward),
                "done": bool(done),
                "episode_id": int(row["episode_id"]),
                "t": int(row["t"]),
                "success": bool(row["success"]),
                "replanned": bool(row["replanned"]),
                "forced_replan": bool(row["forced_replan"]),
                "nfe": int(row["nfe"]),
            }
        )

    def _state_vector(self, obs: Mapping[str, np.ndarray]) -> np.ndarray:
        if "state" not in obs:
            raise KeyError("transition recording requires a 'state' observation")
        return np.asarray(obs["state"], dtype=np.float32).reshape(-1).copy()

    def _write_transition_npz(self, path: Path) -> None:
        records = self.transition_records
        if records:
            obs = np.stack([record["obs"] for record in records]).astype(np.float32)
            next_obs = np.stack([record["next_obs"] for record in records]).astype(np.float32)
            actions = np.stack([record["action"] for record in records]).astype(np.float32)
        else:
            obs = np.zeros((0, 0), dtype=np.float32)
            next_obs = np.zeros((0, 0), dtype=np.float32)
            actions = np.zeros((0, self.diffusion_policy.action_dim), dtype=np.float32)

        np.savez_compressed(
            path,
            obs=obs,
            next_obs=next_obs,
            actions=actions,
            rewards=np.asarray([record["reward"] for record in records], dtype=np.float32),
            dones=np.asarray([record["done"] for record in records], dtype=bool),
            episode_ids=np.asarray(
                [record["episode_id"] for record in records], dtype=np.int64
            ),
            timesteps=np.asarray([record["t"] for record in records], dtype=np.int64),
            successes=np.asarray([record["success"] for record in records], dtype=bool),
            replanned=np.asarray([record["replanned"] for record in records], dtype=bool),
            forced_replan=np.asarray(
                [record["forced_replan"] for record in records], dtype=bool
            ),
            nfe=np.asarray([record["nfe"] for record in records], dtype=np.int64),
            env_name=np.asarray(self.env_name),
            method=np.asarray(self.method),
            seed=np.asarray(self.seed, dtype=np.int64),
        )

    def _write_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
