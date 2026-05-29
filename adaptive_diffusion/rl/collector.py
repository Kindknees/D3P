"""Rollout collection helpers for replan-only PPO."""

from __future__ import annotations

from typing import Any, Mapping

from adaptive_diffusion.rl.rollout_buffer import ReplanRolloutBuffer


_REQUIRED_DECISION_FIELDS = (
    "obs_features",
    "action_mask",
    "action",
    "logprob",
    "value",
)

_REQUIRED_ROW_FIELDS = (
    "controller_action",
    "replanned",
    "hl_reward",
    "env_reward",
    "done",
    "denoise_steps",
    "nfe",
    "discarded_actions",
)


def record_ppo_transition(
    rollout_buffer: ReplanRolloutBuffer,
    decision_info: Any,
    step_row: Mapping[str, Any],
) -> None:
    """Append one executor step and PPO decision to a rollout buffer."""

    if decision_info is None:
        raise ValueError("PPO decision info is required before recording a step")

    missing_decision_fields = [
        field for field in _REQUIRED_DECISION_FIELDS if not hasattr(decision_info, field)
    ]
    if missing_decision_fields:
        raise AttributeError(
            "decision_info is missing fields: "
            + ", ".join(sorted(missing_decision_fields))
        )

    missing_row_fields = [
        field for field in _REQUIRED_ROW_FIELDS if field not in step_row
    ]
    if missing_row_fields:
        raise KeyError(
            "step_row is missing fields: " + ", ".join(sorted(missing_row_fields))
        )

    action = int(decision_info.action)
    row_action = int(step_row["controller_action"])
    if action != row_action:
        raise ValueError(
            f"decision action {action} does not match step row action {row_action}"
        )

    replanned = bool(step_row["replanned"])
    if replanned != bool(action == 1):
        raise ValueError("step row replan flag does not match decision action")

    rollout_buffer.add(
        obs_features=decision_info.obs_features,
        action=action,
        logprob=float(decision_info.logprob),
        value=float(decision_info.value),
        reward_hl=float(step_row["hl_reward"]),
        reward_env=float(step_row["env_reward"]),
        done=bool(step_row["done"]),
        action_mask=decision_info.action_mask,
        replanned=replanned,
        denoise_steps=int(step_row["denoise_steps"]),
        nfe=int(step_row["nfe"]),
        discarded_actions=int(step_row["discarded_actions"]),
    )


class PPOExecutorRolloutCollector:
    """Runs an executor and records every PPO decision into a rollout buffer."""

    def __init__(
        self,
        executor: Any,
        rollout_buffer: ReplanRolloutBuffer,
    ) -> None:
        self.executor = executor
        self.rollout_buffer = rollout_buffer

    def record_step(self, step_row: Mapping[str, Any]) -> None:
        decision_info = getattr(self.executor.controller, "last_decision_info", None)
        record_ppo_transition(self.rollout_buffer, decision_info, step_row)

    def run_episode(
        self,
        episode_id: int,
        deterministic: bool = False,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run one executor episode and record each high-level PPO transition."""

        self.executor.reset(seed=seed)
        rows = []
        for t in range(self.executor.max_episode_steps):
            row = self.executor.step(
                episode_id=episode_id,
                t=t,
                deterministic=deterministic,
            )
            self.record_step(row)
            rows.append(row)
            if row["done"]:
                break
        return rows
