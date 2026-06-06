"""Single-environment adaptive action-buffer executor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from adaptive_diffusion.buffers import ActionBuffer, first_action_jump, mean_overlap_jump
from adaptive_diffusion.controllers.three_way import (
    CONTINUE,
    REPAIR,
    RESET,
    AdaptiveCommitmentController,
    ControllerDecision,
    ExecutionState,
    ThreeWayHeuristicController,
)
from adaptive_diffusion.repair import RepairConfig, RepairResult
from utils.execution_logger import ExecutionLogger


class EnvProtocol(Protocol):
    def reset(self, seed: int | None = None):
        ...

    def step(self, action: np.ndarray):
        ...


class PolicyProtocol(Protocol):
    nfe_per_sample: int

    def sample_action_chunk(self, obs) -> np.ndarray:
        ...


RepairFn = Callable[[object, np.ndarray, RepairConfig], RepairResult]
UncertaintyResult = float | tuple[float | None, int] | None
UncertaintyFn = Callable[[object, dict, int], UncertaintyResult]


@dataclass(frozen=True)
class ActionNoiseConfig:
    enabled: bool = False
    noise_type: str = "impulse"
    sigma: float = 0.0
    probability: float = 0.0
    clip_min: float | None = None
    clip_max: float | None = None


@dataclass(frozen=True)
class ExecutorConfig:
    method: str
    task: str
    seed: int
    action_dim: int
    chunk_size: int = 4
    max_steps: int = 300
    success_threshold: float = 1.0
    nfe_per_repair: int = 10
    repair_config: RepairConfig | None = None
    repair_reject_fallback: str = "reset"
    action_noise_config: ActionNoiseConfig | None = None


@dataclass
class EpisodeStats:
    success: bool
    episode_return: float
    steps: int
    num_repairs: int
    num_repairs_rejected: int
    num_repair_candidates: int
    num_resets: int
    num_random_resets: int
    num_buffer_empty_resets: int
    num_discard_nonempty_buffer: int
    total_nfe: int
    mean_action_jump: float
    mean_buffer_jump: float
    max_action_jump: float
    max_buffer_jump: float


class AdaptiveActionExecutor:
    """Execute a policy with fixed, adaptive-commitment, or repair control."""

    def __init__(
        self,
        env: EnvProtocol,
        policy: PolicyProtocol,
        config: ExecutorConfig,
        controller: AdaptiveCommitmentController | ThreeWayHeuristicController | None,
        logger: ExecutionLogger,
        uncertainty_fn: UncertaintyFn | None = None,
        repair_fn: RepairFn | None = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self.config = config
        self.controller = controller
        self.logger = logger
        self.uncertainty_fn = uncertainty_fn
        self.repair_fn = repair_fn

    def run_episode(self, episode_id: int = 0) -> EpisodeStats:
        obs = self.env.reset(seed=self.config.seed + episode_id)
        buffer = ActionBuffer.empty(self.config.action_dim)
        plan_age = 0
        committed_steps_remaining = 0
        episode_return = 0.0
        total_nfe = 0
        num_repairs = 0
        num_repairs_rejected = 0
        num_repair_candidates = 0
        num_resets = 0
        num_random_resets = 0
        num_buffer_empty_resets = 0
        num_discard_nonempty_buffer = 0
        action_jumps: list[float] = []
        buffer_jumps: list[float] = []
        td_error: float | None = None
        done = False
        action_noise_rng = np.random.default_rng(
            self.config.seed * 1_000_003 + episode_id
        )

        for step in range(self.config.max_steps):
            phase = plan_age % self.config.chunk_size
            decision = self._decide(buffer, plan_age, committed_steps_remaining, td_error)
            event_nfe = 0
            action_jump = 0.0
            buffer_jump = 0.0
            accepted = True
            fallback_reason = ""
            step_triggered_replan = False
            step_repair_candidate_generated = False
            step_repair_accepted = False
            step_repair_rejected = False
            step_random_reset = False
            step_discarded_nonempty_buffer = False
            step_action_jump = 0.0
            step_buffer_jump = 0.0

            if decision.action == RESET:
                old_nonempty = not buffer.is_empty()
                old_actions = buffer.actions.copy() if old_nonempty else None
                if old_nonempty:
                    num_discard_nonempty_buffer += 1
                    step_discarded_nonempty_buffer = True
                chunk, sample_nfe = self._sample_action_chunk(obs, decision.sample_nfe)
                new_actions = chunk[: self.config.chunk_size]
                if old_actions is not None:
                    action_jump = first_action_jump(old_actions, new_actions)
                    buffer_jump = mean_overlap_jump(old_actions, new_actions)
                    action_jumps.append(action_jump)
                    buffer_jumps.append(buffer_jump)
                    step_action_jump = action_jump
                    step_buffer_jump = buffer_jump
                buffer = ActionBuffer(new_actions)
                plan_age = 0
                committed_steps_remaining = (
                    decision.commitment_horizon or self.config.chunk_size
                )
                event_nfe = sample_nfe
                total_nfe += event_nfe
                num_resets += 1
                if decision.reason == "buffer_empty":
                    num_buffer_empty_resets += 1
                else:
                    num_random_resets += 1
                    step_random_reset = True
                    step_triggered_replan = True
                self._log_event(
                    episode_id,
                    step,
                    decision,
                    phase,
                    plan_age,
                    buffer,
                    event_nfe,
                    old_nonempty=old_nonempty,
                    action_jump=action_jump,
                    buffer_jump=buffer_jump,
                    accepted=accepted,
                    fallback_reason=fallback_reason,
                    td_error=td_error,
                )

            elif decision.action == REPAIR:
                step_repair_candidate_generated = True
                num_repair_candidates += 1
                if self.repair_fn is None or self.config.repair_config is None:
                    decision = ControllerDecision(RESET, "repair_unavailable", self.config.chunk_size)
                    chunk, sample_nfe = self._sample_action_chunk(obs, decision.sample_nfe)
                    buffer = ActionBuffer(chunk[: self.config.chunk_size])
                    plan_age = 0
                    committed_steps_remaining = self.config.chunk_size
                    event_nfe = sample_nfe
                    total_nfe += event_nfe
                    num_resets += 1
                    num_random_resets += 1
                    step_random_reset = True
                    step_triggered_replan = True
                    fallback_reason = "repair_unavailable"
                else:
                    residual = buffer.actions.copy()
                    repair = self.repair_fn(obs, residual, self.config.repair_config)
                    action_jump = repair.action_jump
                    buffer_jump = repair.buffer_jump
                    accepted = repair.accepted
                    fallback_reason = repair.fallback_reason
                    event_nfe = self.config.nfe_per_repair
                    total_nfe += event_nfe
                    if repair.accepted:
                        replacement_len = buffer.remaining_length
                        buffer.replace(repair.repaired_actions[:replacement_len])
                        num_repairs += 1
                        step_repair_accepted = True
                        step_action_jump = action_jump
                        step_buffer_jump = buffer_jump
                        action_jumps.append(action_jump)
                        buffer_jumps.append(buffer_jump)
                    else:
                        num_repairs_rejected += 1
                        step_repair_rejected = True
                        if self.config.repair_reject_fallback == "continue":
                            pass
                        elif self.config.repair_reject_fallback == "reset":
                            old_nonempty = not buffer.is_empty()
                            old_actions = buffer.actions.copy() if old_nonempty else None
                            chunk, sample_nfe = self._sample_action_chunk(obs, decision.sample_nfe)
                            new_actions = chunk[: self.config.chunk_size]
                            if old_actions is not None:
                                reset_action_jump = first_action_jump(old_actions, new_actions)
                                reset_buffer_jump = mean_overlap_jump(old_actions, new_actions)
                                action_jumps.append(reset_action_jump)
                                buffer_jumps.append(reset_buffer_jump)
                                step_action_jump = max(step_action_jump, reset_action_jump)
                                step_buffer_jump = max(step_buffer_jump, reset_buffer_jump)
                                num_discard_nonempty_buffer += 1
                                step_discarded_nonempty_buffer = True
                            buffer = ActionBuffer(new_actions)
                            plan_age = 0
                            committed_steps_remaining = self.config.chunk_size
                            event_nfe += sample_nfe
                            total_nfe += sample_nfe
                            num_resets += 1
                            num_random_resets += 1
                            step_random_reset = True
                            step_triggered_replan = True
                        else:
                            raise ValueError(
                                f"Unsupported repair_reject_fallback: {self.config.repair_reject_fallback}"
                            )
                self._log_event(
                    episode_id,
                    step,
                    decision,
                    phase,
                    plan_age,
                    buffer,
                    event_nfe,
                    old_nonempty=True,
                    action_jump=action_jump,
                    buffer_jump=buffer_jump,
                    accepted=accepted,
                    fallback_reason=fallback_reason,
                    td_error=td_error,
                )

            if buffer.is_empty():
                raise RuntimeError("Controller left the action buffer empty before execution.")

            action = buffer.consume_one()
            executed_action, action_noise_applied, action_noise_l2 = (
                self._apply_action_noise(action, action_noise_rng)
            )
            obs, reward, done, info = self.env.step(executed_action)
            episode_return += float(reward)
            committed_steps_remaining = max(0, committed_steps_remaining - 1)
            plan_age += 1
            if self.uncertainty_fn is not None:
                td_error, uncertainty_nfe = _parse_uncertainty_result(
                    self.uncertainty_fn(obs, info, step)
                )
                total_nfe += uncertainty_nfe
                event_nfe += uncertainty_nfe

            self.logger.log_step(
                task=self.config.task,
                seed=self.config.seed,
                episode_id=episode_id,
                timestep=step,
                method=self.config.method,
                phase=phase,
                plan_age=plan_age,
                remaining_buffer_length=buffer.remaining_length,
                commitment_horizon=decision.commitment_horizon or "",
                committed_steps_remaining=committed_steps_remaining,
                td_error="" if td_error is None else td_error,
                policy_disagreement="" if td_error is None else td_error,
                likelihood_score="",
                criticality_score="",
                triggered_replan=step_triggered_replan,
                repair_candidate_generated=step_repair_candidate_generated,
                repair_accepted=step_repair_accepted,
                repair_rejected=step_repair_rejected,
                random_reset=step_random_reset,
                discarded_nonempty_buffer=step_discarded_nonempty_buffer,
                action_jump=step_action_jump,
                buffer_jump=step_buffer_jump,
                executed_action_l2=np.linalg.norm(executed_action),
                action_noise_applied=action_noise_applied,
                action_noise_l2=action_noise_l2,
                reward=reward,
                done=done,
                success=episode_return >= self.config.success_threshold,
                nfe_this_step=event_nfe,
                wall_time_ms="",
            )
            if done:
                break

        success = episode_return >= self.config.success_threshold
        stats = EpisodeStats(
            success=success,
            episode_return=episode_return,
            steps=step + 1,
            num_repairs=num_repairs,
            num_repairs_rejected=num_repairs_rejected,
            num_repair_candidates=num_repair_candidates,
            num_resets=num_resets,
            num_random_resets=num_random_resets,
            num_buffer_empty_resets=num_buffer_empty_resets,
            num_discard_nonempty_buffer=num_discard_nonempty_buffer,
            total_nfe=total_nfe,
            mean_action_jump=_mean_or_zero(action_jumps),
            mean_buffer_jump=_mean_or_zero(buffer_jumps),
            max_action_jump=_max_or_zero(action_jumps),
            max_buffer_jump=_max_or_zero(buffer_jumps),
        )
        self._log_episode_summary(episode_id, stats)
        return stats

    def _sample_action_chunk(self, obs, sample_nfe: int | None) -> tuple[np.ndarray, int]:
        if sample_nfe is not None and hasattr(self.policy, "sample_action_chunk_nfe"):
            return self.policy.sample_action_chunk_nfe(obs, sample_nfe), int(sample_nfe)
        return self.policy.sample_action_chunk(obs), int(self.policy.nfe_per_sample)

    def _apply_action_noise(
        self, action: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, bool, float]:
        config = self.config.action_noise_config
        executed = np.asarray(action).copy()
        if config is None or not config.enabled:
            return executed, False, 0.0
        if config.noise_type != "impulse":
            raise ValueError(f"Unsupported action noise type: {config.noise_type}")
        if config.probability <= 0 or config.sigma <= 0:
            return executed, False, 0.0
        if rng.random() >= config.probability:
            return executed, False, 0.0

        noise = rng.normal(0.0, config.sigma, size=executed.shape).astype(
            executed.dtype, copy=False
        )
        executed = executed + noise
        if config.clip_min is not None or config.clip_max is not None:
            clip_min = -np.inf if config.clip_min is None else config.clip_min
            clip_max = np.inf if config.clip_max is None else config.clip_max
            executed = np.clip(executed, clip_min, clip_max)
        return executed, True, float(np.linalg.norm(executed - action))

    def _decide(
        self,
        buffer: ActionBuffer,
        plan_age: int,
        committed_steps_remaining: int,
        td_error: float | None,
    ) -> ControllerDecision:
        if self.config.method == "fixed_chunk":
            if buffer.is_empty():
                return ControllerDecision(RESET, "buffer_empty", self.config.chunk_size)
            return ControllerDecision(CONTINUE, "within_fixed_chunk")
        if self.controller is None:
            raise ValueError(f"Method {self.config.method} requires a controller.")
        return self.controller.decide(
            ExecutionState(
                remaining_buffer_length=buffer.remaining_length,
                plan_age=plan_age,
                committed_steps_remaining=committed_steps_remaining,
                td_error=td_error,
            )
        )

    def _log_event(
        self,
        episode_id: int,
        step: int,
        decision: ControllerDecision,
        phase: int,
        plan_age: int,
        buffer: ActionBuffer,
        nfe_event: int,
        old_nonempty: bool,
        action_jump: float,
        buffer_jump: float,
        accepted: bool,
        fallback_reason: str,
        td_error: float | None,
    ) -> None:
        first_l2 = np.linalg.norm(buffer.first()) if not buffer.is_empty() else ""
        self.logger.log_event(
            task=self.config.task,
            seed=self.config.seed,
            episode_id=episode_id,
            timestep=step,
            method=self.config.method,
            event_type=decision.action if decision.action != RESET else "reset",
            phase=phase,
            plan_age=plan_age,
            remaining_buffer_length=buffer.remaining_length,
            td_error="" if td_error is None else td_error,
            likelihood_score="",
            criticality_score="",
            old_first_action_l2="" if not old_nonempty else first_l2,
            new_first_action_l2=first_l2,
            action_jump=action_jump,
            buffer_jump=buffer_jump,
            repair_noise_ratio=(
                "" if self.config.repair_config is None else self.config.repair_config.noise_ratio
            ),
            repair_anchor_rho=(
                "" if self.config.repair_config is None else self.config.repair_config.anchor_rho
            ),
            num_candidates=1,
            selected_candidate_score="",
            nfe_event=nfe_event,
            accepted=accepted,
            fallback_reason=fallback_reason,
        )

    def _log_episode_summary(self, episode_id: int, stats: EpisodeStats) -> None:
        self.logger.log_episode_summary(
            task=self.config.task,
            seed=self.config.seed,
            episode_id=episode_id,
            method=self.config.method,
            success=stats.success,
            episode_return=stats.episode_return,
            episode_length=stats.steps,
            num_replans=stats.num_resets,
            num_repairs=stats.num_repairs,
            num_repairs_accepted=stats.num_repairs,
            num_repairs_rejected=stats.num_repairs_rejected,
            num_repair_candidates=stats.num_repair_candidates,
            num_resets=stats.num_resets,
            num_random_resets=stats.num_random_resets,
            num_buffer_empty_resets=stats.num_buffer_empty_resets,
            num_discard_nonempty_buffer=stats.num_discard_nonempty_buffer,
            total_nfe=stats.total_nfe,
            nfe_per_action=stats.total_nfe / max(1, stats.steps),
            mean_action_jump=stats.mean_action_jump,
            mean_buffer_jump=stats.mean_buffer_jump,
            max_action_jump=stats.max_action_jump,
            max_buffer_jump=stats.max_buffer_jump,
            wall_time_total_ms="",
        )


def run_executor_episodes(
    env_factory: Callable[[], EnvProtocol],
    policy_factory: Callable[[], PolicyProtocol],
    config: ExecutorConfig,
    output_dir: str | Path,
    controller: AdaptiveCommitmentController | ThreeWayHeuristicController | None = None,
    uncertainty_fn: UncertaintyFn | None = None,
    repair_fn: RepairFn | None = None,
    num_episodes: int = 1,
) -> list[EpisodeStats]:
    with ExecutionLogger(output_dir, config_resolved={"method": config.method}) as logger:
        stats = []
        for episode_id in range(num_episodes):
            executor = AdaptiveActionExecutor(
                env=env_factory(),
                policy=policy_factory(),
                config=config,
                controller=controller,
                logger=logger,
                uncertainty_fn=uncertainty_fn,
                repair_fn=repair_fn,
            )
            stats.append(executor.run_episode(episode_id=episode_id))
        return stats


def _parse_uncertainty_result(result: UncertaintyResult) -> tuple[float | None, int]:
    if isinstance(result, tuple):
        value, nfe = result
        return value, int(nfe)
    return result, 0


def _mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _max_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.max(values))
