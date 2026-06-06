"""Robomimic adaptive commitment eval agent."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from adaptive_diffusion.controllers.three_way import (
    AdaptiveCommitmentController,
    BinaryResetController,
    BenefitTableController,
    BoundaryOnlyResetController,
    CriticalityAwareThreeWayController,
    PhaseAwareHysteresisResetController,
    RepairOnlyController,
    ThreeWayControllerConfig,
    ThreeWayHeuristicController,
)
from adaptive_diffusion.diffusion_policy_adapter import (
    DiffusionPolicyAdapter,
    DiffusionResidualRepairer,
)
from adaptive_diffusion.executor import (
    ActionNoiseConfig,
    AdaptiveActionExecutor,
    ExecutorConfig,
)
from adaptive_diffusion.repair import RepairConfig
from adaptive_diffusion.vector_env_adapter import SingleVectorEnvAdapter
from agent.eval.eval_agent import EvalAgent
from utils.execution_logger import ExecutionLogger

log = logging.getLogger(__name__)


class EvalAdaptiveCommitmentAgent(EvalAgent):
    """Evaluate adaptive action-buffer execution with a pretrained diffusion policy."""

    def __init__(self, cfg):
        if cfg.env.n_envs != 1:
            raise ValueError("EvalAdaptiveCommitmentAgent requires env.n_envs=1.")
        if cfg.act_steps != 1:
            raise ValueError(
                "EvalAdaptiveCommitmentAgent requires act_steps=1 so the "
                "executor can consume one buffered action per env step."
            )
        super().__init__(cfg)

    def run(self):
        method_cfg = self.cfg.adaptive_commitment
        method = method_cfg.method
        output_dir = method_cfg.get(
            "output_dir",
            os.path.join(self.logdir, "adaptive_commitment_logs"),
        )
        logger = ExecutionLogger(
            output_dir=output_dir,
            config_resolved=OmegaConf.to_yaml(self.cfg, resolve=True),
        )
        policy = DiffusionPolicyAdapter(
            model=self.model,
            device=self.device,
            deterministic=True,
        )
        env = SingleVectorEnvAdapter(self.venv)
        controller = self._make_controller(method_cfg)
        repair_config = self._make_repair_config(method_cfg)
        repairer = DiffusionResidualRepairer(policy) if repair_config is not None else None
        executor_config = ExecutorConfig(
            method=method,
            task=self.env_name,
            seed=self.seed,
            action_dim=self.action_dim,
            chunk_size=self.horizon_steps,
            max_steps=self.max_episode_steps,
            success_threshold=self.best_reward_threshold_for_success,
            nfe_per_repair=method_cfg.get("repair", {}).get("nfe", 10),
            repair_config=repair_config,
            repair_reject_fallback=method_cfg.get("repair", {}).get(
                "reject_fallback", "reset"
            ),
            action_noise_config=self._make_action_noise_config(method_cfg),
        )
        stats = []
        num_episodes = method_cfg.get("episodes", 1)
        for episode_id in range(num_episodes):
            executor = AdaptiveActionExecutor(
                env=env,
                policy=policy,
                config=executor_config,
                controller=controller,
                logger=logger,
                uncertainty_fn=self._make_uncertainty_fn(method_cfg, policy),
                repair_fn=repairer,
            )
            stats.append(executor.run_episode(episode_id=episode_id))
        logger.close()
        self._write_result_npz(stats)
        log.info("adaptive eval finished: %d episodes", len(stats))

    def _make_controller(self, method_cfg):
        method = method_cfg.method
        commitment = method_cfg.get("commitment", {})
        high = float(commitment.get("high_td_threshold", 0.09))
        medium = float(commitment.get("medium_td_threshold", 0.05))
        horizons = tuple(commitment.get("horizons", [1, 2, self.horizon_steps]))
        if method == "fixed_chunk":
            return None
        if method == "td_replan_anywhere_random_reset":
            reset_cfg = method_cfg.get("reset", {})
            threshold = float(reset_cfg.get("trigger_td_threshold", high))
            return BinaryResetController(threshold, chunk_size=self.horizon_steps)
        if method == "td_replan_boundary_only":
            return BoundaryOnlyResetController(high, medium, horizons=horizons)
        if method == "phase_aware_hysteresis":
            phase_cfg = method_cfg.get("phase_hysteresis", {})
            thresholds = tuple(
                float(value) for value in phase_cfg.get(
                    "phase_thresholds", [high, high, high, high]
                )
            )
            min_consecutive = int(phase_cfg.get("min_consecutive", 2))
            return PhaseAwareHysteresisResetController(
                thresholds,
                min_consecutive=min_consecutive,
                chunk_size=self.horizon_steps,
            )
        if method == "adaptive_commitment_heuristic":
            return AdaptiveCommitmentController(high, medium, horizons=horizons)
        if method.startswith("td_repair_residual_"):
            repair = method_cfg.get("repair", {})
            repair_threshold = float(repair.get("trigger_td_threshold", high))
            return RepairOnlyController(repair_threshold, chunk_size=self.horizon_steps)
        if method == "adaptive_commitment_with_repair":
            repair = method_cfg.get("repair", {})
            repair_threshold = float(repair.get("trigger_td_threshold", high))
            return ThreeWayHeuristicController(
                ThreeWayControllerConfig(
                    repair_td_threshold=repair_threshold,
                    high_td_threshold=high,
                    medium_td_threshold=medium,
                    horizons=horizons,
                    catastrophic_reset_enabled=method_cfg.get("reset", {}).get(
                        "catastrophic_enabled", False
                    ),
                )
            )
        if method == "criticality_aware_commitment_with_repair":
            repair = method_cfg.get("repair", {})
            criticality = method_cfg.get("criticality", {})
            repair_threshold = float(repair.get("trigger_td_threshold", high))
            return CriticalityAwareThreeWayController(
                ThreeWayControllerConfig(
                    repair_td_threshold=repair_threshold,
                    high_td_threshold=high,
                    medium_td_threshold=medium,
                    horizons=horizons,
                    catastrophic_reset_enabled=method_cfg.get("reset", {}).get(
                        "catastrophic_enabled", False
                    ),
                ),
                low_nfe=int(criticality.get("low_nfe", 10)),
                high_nfe=int(criticality.get("high_nfe", self.horizon_steps * 5)),
                criticality_threshold=float(criticality.get("threshold", high)),
            )
        if method == "benefit_predictor_controller":
            benefit_cfg = method_cfg.get("benefit_predictor", {})
            return BenefitTableController(
                task=self.env_name,
                table=self._load_benefit_table(benefit_cfg),
                high_td_threshold=high,
                medium_td_threshold=medium,
                horizons=horizons,
                min_benefit=float(benefit_cfg.get("min_benefit", 5.0)),
                min_uncertainty=float(benefit_cfg.get("min_uncertainty", 0.0)),
                allow_reset=benefit_cfg.get("allow_reset", True),
                allow_repair=benefit_cfg.get("allow_repair", True),
                max_interventions=benefit_cfg.get("max_interventions", 1),
            )
        raise ValueError(f"Unsupported adaptive commitment method: {method}")

    def _load_benefit_table(self, benefit_cfg):
        path = Path(benefit_cfg.get("branch_logs_path", ""))
        if not path.exists():
            raise ValueError(f"Benefit predictor branch_logs_path does not exist: {path}")
        grouped = {}
        with path.open(newline="") as file_obj:
            for row in csv.DictReader(file_obj):
                key = (
                    row["task"],
                    int(float(row["phase"])),
                    int(float(row["remaining_buffer_length"])),
                )
                entry = grouped.setdefault(
                    key,
                    {"continue": [], "repair_050": [], "reset": []},
                )
                entry["continue"].append(float(row["G_continue"]))
                entry["repair_050"].append(float(row["G_repair_050"]))
                entry["reset"].append(float(row["G_reset"]))

        table = {}
        for key, values in grouped.items():
            means = {name: float(np.mean(scores)) for name, scores in values.items()}
            best = max(means, key=means.get)
            benefit = 0.0 if best == "continue" else means[best] - means["continue"]
            action = {"continue": "continue", "repair_050": "repair", "reset": "reset"}[best]
            table[key] = (action, benefit)
        return table

    def _make_repair_config(self, method_cfg):
        repair = method_cfg.get("repair", {})
        if not repair.get("enabled", False):
            return None
        return RepairConfig(
            target_len=self.horizon_steps,
            noise_ratio=float(repair.get("noise_ratio", 0.5)),
            anchor_rho=float(repair.get("anchor_rho", 0.5)),
            accept_min_action_jump=repair.get("accept_min_action_jump", None),
            accept_max_action_jump=repair.get("accept_max_action_jump", None),
            accept_min_buffer_jump=repair.get("accept_min_buffer_jump", None),
            accept_max_buffer_jump=repair.get("accept_max_buffer_jump", None),
        )

    def _make_action_noise_config(self, method_cfg):
        action_noise = method_cfg.get("action_noise", {})
        if not action_noise.get("enabled", False):
            return None
        return ActionNoiseConfig(
            enabled=True,
            noise_type=action_noise.get("type", "impulse"),
            sigma=float(action_noise.get("sigma", 0.0)),
            probability=float(action_noise.get("probability", 0.0)),
            clip_min=action_noise.get("clip_min", None),
            clip_max=action_noise.get("clip_max", None),
        )

    def _make_uncertainty_fn(self, method_cfg, policy):
        uncertainty = method_cfg.get("uncertainty", {})
        threshold_source = uncertainty.get("source", "none")
        if threshold_source == "none":
            return None
        if threshold_source == "info_td_error":
            def uncertainty_fn(obs, info, step):
                value = info.get("td_error", info.get("bad_td", None))
                if value is None:
                    return None
                return float(np.asarray(value).reshape(-1)[0])

            return uncertainty_fn
        if threshold_source == "policy_disagreement":
            num_samples = int(uncertainty.get("num_samples", 2))
            sample_interval = int(uncertainty.get("sample_interval", 1))
            if num_samples < 2:
                raise ValueError("policy_disagreement requires num_samples >= 2.")
            if sample_interval <= 0:
                raise ValueError("sample_interval must be positive.")

            def uncertainty_fn(obs, info, step):
                if step % sample_interval != 0:
                    return 0.0, 0
                value = policy.action_disagreement(obs, num_samples=num_samples)
                return value, policy.nfe_per_sample * num_samples

            return uncertainty_fn
        raise ValueError(f"Unsupported uncertainty source: {threshold_source}")

    def _write_result_npz(self, stats):
        success_rate = np.mean([stat.success for stat in stats]) if stats else 0.0
        episode_return = (
            np.mean([stat.episode_return for stat in stats]) if stats else 0.0
        )
        nfe_per_action = np.mean(
            [stat.total_nfe / max(1, stat.steps) for stat in stats]
        ) if stats else 0.0
        np.savez(
            self.result_path,
            num_episode=len(stats),
            eval_success_rate=success_rate,
            eval_episode_reward=episode_return,
            eval_best_reward=success_rate,
            nfe_per_action=nfe_per_action,
        )
