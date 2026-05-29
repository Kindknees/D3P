"""RL utilities for adaptive diffusion controllers."""

from adaptive_diffusion.rl.actor_critic import ReplanActorCritic
from adaptive_diffusion.rl.advantage import compute_gae
from adaptive_diffusion.rl.collector import (
    PPOExecutorRolloutCollector,
    record_ppo_transition,
)
from adaptive_diffusion.rl.features import (
    ReplanFeatureConfig,
    build_replan_features,
    replan_action_mask,
)
from adaptive_diffusion.rl.masked_categorical import MaskedCategorical
from adaptive_diffusion.rl.ppo import (
    PPOBCRegularizerBatch,
    PPOUpdateConfig,
    PPOUpdateStats,
    load_replan_actor_critic_checkpoint,
    ppo_config_to_dict,
    save_replan_actor_critic_checkpoint,
    update_replan_ppo,
)
from adaptive_diffusion.rl.rollout_buffer import ReplanRolloutBuffer, RolloutBatch

__all__ = [
    "MaskedCategorical",
    "PPOExecutorRolloutCollector",
    "PPOBCRegularizerBatch",
    "PPOUpdateConfig",
    "PPOUpdateStats",
    "ReplanActorCritic",
    "ReplanFeatureConfig",
    "ReplanRolloutBuffer",
    "RolloutBatch",
    "build_replan_features",
    "compute_gae",
    "load_replan_actor_critic_checkpoint",
    "ppo_config_to_dict",
    "record_ppo_transition",
    "save_replan_actor_critic_checkpoint",
    "replan_action_mask",
    "update_replan_ppo",
]
