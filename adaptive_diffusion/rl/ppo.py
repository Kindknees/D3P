"""PPO update utilities for replan-only controllers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from adaptive_diffusion.rl.actor_critic import ReplanActorCritic
from adaptive_diffusion.rl.rollout_buffer import RolloutBatch


@dataclass(frozen=True)
class PPOUpdateConfig:
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    epochs: int = 4
    minibatch_size: int = 64
    max_grad_norm: float | None = 0.5
    normalize_advantages: bool = True
    shuffle: bool = True
    bc_regularizer_coef: float = 0.0
    bc_regularizer_minibatch_size: int | None = None


@dataclass(frozen=True)
class PPOBCRegularizerBatch:
    obs_features: np.ndarray
    action_masks: np.ndarray
    actions: np.ndarray
    sample_weights: np.ndarray | None = None

    @property
    def size(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class PPOUpdateStats:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    bc_regularizer_loss: float
    bc_regularizer_accuracy: float
    num_updates: int


def update_replan_ppo(
    model: ReplanActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    config: PPOUpdateConfig | None = None,
    device: str | torch.device = "cpu",
    seed: int | None = None,
    bc_regularizer_batch: PPOBCRegularizerBatch | None = None,
) -> PPOUpdateStats:
    """Run PPO clipped-policy updates on one high-level rollout batch."""

    config = config or PPOUpdateConfig()
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")
    if config.bc_regularizer_coef < 0.0:
        raise ValueError("bc_regularizer_coef must be non-negative")
    if (
        config.bc_regularizer_minibatch_size is not None
        and config.bc_regularizer_minibatch_size <= 0
    ):
        raise ValueError("bc_regularizer_minibatch_size must be positive")
    if config.bc_regularizer_coef > 0.0 and bc_regularizer_batch is None:
        raise ValueError("bc_regularizer_batch is required when bc_regularizer_coef > 0")

    batch_size = int(len(batch.actions))
    if batch_size == 0:
        raise ValueError("cannot update PPO with an empty batch")

    device = torch.device(device)
    model.to(device)
    model.train()

    features = torch.as_tensor(batch.obs_features, dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch.actions, dtype=torch.long, device=device)
    old_logprobs = torch.as_tensor(batch.logprobs, dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=device)
    advantages = torch.as_tensor(batch.advantages, dtype=torch.float32, device=device)
    action_masks = torch.as_tensor(batch.action_masks, dtype=torch.bool, device=device)
    bc_tensors = None
    if config.bc_regularizer_coef > 0.0:
        bc_tensors = _prepare_bc_regularizer_tensors(bc_regularizer_batch, device)

    if config.normalize_advantages and batch_size > 1:
        std = advantages.std(unbiased=False)
        if float(std.detach().cpu()) > 1e-8:
            advantages = (advantages - advantages.mean()) / (std + 1e-8)

    rng = np.random.default_rng(seed)
    stats: list[dict[str, float]] = []
    for _ in range(config.epochs):
        indices = np.arange(batch_size)
        if config.shuffle:
            rng.shuffle(indices)
        for start in range(0, batch_size, config.minibatch_size):
            mb_idx = torch.as_tensor(
                indices[start : start + config.minibatch_size],
                dtype=torch.long,
                device=device,
            )
            output = model.evaluate_actions(
                features[mb_idx],
                action_masks[mb_idx],
                actions[mb_idx],
            )
            new_logprobs = output["logprob"]
            values = output["value"]
            entropy = output["entropy"].mean()

            ratio = torch.exp(new_logprobs - old_logprobs[mb_idx])
            policy_unclipped = ratio * advantages[mb_idx]
            policy_clipped = torch.clamp(
                ratio,
                1.0 - config.clip_ratio,
                1.0 + config.clip_ratio,
            ) * advantages[mb_idx]
            policy_loss = -torch.min(policy_unclipped, policy_clipped).mean()
            value_loss = torch.nn.functional.mse_loss(values, returns[mb_idx])
            bc_loss = torch.zeros((), dtype=torch.float32, device=device)
            bc_accuracy = torch.zeros((), dtype=torch.float32, device=device)
            if bc_tensors is not None:
                bc_loss, bc_accuracy = _bc_regularizer_loss(
                    model=model,
                    tensors=bc_tensors,
                    config=config,
                    rng=rng,
                    device=device,
                )
            loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy
                + config.bc_regularizer_coef * bc_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                logratio = new_logprobs - old_logprobs[mb_idx]
                approx_kl = ((torch.exp(logratio) - 1.0) - logratio).mean()
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > config.clip_ratio).float().mean()
                )
            stats.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                    "approx_kl": float(approx_kl.detach().cpu()),
                    "clip_fraction": float(clip_fraction.detach().cpu()),
                    "bc_regularizer_loss": float(bc_loss.detach().cpu()),
                    "bc_regularizer_accuracy": float(bc_accuracy.detach().cpu()),
                }
            )

    model.eval()
    return _mean_stats(stats)


def _prepare_bc_regularizer_tensors(
    batch: PPOBCRegularizerBatch | None,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if batch is None:
        raise ValueError("bc_regularizer_batch is required")
    if batch.size <= 0:
        raise ValueError("bc_regularizer_batch cannot be empty")

    features_np = np.asarray(batch.obs_features, dtype=np.float32)
    masks_np = np.asarray(batch.action_masks, dtype=bool)
    actions_np = np.asarray(batch.actions, dtype=np.int64).reshape(-1)
    if features_np.ndim != 2:
        raise ValueError("bc_regularizer obs_features must be rank 2")
    if masks_np.shape != (batch.size, 2):
        raise ValueError("bc_regularizer action_masks must have shape [N, 2]")
    if features_np.shape[0] != batch.size:
        raise ValueError("bc_regularizer obs_features/actions size mismatch")
    if np.any((actions_np < 0) | (actions_np > 1)):
        raise ValueError("bc_regularizer actions must be 0 or 1")
    if not np.all(masks_np[np.arange(batch.size), actions_np]):
        raise ValueError("bc_regularizer actions must be allowed by action_masks")

    if batch.sample_weights is None:
        weights_np = np.ones(batch.size, dtype=np.float32)
    else:
        weights_np = np.asarray(batch.sample_weights, dtype=np.float32).reshape(-1)
        if weights_np.shape != (batch.size,):
            raise ValueError("bc_regularizer sample_weights must have shape [N]")
        if np.any(weights_np <= 0.0):
            raise ValueError("bc_regularizer sample_weights must be positive")

    return {
        "features": torch.as_tensor(features_np, dtype=torch.float32, device=device),
        "masks": torch.as_tensor(masks_np, dtype=torch.bool, device=device),
        "actions": torch.as_tensor(actions_np, dtype=torch.long, device=device),
        "weights": torch.as_tensor(weights_np, dtype=torch.float32, device=device),
    }


def _bc_regularizer_loss(
    model: ReplanActorCritic,
    tensors: dict[str, torch.Tensor],
    config: PPOUpdateConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(tensors["actions"].shape[0])
    minibatch_size = int(
        config.bc_regularizer_minibatch_size or config.minibatch_size
    )
    sample_size = min(minibatch_size, batch_size)
    indices_np = rng.choice(
        batch_size,
        size=sample_size,
        replace=batch_size < sample_size,
    )
    indices = torch.as_tensor(indices_np, dtype=torch.long, device=device)

    logits = model.actor(tensors["features"][indices])
    masked_logits = logits.masked_fill(
        ~tensors["masks"][indices],
        torch.finfo(logits.dtype).min,
    )
    actions = tensors["actions"][indices]
    weights = tensors["weights"][indices]
    per_example_loss = torch.nn.functional.cross_entropy(
        masked_logits,
        actions,
        reduction="none",
    )
    loss = (per_example_loss * weights).sum() / weights.sum().clamp_min(1.0e-8)
    predictions = torch.argmax(masked_logits, dim=-1)
    accuracy = (predictions == actions).float().mean()
    return loss, accuracy


def save_replan_actor_critic_checkpoint(
    path: str | Path,
    model: ReplanActorCritic,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save enough actor-critic metadata to restore the model shape."""

    payload = {
        "state_dict": model.state_dict(),
        "input_dim": int(model.input_dim),
        "hidden_dims": list(model.hidden_dims),
        "activation": model.activation,
        "metadata": dict(metadata or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_replan_actor_critic_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[ReplanActorCritic, dict[str, Any]]:
    """Load a replan actor-critic checkpoint saved by this module."""

    payload = torch.load(path, map_location=map_location)
    model = ReplanActorCritic(
        input_dim=int(payload["input_dim"]),
        hidden_dims=tuple(int(dim) for dim in payload["hidden_dims"]),
        activation=str(payload["activation"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, dict(payload.get("metadata", {}))


def _mean_stats(stats: list[dict[str, float]]) -> PPOUpdateStats:
    if not stats:
        raise ValueError("no PPO updates were run")
    return PPOUpdateStats(
        loss=float(np.mean([row["loss"] for row in stats])),
        policy_loss=float(np.mean([row["policy_loss"] for row in stats])),
        value_loss=float(np.mean([row["value_loss"] for row in stats])),
        entropy=float(np.mean([row["entropy"] for row in stats])),
        approx_kl=float(np.mean([row["approx_kl"] for row in stats])),
        clip_fraction=float(np.mean([row["clip_fraction"] for row in stats])),
        bc_regularizer_loss=float(
            np.mean([row["bc_regularizer_loss"] for row in stats])
        ),
        bc_regularizer_accuracy=float(
            np.mean([row["bc_regularizer_accuracy"] for row in stats])
        ),
        num_updates=len(stats),
    )


def ppo_config_to_dict(config: PPOUpdateConfig) -> dict[str, Any]:
    return asdict(config)
