"""TD critic utilities for TD-error replanning signals."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TransitionDataset:
    obs: np.ndarray
    next_obs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

    @property
    def obs_dim(self) -> int:
        return int(self.obs.shape[1])

    @property
    def size(self) -> int:
        return int(self.obs.shape[0])


def flatten_state_obs(obs: Mapping[str, Any]) -> np.ndarray:
    """Flatten the state observation used by state-based Robomimic runs."""

    if "state" not in obs:
        raise KeyError("TD critic expects observation dict with key 'state'")
    return np.asarray(obs["state"], dtype=np.float32).reshape(-1)


def load_transition_dataset(path: str | Path) -> TransitionDataset:
    """Load an executor `transitions.npz` file."""

    with np.load(path) as data:
        obs = np.asarray(data["obs"], dtype=np.float32)
        next_obs = np.asarray(data["next_obs"], dtype=np.float32)
        rewards = np.asarray(data["rewards"], dtype=np.float32).reshape(-1)
        dones = np.asarray(data["dones"], dtype=np.float32).reshape(-1)

    if obs.ndim != 2 or next_obs.ndim != 2:
        raise ValueError("obs and next_obs must have shape [N, obs_dim]")
    if obs.shape != next_obs.shape:
        raise ValueError("obs and next_obs shapes must match")
    if len(rewards) != len(obs) or len(dones) != len(obs):
        raise ValueError("rewards/dones length must match obs length")
    if len(obs) == 0:
        raise ValueError("transition dataset is empty")
    return TransitionDataset(obs=obs, next_obs=next_obs, rewards=rewards, dones=dones)


class TDCritic(nn.Module):
    """Small MLP value model V(o) for TD-error signals."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        if activation not in {"relu", "tanh"}:
            raise ValueError("activation must be 'relu' or 'tanh'")
        self.obs_dim = int(obs_dim)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.activation = activation

        dims = [self.obs_dim, *self.hidden_dims, 1]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.tanh(x) if self.activation == "tanh" else F.relu(x)
        return self.layers[-1](x).squeeze(-1)


class TDCriticValueFunction:
    """Callable value function loaded from a trained TD critic checkpoint."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        payload = torch.load(checkpoint_path, map_location=self.device)
        self.obs_mean = torch.as_tensor(
            payload["obs_mean"], dtype=torch.float32, device=self.device
        )
        self.obs_std = torch.as_tensor(
            payload["obs_std"], dtype=torch.float32, device=self.device
        )
        self.model = TDCritic(
            obs_dim=int(payload["obs_dim"]),
            hidden_dims=tuple(payload["hidden_dims"]),
            activation=str(payload["activation"]),
        ).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()

    def __call__(self, obs: Mapping[str, Any]) -> float:
        return float(self.values(flatten_state_obs(obs))[0])

    def values(self, obs: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Evaluate V(o) for a batch of flattened observations."""

        array = np.asarray(obs, dtype=np.float32)
        if array.ndim == 1:
            array = array[None]
        if array.ndim != 2:
            raise ValueError("obs must have shape [obs_dim] or [N, obs_dim]")
        if array.shape[1] != self.obs_mean.numel():
            raise ValueError(
                f"obs_dim {array.shape[1]} does not match critic "
                f"obs_dim {self.obs_mean.numel()}"
            )

        outputs = []
        with torch.no_grad():
            for start in range(0, len(array), batch_size):
                batch = torch.from_numpy(array[start : start + batch_size]).float().to(
                    self.device
                )
                batch = (batch - self.obs_mean) / self.obs_std
                value = self.model(batch)
                outputs.append(value.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)


def format_percentile_key(percentile: float) -> str:
    value = float(percentile)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def running_z_scores(values: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return online z-scores matching `TDErrorSignal` normalization."""

    if eps <= 0.0:
        raise ValueError("eps must be positive")
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    z_scores = np.zeros_like(array, dtype=np.float32)
    means = np.zeros_like(array, dtype=np.float32)
    stds = np.zeros_like(array, dtype=np.float32)
    count = 0
    mean = 0.0
    m2 = 0.0
    for i, value in enumerate(array):
        count += 1
        delta = float(value) - mean
        mean += delta / count
        delta2 = float(value) - mean
        m2 += delta * delta2
        std = float(np.sqrt(m2 / (count - 1))) if count > 1 else 0.0
        z_scores[i] = 0.0 if std <= eps else (float(value) - mean) / (std + eps)
        means[i] = mean
        stds[i] = std
    return z_scores, means, stds


def compute_td_error_statistics(
    dataset: TransitionDataset,
    value_fn: Callable[[Mapping[str, Any]], float] | TDCriticValueFunction,
    gamma: float = 0.99,
    eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Compute raw TD errors and online-normalized uncertainty values."""

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if hasattr(value_fn, "values"):
        values = value_fn.values(dataset.obs)
        next_values = value_fn.values(dataset.next_obs)
    else:
        values = np.asarray(
            [value_fn({"state": obs}) for obs in dataset.obs], dtype=np.float32
        )
        next_values = np.asarray(
            [value_fn({"state": obs}) for obs in dataset.next_obs], dtype=np.float32
        )
    not_done = 1.0 - dataset.dones.astype(np.float32)
    td_errors = dataset.rewards + float(gamma) * next_values * not_done - values
    abs_td_errors = np.abs(td_errors).astype(np.float32)
    uncertainties, running_mean, running_std = running_z_scores(abs_td_errors, eps=eps)
    return {
        "values": values.astype(np.float32),
        "next_values": next_values.astype(np.float32),
        "td_errors": td_errors.astype(np.float32),
        "abs_td_errors": abs_td_errors,
        "uncertainties": uncertainties,
        "running_mean": running_mean,
        "running_std": running_std,
    }


def calibrate_td_thresholds(
    dataset_path: str | Path,
    critic_checkpoint: str | Path,
    output_json: str | Path | None = None,
    gamma: float = 0.99,
    percentiles: Sequence[float] = (70, 80, 90, 95),
    z_thresholds: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
    device: str = "cpu",
) -> dict[str, Any]:
    """Calibrate TD-error uncertainty thresholds from collected transitions."""

    dataset = load_transition_dataset(dataset_path)
    value_fn = TDCriticValueFunction(critic_checkpoint, device=device)
    stats = compute_td_error_statistics(dataset, value_fn=value_fn, gamma=gamma)
    uncertainties = stats["uncertainties"]
    abs_td_errors = stats["abs_td_errors"]
    percentile_thresholds = {
        format_percentile_key(percentile): float(np.percentile(uncertainties, percentile))
        for percentile in percentiles
    }
    abs_td_error_percentiles = {
        format_percentile_key(percentile): float(np.percentile(abs_td_errors, percentile))
        for percentile in percentiles
    }
    summary = {
        "dataset_path": str(dataset_path),
        "critic_checkpoint": str(critic_checkpoint),
        "num_transitions": dataset.size,
        "obs_dim": dataset.obs_dim,
        "gamma": float(gamma),
        "percentiles": [float(percentile) for percentile in percentiles],
        "percentile_thresholds": percentile_thresholds,
        "z_score_thresholds": [float(threshold) for threshold in z_thresholds],
        "abs_td_error_percentiles": abs_td_error_percentiles,
        "mean_abs_td_error": float(np.mean(abs_td_errors)),
        "std_abs_td_error": float(np.std(abs_td_errors)),
        "max_abs_td_error": float(np.max(abs_td_errors)),
        "mean_uncertainty": float(np.mean(uncertainties)),
        "std_uncertainty": float(np.std(uncertainties)),
        "max_uncertainty": float(np.max(uncertainties)),
        "min_uncertainty": float(np.min(uncertainties)),
    }
    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return summary


def train_td_critic(
    dataset_path: str | Path,
    output_dir: str | Path,
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    seed: int = 0,
    hidden_dims: Sequence[int] = (256, 256),
    activation: str = "relu",
    device: str = "cpu",
) -> dict[str, Any]:
    """Train a semi-gradient TD(0) critic from collected transitions."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = load_transition_dataset(dataset_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device)

    obs_mean = dataset.obs.mean(axis=0).astype(np.float32)
    obs_std = dataset.obs.std(axis=0).astype(np.float32)
    obs_std = np.where(obs_std < 1e-6, 1.0, obs_std).astype(np.float32)

    obs = torch.from_numpy((dataset.obs - obs_mean) / obs_std).float().to(torch_device)
    next_obs = torch.from_numpy((dataset.next_obs - obs_mean) / obs_std).float().to(
        torch_device
    )
    rewards = torch.from_numpy(dataset.rewards).float().to(torch_device)
    dones = torch.from_numpy(dataset.dones).float().to(torch_device)

    model = TDCritic(
        obs_dim=dataset.obs_dim,
        hidden_dims=hidden_dims,
        activation=activation,
    ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    last_loss = 0.0
    for _ in range(epochs):
        order = torch.randperm(dataset.size, device=torch_device)
        for start in range(0, dataset.size, batch_size):
            idx = order[start : start + batch_size]
            pred = model(obs[idx])
            with torch.no_grad():
                bootstrap = model(next_obs[idx])
                target = rewards[idx] + gamma * bootstrap * (1.0 - dones[idx])
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())

    checkpoint_path = output_path / "td_critic.pt"
    summary = {
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(checkpoint_path),
        "num_transitions": dataset.size,
        "obs_dim": dataset.obs_dim,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "gamma": float(gamma),
        "hidden_dims": [int(dim) for dim in hidden_dims],
        "activation": activation,
        "seed": int(seed),
        "final_loss": float(last_loss),
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_dim": dataset.obs_dim,
            "hidden_dims": [int(dim) for dim in hidden_dims],
            "activation": activation,
            "obs_mean": torch.from_numpy(obs_mean),
            "obs_std": torch.from_numpy(obs_std),
            "gamma": float(gamma),
            "summary": summary,
        },
        checkpoint_path,
    )
    with (output_path / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
