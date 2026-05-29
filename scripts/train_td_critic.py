#!/usr/bin/env python3
"""Train a TD critic from executor transitions.npz."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptive_diffusion.td_critic import train_td_critic  # noqa: E402


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(part) for part in value.split(",") if part)
    if not dims:
        raise argparse.ArgumentTypeError("hidden_dims must contain at least one dim")
    if any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("hidden_dims must be positive")
    return dims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden_dims", type=parse_hidden_dims, default=(256, 256))
    parser.add_argument("--activation", choices=["relu", "tanh"], default="relu")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.dataset.parent / "td_critic"
    summary = train_td_critic(
        dataset_path=args.dataset,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        seed=args.seed,
        hidden_dims=args.hidden_dims,
        activation=args.activation,
        device=args.device,
    )
    print(OmegaConf.to_yaml(OmegaConf.create(summary), resolve=True))


if __name__ == "__main__":
    main()
