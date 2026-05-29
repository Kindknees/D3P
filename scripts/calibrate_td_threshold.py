#!/usr/bin/env python3
"""Calibrate TD-error thresholds from collected transitions and a TD critic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptive_diffusion.td_critic import calibrate_td_thresholds  # noqa: E402


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in value.split(",") if part)
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--critic_checkpoint", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--percentiles", type=parse_float_list, default=(70, 80, 90, 95)
    )
    parser.add_argument(
        "--z_thresholds", type=parse_float_list, default=(0.5, 1.0, 1.5, 2.0)
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_json = args.output_json or args.dataset.parent / "td_threshold_calibration.json"
    summary = calibrate_td_thresholds(
        dataset_path=args.dataset,
        critic_checkpoint=args.critic_checkpoint,
        output_json=output_json,
        gamma=args.gamma,
        percentiles=args.percentiles,
        z_thresholds=args.z_thresholds,
        device=args.device,
    )
    summary["output_json"] = str(output_json)
    print(OmegaConf.to_yaml(OmegaConf.create(summary), resolve=True))


if __name__ == "__main__":
    main()
