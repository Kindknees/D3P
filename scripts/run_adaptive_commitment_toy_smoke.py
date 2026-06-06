"""Run a deterministic toy smoke for the adaptive commitment pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from adaptive_diffusion.controllers.three_way import (
    AdaptiveCommitmentController,
    ThreeWayControllerConfig,
    ThreeWayHeuristicController,
)
from adaptive_diffusion.executor import ExecutorConfig, run_executor_episodes
from adaptive_diffusion.repair import RepairConfig, finalize_repair_candidate


class LineReachEnv:
    """One-dimensional reach task used only to smoke-test the executor."""

    def __init__(self, max_steps: int = 6):
        self.max_steps = max_steps
        self.x = 0.0
        self.step_count = 0

    def reset(self, seed=None):
        self.x = 0.0
        self.step_count = 0
        return np.array([self.x], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        self.x += float(action[0])
        success = self.x >= 1.0
        done = success or self.step_count >= self.max_steps
        reward = 1.0 if success else 0.0
        return np.array([self.x], dtype=np.float32), reward, done, {"x": self.x}


class FragilePolicy:
    nfe_per_sample = 4

    def sample_action_chunk(self, obs):
        return np.array([[0.25], [-0.25], [0.25], [-0.25]], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="outputs/adaptive_commitment/toy_smoke",
        help="Directory for toy smoke logs.",
    )
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    run_fixed(output_root / "fixed_chunk", args.episodes)
    run_adaptive(output_root / "adaptive_commitment_heuristic", args.episodes)
    run_repair(output_root / "adaptive_commitment_with_repair", args.episodes)
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_adaptive_commitment.py",
            str(output_root),
            "--baseline",
            "fixed_chunk",
            "--fallback-baseline",
            "fixed_chunk",
            "--max-nfe-ratio",
            "2.0",
        ],
        check=True,
    )


def run_fixed(output_dir: Path, episodes: int) -> None:
    config = ExecutorConfig(
        method="fixed_chunk",
        task="toy_line_reach",
        seed=0,
        action_dim=1,
        chunk_size=4,
        max_steps=6,
        success_threshold=1.0,
    )
    run_executor_episodes(
        env_factory=LineReachEnv,
        policy_factory=FragilePolicy,
        config=config,
        output_dir=output_dir,
        num_episodes=episodes,
    )


def run_adaptive(output_dir: Path, episodes: int) -> None:
    config = ExecutorConfig(
        method="adaptive_commitment_heuristic",
        task="toy_line_reach",
        seed=0,
        action_dim=1,
        chunk_size=4,
        max_steps=6,
        success_threshold=1.0,
    )
    controller = AdaptiveCommitmentController(
        high_td_threshold=0.9,
        medium_td_threshold=0.5,
    )
    run_executor_episodes(
        env_factory=LineReachEnv,
        policy_factory=FragilePolicy,
        config=config,
        output_dir=output_dir,
        controller=controller,
        uncertainty_fn=toy_uncertainty,
        num_episodes=episodes,
    )


def run_repair(output_dir: Path, episodes: int) -> None:
    config = ExecutorConfig(
        method="adaptive_commitment_with_repair",
        task="toy_line_reach",
        seed=0,
        action_dim=1,
        chunk_size=4,
        max_steps=6,
        success_threshold=1.0,
        nfe_per_repair=2,
        repair_config=RepairConfig(target_len=4, anchor_rho=1.0),
    )
    controller = ThreeWayHeuristicController(
        ThreeWayControllerConfig(
            repair_td_threshold=0.5,
            high_td_threshold=0.9,
            medium_td_threshold=0.5,
        )
    )
    run_executor_episodes(
        env_factory=LineReachEnv,
        policy_factory=FragilePolicy,
        config=config,
        output_dir=output_dir,
        controller=controller,
        uncertainty_fn=toy_uncertainty,
        repair_fn=toy_repair,
        num_episodes=episodes,
    )


def toy_uncertainty(obs, info, step):
    return 1.0 if step == 0 else 0.0


def toy_repair(obs, residual, repair_config):
    repaired = residual.copy()
    repaired[:, 0] = 0.5
    return finalize_repair_candidate(residual, repaired, repair_config)


if __name__ == "__main__":
    main()
