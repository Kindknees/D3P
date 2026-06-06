"""CSV logging utilities for adaptive diffusion execution experiments."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Mapping


STEP_LOG_FIELDS = [
    "task",
    "seed",
    "episode_id",
    "timestep",
    "method",
    "phase",
    "plan_age",
    "remaining_buffer_length",
    "commitment_horizon",
    "committed_steps_remaining",
    "td_error",
    "policy_disagreement",
    "likelihood_score",
    "criticality_score",
    "triggered_replan",
    "repair_candidate_generated",
    "repair_accepted",
    "repair_rejected",
    "random_reset",
    "discarded_nonempty_buffer",
    "action_jump",
    "buffer_jump",
    "executed_action_l2",
    "action_noise_applied",
    "action_noise_l2",
    "reward",
    "done",
    "success",
    "nfe_this_step",
    "wall_time_ms",
]

EVENT_LOG_FIELDS = [
    "task",
    "seed",
    "episode_id",
    "timestep",
    "method",
    "event_type",
    "phase",
    "plan_age",
    "remaining_buffer_length",
    "td_error",
    "likelihood_score",
    "criticality_score",
    "old_first_action_l2",
    "new_first_action_l2",
    "action_jump",
    "buffer_jump",
    "repair_noise_ratio",
    "repair_anchor_rho",
    "num_candidates",
    "selected_candidate_score",
    "nfe_event",
    "accepted",
    "fallback_reason",
]

EPISODE_SUMMARY_FIELDS = [
    "task",
    "seed",
    "episode_id",
    "method",
    "success",
    "episode_return",
    "episode_length",
    "num_replans",
    "num_repairs",
    "num_repairs_accepted",
    "num_repairs_rejected",
    "num_repair_candidates",
    "num_resets",
    "num_random_resets",
    "num_buffer_empty_resets",
    "num_discard_nonempty_buffer",
    "total_nfe",
    "nfe_per_action",
    "mean_action_jump",
    "mean_buffer_jump",
    "max_action_jump",
    "max_buffer_jump",
    "wall_time_total_ms",
]


class ExecutionLogger:
    """Write stable CSV schemas for step, event, and episode-level logs."""

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        config_resolved: str | Mapping[str, Any] | None = None,
        enabled: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self._files: list[Any] = []
        self._writers: dict[str, csv.DictWriter[str]] = {}
        if not enabled:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._open_csv("step", "step_logs.csv", STEP_LOG_FIELDS)
        self._open_csv("event", "event_logs.csv", EVENT_LOG_FIELDS)
        self._open_csv("episode", "episode_summary.csv", EPISODE_SUMMARY_FIELDS)
        self.write_config_resolved(config_resolved)

    def _open_csv(self, key: str, filename: str, fields: list[str]) -> None:
        file_obj = (self.output_dir / filename).open("w", newline="")
        writer = csv.DictWriter(file_obj, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        self._files.append(file_obj)
        self._writers[key] = writer

    def write_config_resolved(
        self, config_resolved: str | Mapping[str, Any] | None
    ) -> None:
        if not self.enabled:
            return
        config_path = self.output_dir / "config_resolved.yaml"
        if config_resolved is None:
            config_path.write_text("", encoding="utf-8")
        elif isinstance(config_resolved, str):
            config_path.write_text(config_resolved, encoding="utf-8")
        else:
            config_path.write_text(
                _mapping_to_yaml(config_resolved), encoding="utf-8"
            )

    def log_step(self, **row: Any) -> None:
        self._write("step", STEP_LOG_FIELDS, row)

    def log_event(self, **row: Any) -> None:
        self._write("event", EVENT_LOG_FIELDS, row)

    def log_episode_summary(self, **row: Any) -> None:
        self._write("episode", EPISODE_SUMMARY_FIELDS, row)

    def _write(self, key: str, fields: list[str], row: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        normalized = {field: _cell_value(row.get(field, "")) for field in fields}
        self._writers[key].writerow(normalized)

    def flush(self) -> None:
        for file_obj in self._files:
            file_obj.flush()

    def close(self) -> None:
        for file_obj in self._files:
            file_obj.close()
        self._files = []

    def __enter__(self) -> "ExecutionLogger":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def _cell_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    return value


def _mapping_to_yaml(mapping: Mapping[str, Any], indent: int = 0) -> str:
    lines = []
    prefix = " " * indent
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            lines.append(f"{prefix}{key}:")
            lines.append(_mapping_to_yaml(value, indent + 2).rstrip())
        else:
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(lines) + "\n"
