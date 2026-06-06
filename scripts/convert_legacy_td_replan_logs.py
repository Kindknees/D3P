"""Convert legacy TD-replan episode CSVs into adaptive-commitment summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from utils.execution_logger import EPISODE_SUMMARY_FIELDS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_root", help="Directory containing legacy episodes.csv files.")
    parser.add_argument("output_dir", help="Directory for converted episode_summary.csv.")
    args = parser.parse_args()

    legacy_root = Path(args.legacy_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "episode_summary.csv"

    rows = []
    for path in sorted(legacy_root.rglob("episodes.csv")):
        method = infer_method_name(path.parent.name)
        with path.open(newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            for row in reader:
                rows.append(convert_row(row, method))

    if not rows:
        raise SystemExit(f"No legacy episodes.csv files found under {legacy_root}")

    with output_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=EPISODE_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


def infer_method_name(run_name: str) -> str:
    if "chunk_only" in run_name or "calibration" in run_name:
        return "fixed_chunk"
    if "td_q" in run_name:
        return f"td_replan_anywhere_random_reset_{run_name.rsplit('_', 1)[-1]}"
    if "random" in run_name:
        return f"random_reset_{run_name.rsplit('_', 1)[-1]}"
    if "always" in run_name:
        return "reset_every_step"
    if "fixed2" in run_name:
        return "fixed_commitment_2"
    return run_name


def convert_row(row: dict[str, str], method: str) -> dict[str, object]:
    total_nfe = _float(row.get("total_nfe"))
    episode_length = max(1.0, _float(row.get("episode_length")))
    return {
        "task": row.get("env_name", ""),
        "seed": row.get("episode_seed", row.get("seed", "")),
        "episode_id": row.get("episode", ""),
        "method": method,
        "success": _boolish(row.get("success")),
        "episode_return": _float(row.get("episode_return")),
        "num_repairs": 0,
        "num_resets": _float(row.get("num_replans")),
        "num_random_resets": _float(row.get("num_random_replans")),
        "num_buffer_empty_resets": _float(row.get("num_empty_buffer_replans")),
        "num_discard_nonempty_buffer": _float(row.get("discarded_actions")),
        "total_nfe": total_nfe,
        "nfe_per_action": total_nfe / episode_length,
        "mean_action_jump": "",
        "mean_buffer_jump": "",
        "wall_time_total_ms": _float(row.get("total_sample_time")) * 1000,
    }


def _float(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _boolish(value: str | None) -> int:
    if value is None:
        return 0
    return int(value.lower() in ("1", "true", "yes"))


if __name__ == "__main__":
    main()
