#!/usr/bin/env python3
"""Create summary tables and Pareto SVG plots from eval outputs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptive_diffusion.plots import (  # noqa: E402
    discover_eval_summaries,
    load_rows,
    parse_labeled_run,
    write_analysis_outputs,
    write_timeline_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        default=[],
        help="Directory or summary JSON to discover recursively. Can be repeated.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Explicit LABEL=PATH run or eval_summary.json. Can be repeated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for CSV, Markdown, JSON, and SVG artifacts.",
    )
    parser.add_argument(
        "--plot",
        default="pareto,summary_table",
        help="Comma-separated plot/output groups. Supported: pareto,summary_table,replan_timeline.",
    )
    parser.add_argument(
        "--timeline_episodes",
        type=int,
        default=1,
        help="Number of episodes per run to render for replan_timeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = [parse_labeled_run(value) for value in args.run]
    runs.extend(discover_eval_summaries(args.input))
    if not runs:
        raise SystemExit("provide at least one --run or --input")

    requested_plots = {part.strip() for part in args.plot.split(",") if part.strip()}
    supported = {"pareto", "summary_table", "replan_timeline"}
    unsupported = requested_plots - supported
    if unsupported:
        raise SystemExit(
            "unsupported --plot values: " + ",".join(sorted(unsupported))
        )
    if args.timeline_episodes <= 0:
        raise SystemExit("--timeline_episodes must be positive")

    output_dir = args.output
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / "outputs" / "plots" / f"{timestamp}_adaptive_replanning"

    rows = load_rows(runs)
    summary = write_analysis_outputs(output_dir, rows)
    if "replan_timeline" in requested_plots:
        timeline_paths = write_timeline_outputs(
            output_dir,
            runs,
            max_episodes_per_run=args.timeline_episodes,
        )
        summary["outputs"]["replan_timeline_plots"] = timeline_paths
        summary["outputs"]["timeline_manifest"] = str(output_dir / "timeline_manifest.csv")
        with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print(f"Wrote analysis outputs to {output_dir}")
    print(OmegaConf.to_yaml(OmegaConf.create(summary), resolve=True))


if __name__ == "__main__":
    main()
