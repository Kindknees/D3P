"""Analysis and plotting helpers for adaptive replanning runs."""

from __future__ import annotations

import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

METRIC_KEYS = (
    "success_rate",
    "mean_episode_return",
    "median_episode_return",
    "mean_high_level_return",
    "mean_episode_length",
    "mean_nfe_per_action",
    "mean_inference_wall_time_per_action",
    "mean_controller_wall_time_per_action",
    "mean_total_wall_time_per_action",
    "replan_rate",
    "forced_replan_rate",
    "learned_replan_rate",
    "mean_discarded_actions_per_episode",
    "mean_buffer_remaining_when_replanned",
)

SUMMARY_COLUMNS = (
    "label",
    "env",
    "method",
    "seed",
    "episodes",
    *METRIC_KEYS,
    "source_path",
)

AGGREGATE_COLUMNS = (
    "label",
    "env",
    "method",
    "num_runs",
    "seeds",
    "episodes_total",
    "success_rate_mean",
    "success_rate_sem",
    "mean_episode_return_mean",
    "mean_episode_return_sem",
    "mean_high_level_return_mean",
    "mean_high_level_return_sem",
    "mean_nfe_per_action_mean",
    "mean_nfe_per_action_sem",
    "mean_total_wall_time_per_action_mean",
    "mean_total_wall_time_per_action_sem",
    "replan_rate_mean",
    "replan_rate_sem",
    "learned_replan_rate_mean",
    "learned_replan_rate_sem",
    "mean_discarded_actions_per_episode_mean",
    "mean_discarded_actions_per_episode_sem",
)


@dataclass(frozen=True)
class LabeledRun:
    label: str
    path: Path


def parse_labeled_run(value: str) -> LabeledRun:
    """Parse `label=/path/to/run` CLI values."""

    if "=" not in value:
        path = Path(value)
        return LabeledRun(label=path.name, path=path)
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("run label cannot be empty")
    return LabeledRun(label=label, path=Path(path))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_summary(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load an eval-like summary from a run directory or summary JSON file."""

    path = Path(path)
    if path.is_file():
        data = load_json(path)
        if path.name == "train_summary.json" and "eval_summary" in data:
            nested = dict(data["eval_summary"])
            nested.setdefault("env", data.get("env", ""))
            nested.setdefault("method", data.get("method", ""))
            nested.setdefault("seed", data.get("seed", ""))
            return nested, path
        return data, path

    if not path.exists():
        raise FileNotFoundError(f"run path does not exist: {path}")

    candidates = (
        path / "eval_summary.json",
        path / "eval" / "eval_summary.json",
        path / "train_summary.json",
    )
    for candidate in candidates:
        if not candidate.exists():
            continue
        data = load_json(candidate)
        if candidate.name == "train_summary.json" and "eval_summary" in data:
            nested = dict(data["eval_summary"])
            nested.setdefault("env", data.get("env", ""))
            nested.setdefault("method", data.get("method", ""))
            nested.setdefault("seed", data.get("seed", ""))
            return nested, candidate
        return data, candidate
    raise FileNotFoundError(f"no eval_summary.json or train_summary.json under {path}")


def discover_eval_summaries(input_paths: Iterable[str | Path]) -> list[LabeledRun]:
    """Discover eval summaries under input paths using directory names as labels."""

    runs: list[LabeledRun] = []
    seen: set[Path] = set()
    for input_path in input_paths:
        path = Path(input_path)
        if path.is_file():
            resolved = path.resolve()
            if resolved not in seen:
                runs.append(LabeledRun(label=path.parent.name, path=path))
                seen.add(resolved)
            continue
        if not path.exists():
            raise FileNotFoundError(f"input path does not exist: {path}")
        for summary_path in sorted(path.rglob("eval_summary.json")):
            resolved = summary_path.resolve()
            if resolved in seen:
                continue
            label_dir = summary_path.parent
            if label_dir.name == "eval":
                label_dir = label_dir.parent
            runs.append(LabeledRun(label=label_dir.name, path=summary_path))
            seen.add(resolved)
    return runs


def row_from_summary(label: str, summary: dict[str, Any], source_path: str | Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "label": label,
        "env": summary.get("env", ""),
        "method": summary.get("method", ""),
        "seed": summary.get("seed", ""),
        "episodes": summary.get("episodes", ""),
        "source_path": str(source_path),
    }
    for key in METRIC_KEYS:
        row[key] = summary.get(key, "")
    return {column: row.get(column, "") for column in SUMMARY_COLUMNS}


def load_rows(runs: Sequence[LabeledRun]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        summary, source_path = resolve_summary(run.path)
        rows.append(row_from_summary(run.label, summary, source_path))
    return rows


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _mean(values: Sequence[float]) -> float | str:
    if not values:
        return ""
    return float(sum(values) / len(values))


def _sem(values: Sequence[float]) -> float | str:
    if len(values) <= 1:
        return 0.0 if values else ""
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return float(math.sqrt(variance) / math.sqrt(len(values)))


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("label", "")), str(row.get("env", "")), str(row.get("method", "")))
        groups.setdefault(key, []).append(row)

    aggregates = []
    for (label, env, method), group_rows in sorted(groups.items()):
        aggregate: dict[str, Any] = {
            "label": label,
            "env": env,
            "method": method,
            "num_runs": len(group_rows),
            "seeds": ",".join(str(row.get("seed", "")) for row in group_rows),
            "episodes_total": int(
                sum(_float_or_none(row.get("episodes")) or 0.0 for row in group_rows)
            ),
        }
        for metric in (
            "success_rate",
            "mean_episode_return",
            "mean_high_level_return",
            "mean_nfe_per_action",
            "mean_total_wall_time_per_action",
            "replan_rate",
            "learned_replan_rate",
            "mean_discarded_actions_per_episode",
        ):
            values = [
                value
                for value in (_float_or_none(row.get(metric)) for row in group_rows)
                if value is not None
            ]
            aggregate[f"{metric}_mean"] = _mean(values)
            aggregate[f"{metric}_sem"] = _sem(values)
        aggregates.append(
            {column: aggregate.get(column, "") for column in AGGREGATE_COLUMNS}
        )
    return aggregates


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


INTEGER_COLUMNS = {"seed", "episodes", "num_runs", "episodes_total"}


def format_cell(column: str, value: Any) -> str:
    if column == "seeds":
        return str(value)
    number = _float_or_none(value)
    if number is not None:
        if column in INTEGER_COLUMNS:
            return str(int(round(number)))
        return f"{number:.3f}"
    return str(value)


def write_markdown_table(
    path: str | Path,
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
    title: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(format_cell(column, row.get(column, "")) for column in columns)
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _scale(value: float, low: float, high: float, out_low: float, out_high: float) -> float:
    if high <= low:
        return (out_low + out_high) / 2.0
    return out_low + (value - low) * (out_high - out_low) / (high - low)


def _color(index: int) -> str:
    palette = (
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#8c564b",
        "#e377c2",
    )
    return palette[index % len(palette)]


def write_pareto_svg(
    path: str | Path,
    rows: Sequence[dict[str, Any]],
    y_key: str,
    y_label: str,
    title: str,
    x_key: str = "mean_nfe_per_action_mean",
    x_label: str = "Mean NFE/action",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = []
    for row in rows:
        x = _float_or_none(row.get(x_key))
        y = _float_or_none(row.get(y_key))
        if x is None or y is None:
            continue
        points.append((row, x, y))

    width, height = 900, 580
    left, right, top, bottom = 86, 28, 58, 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    if points:
        x_values = [x for _, x, _ in points]
        y_values = [y for _, _, y in points]
        x_pad = max((max(x_values) - min(x_values)) * 0.08, 0.2)
        y_pad = max((max(y_values) - min(y_values)) * 0.08, 0.05)
        x_min, x_max = min(x_values) - x_pad, max(x_values) + x_pad
        y_min, y_max = min(0.0, min(y_values) - y_pad), max(y_values) + y_pad
    else:
        x_min, x_max, y_min, y_max = 0.0, 1.0, 0.0, 1.0

    labels = sorted({str(row.get("label", "")) for row, _, _ in points})
    color_by_label = {label: _color(index) for index, label in enumerate(labels)}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-family="Arial" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 28}" text-anchor="middle" font-family="Arial" font-size="14">{x_label}</text>',
        f'<text x="22" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 22 {top + plot_h / 2:.1f})">{y_label}</text>',
    ]

    for tick in range(6):
        tx = left + tick * plot_w / 5.0
        value = x_min + tick * (x_max - x_min) / 5.0
        svg.append(f'<line x1="{tx:.1f}" y1="{top}" x2="{tx:.1f}" y2="{top + plot_h}" stroke="#eee"/>')
        svg.append(f'<text x="{tx:.1f}" y="{top + plot_h + 20}" text-anchor="middle" font-family="Arial" font-size="11">{value:.2f}</text>')
        ty = top + plot_h - tick * plot_h / 5.0
        y_value = y_min + tick * (y_max - y_min) / 5.0
        svg.append(f'<line x1="{left}" y1="{ty:.1f}" x2="{left + plot_w}" y2="{ty:.1f}" stroke="#eee"/>')
        svg.append(f'<text x="{left - 10}" y="{ty + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{y_value:.2f}</text>')

    sorted_points = sorted(points, key=lambda item: (item[1], -item[2]))
    best_y = -float("inf")
    frontier: list[tuple[dict[str, Any], float, float]] = []
    for row, x, y in sorted_points:
        if y >= best_y:
            frontier.append((row, x, y))
            best_y = y
    if len(frontier) >= 2:
        path_d = []
        for index, (_row, x, y) in enumerate(frontier):
            px = _scale(x, x_min, x_max, left, left + plot_w)
            py = _scale(y, y_min, y_max, top + plot_h, top)
            path_d.append(("M" if index == 0 else "L") + f" {px:.1f} {py:.1f}")
        svg.append(f'<path d="{" ".join(path_d)}" fill="none" stroke="#111" stroke-width="2" stroke-dasharray="5 4"/>')

    for row, x, y in points:
        label = str(row.get("label", ""))
        color = color_by_label[label]
        px = _scale(x, x_min, x_max, left, left + plot_w)
        py = _scale(y, y_min, y_max, top + plot_h, top)
        svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="{color}" stroke="#222"/>')
        svg.append(f'<text x="{px + 8:.1f}" y="{py - 8:.1f}" font-family="Arial" font-size="11">{label}</text>')

    legend_x = left + plot_w - 180
    legend_y = top + 18
    for index, label in enumerate(labels):
        y = legend_y + index * 18
        svg.append(f'<circle cx="{legend_x}" cy="{y}" r="5" fill="{color_by_label[label]}"/>')
        svg.append(f'<text x="{legend_x + 12}" y="{y + 4}" font-family="Arial" font-size="11">{label}</text>')

    if not points:
        svg.append(f'<text x="{width / 2:.1f}" y="{height / 2:.1f}" text-anchor="middle" font-family="Arial" font-size="14">No numeric rows</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_step_metrics(path: str | Path) -> Path:
    """Resolve a step metrics CSV from a run directory or known artifact path."""

    path = Path(path)
    if path.is_file():
        if path.name in {"metrics_step.csv", "eval_metrics_step.csv"}:
            return path
        candidates = (
            path.parent / "eval_metrics_step.csv",
            path.parent / "metrics_step.csv",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"no step metrics CSV next to {path}")

    if not path.exists():
        raise FileNotFoundError(f"run path does not exist: {path}")

    candidates = (
        path / "eval_metrics_step.csv",
        path / "metrics_step.csv",
        path / "eval" / "eval_metrics_step.csv",
        path / "eval" / "metrics_step.csv",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no metrics_step.csv or eval_metrics_step.csv under {path}")


def _safe_filename(value: str) -> str:
    chars = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            chars.append(char)
        else:
            chars.append("_")
    return "".join(chars).strip("_") or "run"


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _episode_key(row: dict[str, Any]) -> tuple[int, str]:
    value = row.get("episode_id", "0")
    try:
        return int(float(value)), str(value)
    except (TypeError, ValueError):
        return 0, str(value)


def _step_key(row: dict[str, Any]) -> tuple[int, str]:
    value = row.get("t", "0")
    try:
        return int(float(value)), str(value)
    except (TypeError, ValueError):
        return 0, str(value)


def rows_for_episode(
    rows: Sequence[dict[str, str]],
    episode_id: str | int | None = None,
) -> tuple[str, list[dict[str, str]]]:
    if not rows:
        raise ValueError("step metrics rows cannot be empty")
    if episode_id is None:
        episode_id = sorted({_episode_key(row) for row in rows})[0][1]
    episode_id_str = str(episode_id)
    filtered = [row for row in rows if str(row.get("episode_id", "0")) == episode_id_str]
    if not filtered:
        raise ValueError(f"episode_id {episode_id_str} not found in step metrics")
    filtered.sort(key=_step_key)
    return episode_id_str, filtered


def _polyline(
    rows: Sequence[tuple[float, float]],
    color: str,
    width: float = 2.0,
) -> str:
    if not rows:
        return ""
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in rows)
    return (
        f'<polyline points="{points}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'
    )


def _timeline_scale(values: Sequence[float], pad: float = 0.05) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if high <= low:
        margin = max(abs(high) * pad, 1.0)
        return low - margin, high + margin
    margin = max((high - low) * pad, 1.0e-6)
    return low - margin, high + margin


def write_replan_timeline_svg(
    path: str | Path,
    rows: Sequence[dict[str, str]],
    label: str,
    episode_id: str | int,
) -> None:
    """Write one replan timeline SVG for one episode worth of step metrics."""

    if not rows:
        raise ValueError("cannot plot an empty timeline")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    steps = [float(_step_key(row)[0]) for row in rows]
    uncertainties = [_float_or_none(row.get("uncertainty")) or 0.0 for row in rows]
    td_errors = [_float_or_none(row.get("td_error")) or 0.0 for row in rows]
    rewards = [_float_or_none(row.get("env_reward")) or 0.0 for row in rows]
    remaining = [
        _float_or_none(row.get("buffer_remaining_after")) or 0.0 for row in rows
    ]
    ages = [_float_or_none(row.get("plan_age_before")) or 0.0 for row in rows]
    discarded = [_float_or_none(row.get("discarded_actions")) or 0.0 for row in rows]

    width, height = 1000, 620
    left, right, top = 82, 32, 58
    panel_h, gap = 190, 66
    top_y = top
    bottom_y = top + panel_h + gap
    plot_w = width - left - right
    x_min, x_max = min(steps), max(steps)
    if x_max <= x_min:
        x_max = x_min + 1.0
    signal_low, signal_high = _timeline_scale(uncertainties + td_errors + rewards)
    buffer_low, buffer_high = _timeline_scale(remaining + ages + discarded)

    def sx(value: float) -> float:
        return _scale(value, x_min, x_max, left, left + plot_w)

    def sy_signal(value: float) -> float:
        return _scale(value, signal_low, signal_high, top_y + panel_h, top_y)

    def sy_buffer(value: float) -> float:
        return _scale(value, buffer_low, buffer_high, bottom_y + panel_h, bottom_y)

    signal_uncertainty = [(sx(t), sy_signal(v)) for t, v in zip(steps, uncertainties)]
    signal_td = [(sx(t), sy_signal(v)) for t, v in zip(steps, td_errors)]
    signal_reward = [(sx(t), sy_signal(v)) for t, v in zip(steps, rewards)]
    buffer_remaining = [(sx(t), sy_buffer(v)) for t, v in zip(steps, remaining)]
    buffer_age = [(sx(t), sy_buffer(v)) for t, v in zip(steps, ages)]

    escaped_label = html.escape(str(label))
    escaped_episode = html.escape(str(episode_id))
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-family="Arial" font-size="20">{escaped_label} episode {escaped_episode} replan timeline</text>',
    ]
    for y0, title, y_low, y_high in (
        (top_y, "uncertainty / TD error / reward", signal_low, signal_high),
        (bottom_y, "buffer remaining / plan age / discards", buffer_low, buffer_high),
    ):
        svg.extend(
            [
                f'<rect x="{left}" y="{y0}" width="{plot_w}" height="{panel_h}" fill="#fafafa" stroke="#ddd"/>',
                f'<text x="{left}" y="{y0 - 10}" font-family="Arial" font-size="13">{title}</text>',
                f'<text x="{left - 8}" y="{y0 + 5}" text-anchor="end" font-family="Arial" font-size="10">{y_high:.2f}</text>',
                f'<text x="{left - 8}" y="{y0 + panel_h}" text-anchor="end" font-family="Arial" font-size="10">{y_low:.2f}</text>',
            ]
        )
        for tick in range(5):
            tx = left + tick * plot_w / 4.0
            value = x_min + tick * (x_max - x_min) / 4.0
            svg.append(f'<line x1="{tx:.1f}" y1="{y0}" x2="{tx:.1f}" y2="{y0 + panel_h}" stroke="#eee"/>')
            if y0 == bottom_y:
                svg.append(f'<text x="{tx:.1f}" y="{y0 + panel_h + 18}" text-anchor="middle" font-family="Arial" font-size="10">{value:.0f}</text>')

    for row, t, discard_count in zip(rows, steps, discarded):
        if _bool_value(row.get("replanned")):
            x = sx(t)
            forced = _bool_value(row.get("forced_replan"))
            color = "#999" if forced else "#d62728"
            svg.append(f'<line x1="{x:.1f}" y1="{top_y}" x2="{x:.1f}" y2="{bottom_y + panel_h}" stroke="{color}" stroke-width="1.4" stroke-dasharray="4 3"/>')
        if discard_count > 0.0:
            x = sx(t)
            y = sy_buffer(discard_count)
            svg.append(f'<rect x="{x - 3:.1f}" y="{y:.1f}" width="6" height="{bottom_y + panel_h - y:.1f}" fill="#ffbb78" opacity="0.75"/>')

    svg.extend(
        [
            _polyline(signal_uncertainty, "#1f77b4"),
            _polyline(signal_td, "#9467bd"),
            _polyline(signal_reward, "#2ca02c", width=1.5),
            _polyline(buffer_remaining, "#17becf"),
            _polyline(buffer_age, "#ff7f0e"),
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 24}" text-anchor="middle" font-family="Arial" font-size="13">environment step</text>',
        ]
    )
    legend = (
        ("uncertainty", "#1f77b4"),
        ("td_error", "#9467bd"),
        ("env_reward", "#2ca02c"),
        ("buffer_remaining", "#17becf"),
        ("plan_age", "#ff7f0e"),
        ("learned replan", "#d62728"),
        ("forced replan", "#999"),
    )
    legend_x = left + plot_w - 176
    legend_y = top_y + 18
    for index, (name, color) in enumerate(legend):
        y = legend_y + index * 18
        svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 18}" y2="{y}" stroke="{color}" stroke-width="2"/>')
        svg.append(f'<text x="{legend_x + 24}" y="{y + 4}" font-family="Arial" font-size="11">{html.escape(name)}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(part for part in svg if part), encoding="utf-8")


def write_timeline_outputs(
    output_dir: str | Path,
    runs: Sequence[LabeledRun],
    max_episodes_per_run: int = 1,
) -> list[str]:
    if max_episodes_per_run <= 0:
        raise ValueError("max_episodes_per_run must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timeline_dir = output_dir / "timelines"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    paths: list[str] = []
    name_counts: dict[str, int] = {}
    for run in runs:
        step_path = resolve_step_metrics(run.path)
        rows = read_csv_rows(step_path)
        episode_ids = [episode_id for _key, episode_id in sorted({_episode_key(row) for row in rows})]
        for episode_id in episode_ids[:max_episodes_per_run]:
            resolved_episode, episode_rows = rows_for_episode(rows, episode_id)
            base_name = (
                f"replan_timeline_{_safe_filename(run.label)}_"
                f"ep{_safe_filename(resolved_episode)}"
            )
            duplicate_index = name_counts.get(base_name, 0)
            name_counts[base_name] = duplicate_index + 1
            file_name = base_name if duplicate_index == 0 else f"{base_name}_run{duplicate_index}"
            output_path = timeline_dir / f"{file_name}.svg"
            write_replan_timeline_svg(output_path, episode_rows, run.label, resolved_episode)
            paths.append(str(output_path))
            manifest_rows.append(
                {
                    "label": run.label,
                    "episode_id": resolved_episode,
                    "step_metrics_path": str(step_path),
                    "timeline_path": str(output_path),
                }
            )
    write_csv(
        output_dir / "timeline_manifest.csv",
        manifest_rows,
        ("label", "episode_id", "step_metrics_path", "timeline_path"),
    )
    return paths


def write_analysis_outputs(output_dir: str | Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregates = aggregate_rows(rows)

    write_csv(output_dir / "summary_table.csv", rows, SUMMARY_COLUMNS)
    write_csv(output_dir / "aggregate_summary.csv", aggregates, AGGREGATE_COLUMNS)
    write_markdown_table(
        output_dir / "summary_table.md",
        rows,
        (
            "label",
            "env",
            "seed",
            "episodes",
            "success_rate",
            "mean_episode_return",
            "mean_nfe_per_action",
            "replan_rate",
            "learned_replan_rate",
            "mean_discarded_actions_per_episode",
        ),
        "Adaptive Replanning Summary Table",
    )
    write_markdown_table(
        output_dir / "aggregate_summary.md",
        aggregates,
        (
            "label",
            "env",
            "num_runs",
            "seeds",
            "success_rate_mean",
            "success_rate_sem",
            "mean_episode_return_mean",
            "mean_nfe_per_action_mean",
            "replan_rate_mean",
            "learned_replan_rate_mean",
        ),
        "Adaptive Replanning Aggregate Summary",
    )

    plot_paths = []
    envs = sorted({str(row.get("env", "")) for row in aggregates})
    for env in envs:
        env_rows = [row for row in aggregates if str(row.get("env", "")) == env]
        safe_env = env or "unknown_env"
        success_path = output_dir / f"pareto_{safe_env}_success.svg"
        return_path = output_dir / f"pareto_{safe_env}_return.svg"
        write_pareto_svg(
            success_path,
            env_rows,
            y_key="success_rate_mean",
            y_label="Success rate",
            title=f"{safe_env} success vs NFE/action",
        )
        write_pareto_svg(
            return_path,
            env_rows,
            y_key="mean_episode_return_mean",
            y_label="Mean episode return",
            title=f"{safe_env} return vs NFE/action",
        )
        plot_paths.extend([str(success_path), str(return_path)])

    summary = {
        "num_rows": len(rows),
        "num_aggregates": len(aggregates),
        "envs": envs,
        "outputs": {
            "summary_table_csv": str(output_dir / "summary_table.csv"),
            "summary_table_md": str(output_dir / "summary_table.md"),
            "aggregate_summary_csv": str(output_dir / "aggregate_summary.csv"),
            "aggregate_summary_md": str(output_dir / "aggregate_summary.md"),
            "pareto_plots": plot_paths,
        },
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
