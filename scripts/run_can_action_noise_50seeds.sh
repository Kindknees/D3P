#!/usr/bin/env bash
set -euo pipefail

# Run the Can H=4 action perturbation comparison over many random seeds and
# aggregate mean/std metrics.
#
# Run from the repo root in WSL:
#   conda activate d3p
#   bash scripts/run_can_action_noise_50seeds.sh
#
# Optional overrides:
#   SEEDS="$(seq 0 9)" bash scripts/run_can_action_noise_50seeds.sh
#   NOISE_LEVELS="0 0.2 0.5" bash scripts/run_can_action_noise_50seeds.sh
#   ENV_N_ENVS=10 bash scripts/run_can_action_noise_50seeds.sh

SEEDS="${SEEDS:-$(seq 0 49)}"
NOISE_LEVELS="${NOISE_LEVELS:-0 0.08 0.2 0.5}"
ENV_N_ENVS="${ENV_N_ENVS:-10}"
OUT_ROOT="${OUT_ROOT:-outputs/can_action_noise_50seeds_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUT_ROOT"

for seed in $SEEDS; do
  seed_dir="$OUT_ROOT/seed_${seed}"
  mkdir -p "$seed_dir"
  echo "=== Running seed ${seed} ==="
  SEED="$seed" \
  ENV_N_ENVS="$ENV_N_ENVS" \
  NOISE_LEVELS="$NOISE_LEVELS" \
  OUT_DIR="$seed_dir" \
    bash scripts/run_can_action_noise_eval.sh | tee "$seed_dir/runner.log"
done

python - "$OUT_ROOT" <<'PY'
import csv
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])

patterns = {
    "no_td": re.compile(
        r"success rate\s+(?P<success>[0-9.]+)\s+\| avg episode reward\s+(?P<reward>[0-9.]+).*avg NFE\s+(?P<nfe>[0-9.]+)"
    ),
    "td": re.compile(
        r"success\s+(?P<success>[0-9.]+)\s+\| reward\s+(?P<reward>[0-9.]+).*avg NFE\s+(?P<nfe>[0-9.]+)\s+\| replan\s+(?P<replan>[0-9.]+)\s+\| trigger\s+(?P<trigger>[0-9.]+)"
    ),
}

rows = []
for seed_dir in sorted(root.glob("seed_*")):
    seed = seed_dir.name.split("_", 1)[1]
    for log_path in sorted(seed_dir.glob("noise_*_*.log")):
        stem = log_path.stem
        if stem.endswith("_no_td"):
            method = "no_td"
            noise = stem[len("noise_") : -len("_no_td")]
        elif stem.endswith("_td"):
            method = "td"
            noise = stem[len("noise_") : -len("_td")]
        else:
            continue
        noise = noise.replace("p", ".")
        text = log_path.read_text(encoding="utf-8", errors="replace")
        eval_lines = [line for line in text.splitlines() if "eval:" in line]
        if not eval_lines:
            continue
        match = patterns[method].search(eval_lines[-1])
        if match is None:
            continue
        row = {
            "seed": seed,
            "noise": noise,
            "method": method,
            "success": float(match.group("success")),
            "reward": float(match.group("reward")),
            "nfe": float(match.group("nfe")),
            "replan": "",
            "trigger": "",
            "log_path": str(log_path),
        }
        if method == "td":
            row["replan"] = float(match.group("replan"))
            row["trigger"] = float(match.group("trigger"))
        rows.append(row)

raw_path = root / "raw_results.csv"
with raw_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "seed",
            "noise",
            "method",
            "success",
            "reward",
            "nfe",
            "replan",
            "trigger",
            "log_path",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

def mean_std(values):
    values = list(values)
    if not values:
        return "", ""
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std

summary_rows = []
groups = sorted({(row["noise"], row["method"]) for row in rows})
for noise, method in groups:
    subset = [row for row in rows if row["noise"] == noise and row["method"] == method]
    success_mean, success_std = mean_std(row["success"] for row in subset)
    reward_mean, reward_std = mean_std(row["reward"] for row in subset)
    nfe_mean, nfe_std = mean_std(row["nfe"] for row in subset)
    replan_values = [row["replan"] for row in subset if row["replan"] != ""]
    trigger_values = [row["trigger"] for row in subset if row["trigger"] != ""]
    replan_mean, replan_std = mean_std(replan_values)
    trigger_mean, trigger_std = mean_std(trigger_values)
    summary_rows.append(
        {
            "noise": noise,
            "method": method,
            "n": len(subset),
            "success_mean": success_mean,
            "success_std": success_std,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "nfe_mean": nfe_mean,
            "nfe_std": nfe_std,
            "replan_mean": replan_mean,
            "replan_std": replan_std,
            "trigger_mean": trigger_mean,
            "trigger_std": trigger_std,
        }
    )

summary_path = root / "aggregate_summary.csv"
with summary_path.open("w", newline="", encoding="utf-8") as f:
    fieldnames = list(summary_rows[0].keys()) if summary_rows else [
        "noise", "method", "n", "success_mean", "success_std", "reward_mean",
        "reward_std", "nfe_mean", "nfe_std", "replan_mean", "replan_std",
        "trigger_mean", "trigger_std",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)

md_path = root / "aggregate_summary.md"
with md_path.open("w", encoding="utf-8") as f:
    f.write("# Can H=4 Action Noise 50-Seed Summary\n\n")
    f.write("| Noise | Method | N | Success | Reward | Avg NFE | Replan | Trigger |\n")
    f.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
    for row in summary_rows:
        def fmt_pair(mean_key, std_key):
            mean = row[mean_key]
            std = row[std_key]
            if mean == "":
                return ""
            return f"{mean:.4f} +/- {std:.4f}"
        f.write(
            "| "
            f"{row['noise']} | {row['method']} | {row['n']} | "
            f"{fmt_pair('success_mean', 'success_std')} | "
            f"{fmt_pair('reward_mean', 'reward_std')} | "
            f"{fmt_pair('nfe_mean', 'nfe_std')} | "
            f"{fmt_pair('replan_mean', 'replan_std')} | "
            f"{fmt_pair('trigger_mean', 'trigger_std')} |\n"
        )

print(f"Wrote {raw_path}")
print(f"Wrote {summary_path}")
print(f"Wrote {md_path}")
PY

echo "All seed logs and aggregate summaries saved to: $OUT_ROOT"
