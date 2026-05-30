#!/usr/bin/env bash
set -euo pipefail

# Reproduce the Can H=4 action perturbation comparison:
#   no-TD fixed chunk vs TD replan, under several action-noise levels.
#
# Run from the repo root in WSL:
#   conda activate d3p
#   bash scripts/run_can_action_noise_eval.sh
#
# Optional overrides:
#   SEED=0 ENV_N_ENVS=10 bash scripts/run_can_action_noise_eval.sh
#   NOISE_LEVELS="0 0.08 0.2 0.5" bash scripts/run_can_action_noise_eval.sh

export DPPO_DATA_DIR="${DPPO_DATA_DIR:-$PWD/data}"
export DPPO_LOG_DIR="${DPPO_LOG_DIR:-$PWD/log}"

default_checkpoint_root="$PWD/log/robomimic-finetune/can_ft_d3p_diffusion_mlp_ta4_td20_tdf10"
STATE_H4="${STATE_H4:-$(find "$default_checkpoint_root" -path "*/checkpoint/state_99.pt" 2>/dev/null | sort | tail -1)}"
ADAPTOR_H4="${ADAPTOR_H4:-$(find "$default_checkpoint_root" -path "*/checkpoint/adaptor_99.pt" 2>/dev/null | sort | tail -1)}"

NOISE_LEVELS="${NOISE_LEVELS:-0 0.08 0.2 0.5}"
ENV_N_ENVS="${ENV_N_ENVS:-10}"
SEED="${SEED:-42}"
NOISE_PROB="${NOISE_PROB:-1.0}"
NOISE_START_STEP="${NOISE_START_STEP:-20}"
NOISE_END_STEP="${NOISE_END_STEP:-260}"
OUT_DIR="${OUT_DIR:-outputs/can_action_noise_eval_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -f "$STATE_H4" ]]; then
  echo "Missing STATE_H4 checkpoint: $STATE_H4" >&2
  exit 1
fi

if [[ ! -f "$ADAPTOR_H4" ]]; then
  echo "Missing ADAPTOR_H4 checkpoint: $ADAPTOR_H4" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

run_eval() {
  local method="$1"
  local noise="$2"
  local log_path="$3"

  local -a cmd=(
    python script/run.py
    --config-name="$method"
    --config-dir=cfg/robomimic/eval/can
    base_policy_path="$STATE_H4"
    adaptor_path="$ADAPTOR_H4"
    env.n_envs="$ENV_N_ENVS"
    seed="$SEED"
  )

  if [[ "$noise" != "0" && "$noise" != "0.0" && "$noise" != "0.00" ]]; then
    cmd+=(
      +env.wrappers.action_perturb.noise_std="$noise"
      +env.wrappers.action_perturb.noise_prob="$NOISE_PROB"
      +env.wrappers.action_perturb.start_step="$NOISE_START_STEP"
      +env.wrappers.action_perturb.end_step="$NOISE_END_STEP"
    )
  fi

  printf 'Running %s noise=%s\n' "$method" "$noise"
  "${cmd[@]}" 2>&1 | tee "$log_path"
}

for noise in $NOISE_LEVELS; do
  label="${noise//./p}"

  run_eval \
    eval_d3p_diffusion_mlp \
    "$noise" \
    "$OUT_DIR/noise_${label}_no_td.log"

  run_eval \
    eval_d3p_replan_diffusion_mlp \
    "$noise" \
    "$OUT_DIR/noise_${label}_td.log"
done

summary="$OUT_DIR/summary.txt"
{
  echo "Can H=4 action perturbation summary"
  echo "STATE_H4=$STATE_H4"
  echo "ADAPTOR_H4=$ADAPTOR_H4"
  echo "ENV_N_ENVS=$ENV_N_ENVS SEED=$SEED"
  echo
  for noise in $NOISE_LEVELS; do
    label="${noise//./p}"
    echo "noise=$noise"
    grep "eval:" "$OUT_DIR/noise_${label}_no_td.log" | tail -1 | sed 's/^/  no-TD: /'
    grep "eval:" "$OUT_DIR/noise_${label}_td.log" | tail -1 | sed 's/^/  TD:    /'
    echo
  done
} | tee "$summary"

echo "Logs and summary saved to: $OUT_DIR"
