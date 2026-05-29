# Adaptive Replanning Experiment Log

Date: 2026-05-29
Branch: `codex/m1-action-buffer-fixed-chunk`

This document tracks the implementation and experiment evidence for the Agents.md objective: adaptive online replanning and dynamic denoising for pretrained Diffusion Policy on Robomimic state tasks.

## Final Objective Breakdown

| Milestone | Scope | Status | Evidence |
| --- | --- | --- | --- |
| M1 | `ActionBuffer`, diffusion-policy chunk interface, fixed-chunk controller, one-action-per-step executor, fixed baseline script | Done | Commits `02888c9`, `aefb872`; unit tests; Lift/Can debug smoke |
| M2 | Full baseline logging schema, uncertainty signal interface, TD-error heuristic controller plumbing | Done | `uncertainty.py`, TD-error controller, executor fields, tests, Lift/Can debug smoke |
| M3 | TD critic or frozen value model, TD-error calibration, threshold sweep on Lift | In progress | Transition NPZ, TD critic training, calibration, TD-error replan smoke, debug sweep, 3x80 short sweep, and fixed compute references work; longer eval remains |
| M4 | Replan-only PPO controller with fixed denoising | In progress | M4a-M4h PPO path and M4i 120-step Lift PPO/reference comparison work; longer multi-seed PPO eval remains |
| M5 | Can/Square transfer, Pareto tables/plots, timeline visualizations | In progress | M5a/M5b add plots; M5c adds final eval suite orchestration; M5d adds Can TD artifact/baselines; M5e adds first Can PPO checkpoint/eval; M5f adds 3-seed Lift/Can debug suite; M5g adds Lift 100-episode/seed suite; M5h adds Lift full-horizon suite; M5i adds Can full-horizon suite; M5j adds Square runner support and Lift/Can combined snapshot; M5k adds runtime artifact overrides; M5l adds final readiness report; M5m adds non-empty artifact validation; M5n adds final evidence audit; M5o adds final report generator; M5p adds final evidence pipeline; M5q adds require-complete gate; M5r adds generated-readiness pipeline; Square artifacts/eval remain |
| M6 | Denoise-only and full hierarchical extensions | In progress | M6a adds rule-based denoise-only runtime, tests, Lift debug smoke, and final-suite dry-run support; M6b adds Lift/Can 3-seed short suite; M6c adds multi-suite final audit/report merge support; M6d adds full-horizon Lift/Can denoise-only evidence; M6e adds Square readiness guidance for denoise-only final-suite coverage; M6f adds final_full expected-label preset for completion-gate coverage; M6g tightens default markdown gate to current denoise/preset sections; M6h makes final_full the pipeline default; M6i makes denoise-only a final-eval default baseline; M6j adds generated final-pipeline close-out command; M6k forwards env_meta overrides through generated close-out commands; M6l refreshes default markdown gate through M6k; M6m exposes missing rows/readiness blockers in pipeline summaries; M6n requires M6m in the default markdown gate; M6o exposes final-report blocker details; M6p rejects duplicate suite paths; M6q adds final env summary CSV; M6r completes final matrix metric columns; M6s adds env summary high-level/wall-time winners; full hierarchical remains deferred |

## Implementation Evidence

### M1: Fixed-Chunk Runtime

Implemented a first runnable slice around the existing pretrained diffusion policy:

- `adaptive_diffusion/buffers.py`: fixed-horizon `ActionBuffer` with padded summary.
- `adaptive_diffusion/diffusion_interface.py`: `TorchDiffusionPolicyAdapter` returning chunk, NFE, denoise steps, wall time, sampler.
- `adaptive_diffusion/controllers/fixed.py`: fixed-chunk controller that replans only on empty buffer.
- `adaptive_diffusion/executor.py`: executes exactly one env action per step and writes CSV/JSON metrics.
- `scripts/run_baseline.py`: Robomimic Lift/Can fixed-chunk smoke path using state observations and `multi_step.n_action_steps=1`.

Verification:

```bash
conda run -n d3p python -m pytest -q tests
conda run -n d3p python scripts/run_baseline.py --env robomimic_lift --method fixed_chunk --episodes 2 --debug true --output_root /tmp/d3p_m1_smoke
conda run -n d3p python scripts/run_baseline.py --env robomimic_can --method fixed_chunk --episodes 2 --debug true --output_root /tmp/d3p_m1_smoke
```

### M2: Logging And Uncertainty Plumbing

This milestone extends M1 without changing DPPO training loops:

- Adds `NullUncertaintySignal` for fixed baselines.
- Adds `TDErrorSignal` with a pluggable value function and running z-score normalization over absolute TD error.
- Adds `TDErrorHeuristicController` that replans when decision-time uncertainty crosses a configured z-score or calibrated percentile threshold.
- Extends per-step metrics with the remaining Agents.md fields required for analysis: `obs_norm`, `controller_wall_time_sec`, `uncertainty`, `td_error`, and `hl_reward`.
- Extends summary metrics with median return, high-level return, controller/total wall time, learned replan rate, plan age at replan, discarded actions, and buffer remaining at replan.

Verification so far:

```bash
conda run -n d3p python -m pytest -q tests/test_action_buffer.py tests/test_fixed_controller.py tests/test_executor_smoke.py tests/test_uncertainty_signal.py tests/test_td_error_controller.py
conda run -n d3p python -m pytest -q tests
conda run -n d3p python scripts/run_baseline.py --env robomimic_lift --method fixed_chunk --episodes 2 --debug true --output_root /tmp/d3p_m2_smoke
conda run -n d3p python scripts/run_baseline.py --env robomimic_can --method fixed_chunk --episodes 2 --debug true --output_root /tmp/d3p_m2_smoke
```

Results: `14 passed`; Lift and Can debug smoke both completed.

## Smoke Results

These are debug rollouts with `episodes=2`, `seed=0`, `debug=true`, `chunk_horizon=4`, `denoise_steps=20`, and `max_episode_steps=20`. They verify execution/logging, not policy quality.

| Stage | Env | Output Dir | Success Rate | Mean Return | Mean Length | Mean NFE/Action | Replan Rate | Forced Replan Rate | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| M1.5 | Lift | `/tmp/d3p_m1_smoke/20260528_231447_robomimic_lift_fixed_chunk_seed0` | 0.0 | 0.0 | 20.0 | 5.0 | 0.25 | 0.25 | Initial runnable fixed baseline |
| M1.5 | Can | `/tmp/d3p_m1_smoke/20260528_231520_robomimic_can_fixed_chunk_seed0` | 0.0 | 0.0 | 20.0 | 5.0 | 0.25 | 0.25 | Initial runnable fixed baseline |
| M2 | Lift | `/tmp/d3p_m2_smoke/20260528_233217_robomimic_lift_fixed_chunk_seed0` | 0.0 | 0.0 | 20.0 | 5.0 | 0.25 | 0.25 | Full required step logging present |
| M2 | Can | `/tmp/d3p_m2_smoke/20260528_233311_robomimic_can_fixed_chunk_seed0` | 0.0 | 0.0 | 20.0 | 5.0 | 0.25 | 0.25 | Full required step logging present |

M2 Lift summary additions:

```json
{
  "median_episode_return": 0.0,
  "mean_high_level_return": 0.0,
  "mean_controller_wall_time_per_action": 0.0000014010,
  "mean_total_wall_time_per_action": 0.0022240411,
  "learned_replan_rate": 0.0,
  "mean_plan_age_at_replan": 3.2,
  "mean_discarded_actions_per_episode": 0.0,
  "mean_buffer_remaining_when_replanned": 0.0
}
```

## M3 TD Critic Smoke

M3a adds a concrete data path for TD-error replanning:

1. Fixed-chunk executor can optionally save `transitions.npz` with state observations, next observations, actions, rewards, dones, episode ids, timesteps, success flags, replanning flags, and NFE.
2. `scripts/train_td_critic.py` trains a small semi-gradient TD(0) value model from `transitions.npz`.
3. `TDCriticValueFunction` can load the checkpoint and feed `TDErrorSignal`.

Verification commands:

```bash
conda run -n d3p python -m pytest -q tests
conda run -n d3p python scripts/run_baseline.py --env robomimic_lift --method fixed_chunk --episodes 2 --debug true --save_transitions true --output_root /tmp/d3p_m3_smoke
conda run -n d3p python scripts/train_td_critic.py --dataset /tmp/d3p_m3_smoke/20260528_234345_robomimic_lift_fixed_chunk_seed0/transitions.npz --output_dir /tmp/d3p_m3_smoke/20260528_234345_robomimic_lift_fixed_chunk_seed0/td_critic --epochs 3 --batch_size 16 --hidden_dims 32 --device cpu
```

Smoke output:

| Artifact | Path | Evidence |
| --- | --- | --- |
| Fixed rollout | `/tmp/d3p_m3_smoke/20260528_234345_robomimic_lift_fixed_chunk_seed0` | Lift, 2 debug episodes, 40 transitions |
| Transition data | `/tmp/d3p_m3_smoke/20260528_234345_robomimic_lift_fixed_chunk_seed0/transitions.npz` | `obs=(40, 19)`, `next_obs=(40, 19)`, `actions=(40, 7)`, `rewards=(40,)`, `dones=(40,)` |
| TD critic | `/tmp/d3p_m3_smoke/20260528_234345_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt` | `obs_dim=19`, `num_transitions=40`, `epochs=3`, `hidden_dims=[32]`, `final_loss=0.0008658448` |

End-to-end TD-error check on one collected transition:

```json
{
  "uncertainty": 0.0,
  "td_error": -0.1382266448,
  "raw_abs_td_error": 0.1382266448,
  "running_mean": 0.1382266448,
  "running_std": 0.0,
  "count": 1
}
```

This is still a smoke artifact, not a meaningful critic for final reporting. The next M3 step is calibration and threshold-sweep evaluation with enough fixed-chunk rollouts to produce a nontrivial TD-error distribution.

## M3b TD-Error Replan Smoke

M3b connects the critic artifact to a runnable TD-error heuristic baseline:

1. `scripts/calibrate_td_threshold.py` writes percentile and z-score threshold metadata from `transitions.npz` plus a trained TD critic.
2. `scripts/run_baseline.py --method td_error_replan` loads `TDCriticValueFunction`, feeds `TDErrorSignal`, and uses `TDErrorHeuristicController` for early replanning.
3. Percentile mode reads the calibrated threshold from JSON; z-score mode uses a direct `--td_threshold`.

Calibration command:

```bash
conda run -n d3p python scripts/calibrate_td_threshold.py --dataset /tmp/d3p_m3_smoke/20260528_234345_robomimic_lift_fixed_chunk_seed0/transitions.npz --critic_checkpoint /tmp/d3p_m3_smoke/20260528_234345_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt --output_json /tmp/d3p_m3_smoke/20260528_234345_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json --percentiles 70,80,90,95 --z_thresholds 0.5,1.0,1.5,2.0 --device cpu
```

Calibration summary:

```json
{
  "num_transitions": 40,
  "obs_dim": 19,
  "percentile_thresholds": {
    "70": -0.0940769516,
    "80": 0.3370157421,
    "90": 0.4861529797,
    "95": 1.8641809940
  },
  "mean_abs_td_error": 0.0465902053,
  "std_abs_td_error": 0.0520309545,
  "max_uncertainty": 4.5773558617
}
```

TD-error replanning smoke command:

```bash
conda run -n d3p python scripts/run_baseline.py --env robomimic_lift --method td_error_replan --episodes 2 --debug true --td_critic_checkpoint /tmp/d3p_m3_smoke/20260528_234345_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt --td_threshold_mode percentile --td_threshold_percentile 90 --td_calibration /tmp/d3p_m3_smoke/20260528_234345_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json --output_root /tmp/d3p_m3b_smoke
```

TD-error smoke result:

| Stage | Env | Output Dir | Success Rate | Mean Return | Mean Length | Mean NFE/Action | Replan Rate | Forced Replan Rate | Learned Replan Rate | Discarded Actions/Episode |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M3b | Lift | `/tmp/d3p_m3b_smoke/20260528_235338_robomimic_lift_td_error_replan_seed0` | 0.0 | 0.0 | 20.0 | 6.0 | 0.30 | 0.25 | 0.05 | 2.5 |

Non-forced replan evidence from step metrics:

```text
40 total steps, 2 learned replans
episode 0 t=10 uncertainty=0.765332 td_error=-0.073334 buffer_remaining_before=2 discarded=2
episode 1 t=9  uncertainty=0.498181 td_error=-0.087833 buffer_remaining_before=3 discarded=3
```

Interpretation: the calibrated heuristic path is live. The debug run increases compute from `5.0` to `6.0` NFE/action and produces early replans. This is only a 2-episode smoke using a tiny critic trained on 40 transitions, so it is not evidence of policy improvement.

## M3c TD-Error Percentile Sweep

M3c adds a reusable sweep runner:

- `scripts/run_td_threshold_sweep.py` runs `run_baseline.py --method td_error_replan` for multiple thresholds.
- Outputs: `sweep_config.yaml`, `sweep_results.csv`, `sweep_summary.json`, `sweep_results.md`, plus per-threshold run directories.
- Selection rule in the generated summary is debug-oriented: highest success rate, then highest mean return, then lowest NFE/action.

Verification:

```bash
conda run -n d3p python -m pytest -q tests
conda run -n d3p python scripts/run_td_threshold_sweep.py --env robomimic_lift --threshold_mode percentile --percentiles 70,80,90,95 --critic_checkpoint /tmp/d3p_m3_smoke/20260528_234345_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt --calibration /tmp/d3p_m3_smoke/20260528_234345_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json --episodes 2 --debug true --output_root /tmp/d3p_m3c_sweep
```

Sweep output directory:

```text
/tmp/d3p_m3c_sweep/20260529_000201_robomimic_lift_td_threshold_sweep_seed0
```

Sweep table:

| Label | Calibrated Threshold | Success Rate | Mean Return | NFE/Action | Replan Rate | Forced Replan Rate | Learned Replan Rate | Discarded/Episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p70 | -0.0940769516 | 0.0 | 0.0 | 9.5 | 0.475 | 0.150 | 0.325 | 17.0 |
| p80 | 0.3370157421 | 0.0 | 0.0 | 5.5 | 0.275 | 0.250 | 0.025 | 0.5 |
| p90 | 0.4861529797 | 0.0 | 0.0 | 6.0 | 0.300 | 0.250 | 0.050 | 2.0 |
| p95 | 1.8641809940 | 0.0 | 0.0 | 5.0 | 0.250 | 0.250 | 0.000 | 0.0 |

Generated files checked:

```text
sweep_results.csv
sweep_summary.json
sweep_results.md
```

Interpretation: threshold sensitivity is now measurable. Lower threshold p70 replans aggressively and raises compute to `9.5` NFE/action. Higher threshold p95 behaves like fixed chunk with `5.0` NFE/action and no learned replans. The debug episodes are too short to evaluate task success; the useful result is that the sweep infrastructure and cost/replanning accounting are working.

## M3d Short-Horizon TD Critic And Sweep

M3d uses the same pipeline with a larger short-horizon dataset: `episodes=3`, `max_episode_steps=80`, `seed=0`. This is still not a final eval, but it is long enough to produce 240 transitions and a more informative sweep than the 40-transition debug artifact.

Code change in this milestone:

- `scripts/run_baseline.py` accepts optional `--max_episode_steps` to override debug/full task length.
- `scripts/run_td_threshold_sweep.py` forwards the same override to each run.

Fixed-chunk collection command:

```bash
conda run -n d3p python scripts/run_baseline.py --env robomimic_lift --method fixed_chunk --episodes 3 --max_episode_steps 80 --save_transitions true --output_root /tmp/d3p_m3d_data
```

Fixed-chunk collection output:

```text
/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0
transitions: obs=(240, 19), next_obs=(240, 19), actions=(240, 7), rewards=(240,), dones=(240,)
success_rate=0.0, mean_return=0.0, mean_nfe_per_action=5.0, replan_rate=0.25
```

TD critic training:

```bash
conda run -n d3p python scripts/train_td_critic.py --dataset /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/transitions.npz --output_dir /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic --epochs 10 --batch_size 64 --hidden_dims 64 --device cpu
```

Critic summary:

```json
{
  "num_transitions": 240,
  "obs_dim": 19,
  "epochs": 10,
  "batch_size": 64,
  "hidden_dims": [64],
  "final_loss": 0.0004099693
}
```

Calibration summary:

```json
{
  "percentile_thresholds": {
    "70": -0.2326159522,
    "80": -0.0140111787,
    "90": 0.2756441593,
    "95": 0.8976731062
  },
  "mean_abs_td_error": 0.0151377153,
  "std_abs_td_error": 0.0233882573,
  "max_uncertainty": 5.1254057884
}
```

Sweep command:

```bash
conda run -n d3p python scripts/run_td_threshold_sweep.py --env robomimic_lift --threshold_mode percentile --percentiles 70,80,90,95 --critic_checkpoint /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt --calibration /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json --episodes 3 --max_episode_steps 80 --output_root /tmp/d3p_m3d_sweep
```

Sweep output directory:

```text
/tmp/d3p_m3d_sweep/20260529_001029_robomimic_lift_td_threshold_sweep_seed0
```

Short-horizon sweep table:

| Label | Calibrated Threshold | Success Rate | Mean Return | NFE/Action | Replan Rate | Forced Replan Rate | Learned Replan Rate | Discarded/Episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p70 | -0.2326159522 | 0.000 | 0.000 | 8.083 | 0.404 | 0.175 | 0.229 | 48.000 |
| p80 | -0.0140111787 | 0.000 | 0.000 | 7.250 | 0.363 | 0.196 | 0.167 | 34.000 |
| p90 | 0.2756441593 | 0.000 | 0.000 | 5.667 | 0.283 | 0.237 | 0.046 | 9.000 |
| p95 | 0.8976731062 | 0.333 | 2.333 | 6.083 | 0.304 | 0.229 | 0.075 | 14.667 |

Generated files checked:

```text
sweep_results.csv
sweep_summary.json
sweep_results.md
```

Interpretation: with a 240-transition critic and 3x80-step runs, the sweep produces a nonzero Lift success signal at p95 while fixed chunk collection at the same horizon had zero success. This is encouraging but still not conclusive: it is one seed, only three short episodes, and the critic is trained from fixed-policy rollouts. The main value of M3d is that the short-horizon experimental pipeline can now produce comparable task and cost metrics.

## M3e Fixed Compute References

M3e adds fixed-compute reference methods required for Pareto context:

- `fixed_chunk`: H=4, NFE=20, replan on empty buffer.
- `high_compute`: H=1, NFE=20, replan every environment step.
- `low_compute`: H=4, NFE=4, replan on empty buffer.

Implementation note: local checkpoints are trained with action horizon 4 (`ta4`). `scripts/run_baseline.py` now separates `policy_horizon=4` from executor `chunk_horizon`, so H=1 high-compute runs still load the same checkpoint shape and execute only the first sampled action.

Verification:

```bash
conda run -n d3p python -m pytest -q tests
conda run -n d3p python scripts/run_baseline.py --env robomimic_lift --method fixed_chunk --episodes 3 --max_episode_steps 80 --output_root /tmp/d3p_m3e_references
conda run -n d3p python scripts/run_baseline.py --env robomimic_lift --method high_compute --episodes 3 --max_episode_steps 80 --output_root /tmp/d3p_m3e_references
conda run -n d3p python scripts/run_baseline.py --env robomimic_lift --method low_compute --episodes 3 --max_episode_steps 80 --output_root /tmp/d3p_m3e_references
```

Reference output directories:

```text
/tmp/d3p_m3e_references/20260529_001759_robomimic_lift_fixed_chunk_seed0
/tmp/d3p_m3e_references/20260529_001834_robomimic_lift_high_compute_seed0
/tmp/d3p_m3e_references/20260529_001909_robomimic_lift_low_compute_seed0
```

Reference table, `episodes=3`, `max_episode_steps=80`, `seed=0`:

| Method | H | Denoise Steps | Success Rate | Mean Return | NFE/Action | Replan Rate | Forced Replan Rate | Learned Replan Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low_compute | 4 | 4 | 0.0 | 0.0 | 1.0 | 0.25 | 0.25 | 0.0 |
| fixed_chunk | 4 | 20 | 0.0 | 0.0 | 5.0 | 0.25 | 0.25 | 0.0 |
| high_compute | 1 | 20 | 0.0 | 0.0 | 20.0 | 1.00 | 1.00 | 0.0 |
| td_error_replan p95 | 4 | 20 | 0.333 | 2.333 | 6.083 | 0.304 | 0.229 | 0.075 |

Interpretation: the fixed references span the intended compute range, but none succeeded on this short single-seed Lift run. The M3d `td_error_replan p95` point is above fixed references in this limited run, at modest extra NFE/action over fixed_chunk. This is a useful Pareto sanity check, not a final claim.

## M4a Replan-Only PPO Scaffolding

M4a starts the trainable replanning method without adding a full training loop yet. The goal is to make the PPO-specific data path testable before collecting on-policy rollouts.

Implemented modules:

- `adaptive_diffusion/rl/features.py`: builds high-level state `x_t = [state, chi(B_t), remaining/H, age/H, uncertainty]` and the binary action mask `[continue, replan]`.
- `adaptive_diffusion/rl/masked_categorical.py`: categorical distribution with invalid action logits masked out.
- `adaptive_diffusion/rl/actor_critic.py`: binary replan actor plus scalar value MLP.
- `adaptive_diffusion/rl/advantage.py`: GAE over environment steps.
- `adaptive_diffusion/rl/rollout_buffer.py`: rollout storage for obs features, action/logprob/value, env and high-level rewards, masks, replans, NFE, and discard counts.
- `adaptive_diffusion/controllers/ppo_replan.py`: deterministic/stochastic wrapper that uses the actor-critic and stores `last_decision_info` for later PPO rollout collection.

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_replan_features.py tests/test_masked_actor_critic.py tests/test_ppo_replan_controller.py tests/test_replan_rollout.py
conda run -n d3p python -m pytest -q tests
```

Results:

```text
M4a narrow tests: 13 passed
Full tests: 42 passed
```

Current limitation: this is infrastructure only. It does not yet include `scripts/train_replan_ppo.py`, PPO minibatch updates, checkpointing, or Robomimic on-policy training runs.

## M4b PPO Rollout Collection

M4b connects the PPO controller decision trace to executor step rows, still without adding PPO weight updates. This creates the first testable on-policy data path for replan-only PPO.

Implemented modules:

- `adaptive_diffusion/rl/collector.py`: records one PPO decision plus one executor step row into `ReplanRolloutBuffer`.
- `PPOExecutorRolloutCollector`: runs an executor episode and records each high-level transition immediately after the controller decision is used.
- Package exports for `PPOExecutorRolloutCollector` and `record_ppo_transition`.

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_ppo_rollout_collector.py tests/test_replan_rollout.py tests/test_ppo_replan_controller.py
```

Results:

```text
M4b narrow tests: 9 passed
```

Collector coverage:

- Unit check that controller decision fields and executor row fields populate rollout storage.
- Validation check for mismatched controller action vs executor row.
- Fake executor episode check with PPO controller: action sequence `[replan, continue, continue, continue, replan]`, NFE sequence `[20, 0, 0, 0, 20]`, and GAE-compatible rollout output shape.

Current limitation: this still stops before PPO optimization. The next slice is M4c: minibatch PPO update logic, checkpoint save/load, and a short fake-env training smoke before attempting Robomimic training.

## M4c PPO Update And Checkpoint Core

M4c adds the optimization core needed by `scripts/train_replan_ppo.py`, still without running expensive Robomimic on-policy training. The boundary is intentionally narrow: prove that a stored high-level rollout can drive a clipped PPO update and that the actor-critic can be saved/restored with shape metadata.

Implemented modules:

- `adaptive_diffusion/rl/ppo.py`: `PPOUpdateConfig`, `PPOUpdateStats`, `update_replan_ppo`, checkpoint save/load helpers, and config serialization.
- `ReplanActorCritic` now stores `hidden_dims` and `activation`, so checkpoints can reconstruct the model architecture before loading weights.
- Package exports for the PPO update and checkpoint helpers.

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_ppo_update.py tests/test_masked_actor_critic.py tests/test_ppo_replan_controller.py
```

Results:

```text
M4c narrow tests: 10 passed
```

Test coverage:

- PPO update changes actor-critic parameters and returns finite `loss`, `policy_loss`, `value_loss`, `entropy`, `approx_kl`, and `clip_fraction` stats.
- Empty rollout batches are rejected.
- Checkpoint roundtrip restores architecture metadata and produces identical logits/value outputs.

Current limitation: this does not yet collect and optimize in a single training script. The next slice is M4d: `scripts/train_replan_ppo.py` with fake-env smoke first, then Robomimic Lift training/eval once the loop is proven.

## M4d PPO Training Script Fake Smoke

M4d wires rollout collection and PPO updates into an executable training entry point. The verified path uses `--env fake` so the training loop can be tested quickly before spending Robomimic compute. The script also keeps Robomimic Lift/Can builder paths available for the next training slice.

Implemented script:

- `scripts/train_replan_ppo.py`: creates a PPO replan controller, collects stochastic high-level rollouts, computes GAE, runs clipped PPO updates, saves `latest.pt` and `best.pt`, then runs deterministic evaluation.
- Output layout now includes `config.yaml`, `metrics_step.csv`, `metrics_episode.csv`, `training_updates.csv`, `train_summary.json`, `checkpoints/latest.pt`, `checkpoints/best.pt`, and `eval/eval_summary.json` plus eval CSVs.

Smoke command:

```bash
conda run -n d3p python scripts/train_replan_ppo.py --env fake --seed 0 --total_env_steps 12 --rollout_steps 6 --max_episode_steps 3 --eval_episodes 2 --hidden_dims 8 --minibatch_size 3 --update_epochs 1 --output_root /tmp/d3p_m4d_smoke
```

Smoke output directory:

```text
/tmp/d3p_m4d_smoke/20260529_005213_fake_ppo_replan_only_seed0
```

Smoke summary:

```json
{
  "actual_env_steps": 12,
  "updates": 2,
  "train_success_rate": 1.0,
  "train_mean_nfe_per_action": 11.6666666667,
  "train_replan_rate": 0.5833333333,
  "train_learned_replan_rate": 0.25,
  "eval_success_rate": 1.0,
  "eval_mean_nfe_per_action": 20.0,
  "eval_replan_rate": 1.0,
  "last_update_continue_actions": 3,
  "last_update_replan_actions": 3,
  "last_update_num_updates": 2
}
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_train_replan_ppo_script.py tests/test_ppo_update.py tests/test_ppo_rollout_collector.py
```

Results:

```text
M4d narrow tests: 7 passed
```

Current limitation: fake-env training proves the software path, not Robomimic task learning. The next slice is M4e: run a short Robomimic Lift PPO training smoke with small `total_env_steps`, verify deterministic eval artifacts, and decide whether the cost penalties need an initial sweep.

## M4e Robomimic Lift PPO Training Smoke

M4e runs the PPO training script on the real Robomimic Lift state task with the pretrained diffusion policy. This is still a debug-scale run, but it verifies that the training loop can construct the Robomimic environment, load the local diffusion checkpoint, collect PPO decisions, update the high-level controller, save checkpoints, and run deterministic evaluation.

Smoke command:

```bash
conda run -n d3p python scripts/train_replan_ppo.py --env robomimic_lift --seed 0 --total_env_steps 30 --rollout_steps 6 --max_episode_steps 6 --eval_episodes 2 --hidden_dims 32 --minibatch_size 6 --update_epochs 1 --output_root /tmp/d3p_m4e_lift_smoke
```

Smoke output directory:

```text
/tmp/d3p_m4e_lift_smoke/20260529_005825_robomimic_lift_ppo_replan_only_seed0
```

Generated artifacts checked:

```text
config.yaml
metrics_step.csv
metrics_episode.csv
training_updates.csv
train_summary.json
checkpoints/latest.pt
checkpoints/best.pt
eval/eval_summary.json
eval/eval_metrics_step.csv
eval/eval_metrics_episode.csv
```

Checkpoint load check:

```text
input_dim=54, hidden_dims=(32,), env_steps=30, eval_episodes=2
```

Smoke summary:

| Split | Episodes | Env Steps | Success | Return | HL Return | NFE/Action | Replan Rate | Forced Replan | Learned Replan | Discarded/Episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 5 | 30 | 0.0 | 0.0 | -0.312 | 12.667 | 0.633 | 0.167 | 0.467 | 7.0 |
| eval deterministic | 2 | 12 | 0.0 | 0.0 | -0.375 | 15.000 | 0.750 | 0.167 | 0.583 | 10.5 |

Last PPO update row:

```json
{
  "update_id": 4,
  "env_steps": 30,
  "rollout_steps": 6,
  "continue_actions": 2,
  "replan_actions": 4,
  "forced_replans": 1,
  "mean_hl_reward": -0.0550000034,
  "mean_env_reward": 0.0,
  "mean_nfe_per_action": 13.3333333333,
  "loss": 0.0118501112,
  "policy_loss": -0.0000001987,
  "value_loss": 0.0351603739,
  "entropy": 0.5729876757,
  "approx_kl": -0.0000000099,
  "clip_fraction": 0.0,
  "num_updates": 1
}
```

Interpretation: the true Robomimic training/eval path is live. The short training run sampled both high-level actions in the last update (`continue=2`, `replan=4`) and had 14 non-forced replans across 30 train steps, so the PPO data path is not mask-only. Deterministic eval leaned toward frequent replanning (`replan_rate=0.75`, `NFE/action=15.0`) and had zero success because this is only 30 total train steps with 6-step episodes. The next M4 step should run a small `lambda_C/lambda_D` diagnostic sweep or a longer Lift PPO run to reduce always-replan tendency before claiming learning performance.

## M4f PPO Cost-Penalty Sweep Diagnostic

M4f adds a reusable PPO cost sweep runner and executes a small Lift diagnostic. The goal is to stop hand-running individual PPO costs and start collecting comparable high-level cost/action-collapse evidence for `lambda_C` and `lambda_D`.

Implemented script:

- `scripts/run_ppo_cost_sweep.py`: runs `scripts/train_replan_ppo.py` over a grid of `lambda_C` and `lambda_D`, aggregates `train_summary.json`, writes `sweep_config.yaml`, `sweep_results.csv`, `sweep_summary.json`, and `sweep_results.md`.
- Selection rule for debug best row: eval success, eval return, eval high-level return, lower NFE/action, then lower replan rate.

Fake sweep verification command:

```bash
conda run -n d3p python scripts/run_ppo_cost_sweep.py --env fake --lambda_C_values 0.0,0.01 --lambda_D_values 0.0 --seed 0 --total_env_steps 12 --rollout_steps 6 --max_episode_steps 3 --eval_episodes 1 --hidden_dims 8 --minibatch_size 3 --update_epochs 1 --device cpu --output_root /tmp/d3p_m4f_fake_sweep
```

Fake sweep output directory:

```text
/tmp/d3p_m4f_fake_sweep/20260529_010605_fake_ppo_cost_sweep_seed0
```

Lift diagnostic command:

```bash
conda run -n d3p python scripts/run_ppo_cost_sweep.py --env robomimic_lift --lambda_C_values 0.0,0.003,0.01 --lambda_D_values 0.03 --seed 0 --total_env_steps 30 --rollout_steps 6 --max_episode_steps 6 --eval_episodes 2 --hidden_dims 32 --minibatch_size 6 --update_epochs 1 --output_root /tmp/d3p_m4f_lift_cost_sweep
```

Lift sweep output directory:

```text
/tmp/d3p_m4f_lift_cost_sweep/20260529_010622_robomimic_lift_ppo_cost_sweep_seed0
```

Lift diagnostic table, `seed=0`, `train_steps=30`, `max_episode_steps=6`, `eval_episodes=2`, `lambda_D=0.03`:

| Label | lambda_C | Eval Success | Eval Return | Eval HL Return | Eval NFE/Action | Eval Replan | Train Replan | Last Continue | Last Replan |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c0_d0p03 | 0.000 | 0.0 | 0.0 | -0.120 | 16.667 | 0.833 | 0.633 | 2 | 4 |
| c0p003_d0p03 | 0.003 | 0.0 | 0.0 | -0.375 | 15.000 | 0.750 | 0.633 | 2 | 4 |
| c0p01_d0p03 | 0.010 | 0.0 | 0.0 | -1.005 | 15.000 | 0.750 | 0.633 | 2 | 4 |

Generated files checked:

```text
sweep_config.yaml
sweep_results.csv
sweep_summary.json
sweep_results.md
per-run config.yaml, metrics_step.csv, metrics_episode.csv, training_updates.csv, train_summary.json
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_ppo_cost_sweep.py tests/test_train_replan_ppo_script.py
```

Results:

```text
M4f narrow tests: 7 passed
```

Interpretation: the cost-sweep infrastructure is working and produces comparable train/eval rows. In this very short Lift run, increasing `lambda_C` from `0.0` to `0.003` or `0.01` lowered deterministic eval replanning from `0.833` to `0.750` and NFE/action from `16.667` to `15.0`, but did not change the last-update action mix (`continue=2`, `replan=4`) or train replan rate (`0.633`). Higher `lambda_C` mostly makes high-level return more negative at this horizon. This suggests the next useful M4 step is a longer run or a broader `lambda_D` sweep, not claiming performance from the 30-step diagnostic.

## M4g Standalone PPO Checkpoint Evaluation

M4g adds the standalone checkpoint evaluation entry point required by Agents.md. This separates training from deterministic evaluation so later long PPO runs and cost sweeps can be re-evaluated without rerunning training.

Implemented script:

- `scripts/eval_policy.py`: loads a saved `ppo_replan_only` actor-critic checkpoint, reconstructs feature dimensions and reward/cost settings from checkpoint metadata, builds the fake or Robomimic environment, and writes eval metrics.
- Output files: `config.yaml`, `metrics_step.csv`, `metrics_episode.csv`, `eval_metrics_step.csv`, `eval_metrics_episode.csv`, and `eval_summary.json`.

Fake checkpoint verification:

```bash
conda run -n d3p python -m pytest -q tests/test_eval_policy_script.py tests/test_train_replan_ppo_script.py
```

Results:

```text
M4g narrow tests: 2 passed
```

Lift checkpoint eval command:

```bash
conda run -n d3p python scripts/eval_policy.py --checkpoint /tmp/d3p_m4f_lift_cost_sweep/20260529_010622_robomimic_lift_ppo_cost_sweep_seed0/runs/20260529_010646_robomimic_lift_ppo_replan_only_seed0/checkpoints/best.pt --episodes 2 --seed 0 --deterministic true --output_root /tmp/d3p_m4g_eval_policy
```

Lift eval output directory:

```text
/tmp/d3p_m4g_eval_policy/20260529_011442_robomimic_lift_ppo_replan_only_eval_seed0
```

Lift standalone eval summary:

```json
{
  "episodes": 2,
  "success_rate": 0.0,
  "mean_episode_return": 0.0,
  "mean_high_level_return": -0.33,
  "mean_nfe_per_action": 13.3333333333,
  "replan_rate": 0.6666666667,
  "forced_replan_rate": 0.1666666667,
  "learned_replan_rate": 0.5,
  "step_rows": 12,
  "non_forced_replans": 6
}
```

Interpretation: PPO checkpoints can now be evaluated independently from the training script. The M4f checkpoint selected here still has zero Lift success and frequent learned replanning at this short horizon, but the evaluation artifacts are reusable and comparable with future long runs and fixed/TD-error baselines.

## M4h Lift PPO Lambda-D Diagnostic

M4h broadens the PPO cost diagnostic over the discard penalty `lambda_D` while keeping `lambda_C=0.003`. This directly tests whether penalizing interrupted non-empty buffers reduces frequent learned replanning more than the inference-cost penalty alone.

Command:

```bash
conda run -n d3p python scripts/run_ppo_cost_sweep.py --env robomimic_lift --lambda_C_values 0.003 --lambda_D_values 0.0,0.01,0.03,0.1 --seed 0 --total_env_steps 30 --rollout_steps 6 --max_episode_steps 6 --eval_episodes 2 --hidden_dims 32 --minibatch_size 6 --update_epochs 1 --output_root /tmp/d3p_m4h_lift_lambda_d_sweep
```

Output directory:

```text
/tmp/d3p_m4h_lift_lambda_d_sweep/20260529_011900_robomimic_lift_ppo_cost_sweep_seed0
```

Generated files checked:

```text
sweep_config.yaml
sweep_results.csv
sweep_summary.json
sweep_results.md
```

Lift lambda-D diagnostic table, `seed=0`, `train_steps=30`, `max_episode_steps=6`, `eval_episodes=2`, `lambda_C=0.003`:

| Label | lambda_D | Eval Success | Eval Return | Eval HL Return | Eval NFE/Action | Eval Replan | Eval Discarded/Episode | Train Replan | Last Continue | Last Replan |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c0p003_d0 | 0.00 | 0.0 | 0.0 | -0.300 | 16.667 | 0.833 | 12.0 | 0.633 | 2 | 4 |
| c0p003_d0p01 | 0.01 | 0.0 | 0.0 | -0.340 | 16.667 | 0.833 | 12.0 | 0.633 | 2 | 4 |
| c0p003_d0p03 | 0.03 | 0.0 | 0.0 | -0.375 | 15.000 | 0.750 | 10.5 | 0.633 | 2 | 4 |
| c0p003_d0p1 | 0.10 | 0.0 | 0.0 | -0.620 | 15.000 | 0.750 | 10.5 | 0.633 | 2 | 4 |

Interpretation: in this short diagnostic, increasing `lambda_D` from `0.0` or `0.01` to `0.03` or `0.1` lowers deterministic eval replan rate from `0.833` to `0.750`, lowers NFE/action from `16.667` to `15.0`, and lowers discarded actions per eval episode from `12.0` to `10.5`. It still does not change train replan rate (`0.633`) or the final update action mix (`continue=2`, `replan=4`). The best debug row by the sweep's selection rule is still `lambda_D=0.0` because all rows have zero raw task return and the less penalized row has the least negative high-level return. This is useful negative evidence: short PPO training responds weakly to discard penalties, so the next useful step is either longer Lift PPO training or changing the rollout/update scale before expecting stable cost-sensitive behavior.

## M4i Longer Lift PPO Diagnostic And Same-Horizon References

M4i increases PPO training from the 30-step diagnostics to a 120-step Lift run and evaluates it against fixed compute references at the same short evaluation horizon. This is still far below final training scale, but it starts to answer whether the PPO controller can move away from the frequent-replan behavior observed in M4e-M4h.

PPO training command:

```bash
conda run -n d3p python scripts/train_replan_ppo.py --env robomimic_lift --seed 0 --total_env_steps 120 --rollout_steps 20 --max_episode_steps 20 --eval_episodes 3 --hidden_dims 64 --minibatch_size 20 --update_epochs 2 --lambda_C 0.003 --lambda_D 0.03 --output_root /tmp/d3p_m4i_lift_ppo_longer
```

PPO training output:

```text
/tmp/d3p_m4i_lift_ppo_longer/20260529_012409_robomimic_lift_ppo_replan_only_seed0
```

Standalone checkpoint eval command:

```bash
conda run -n d3p python scripts/eval_policy.py --checkpoint /tmp/d3p_m4i_lift_ppo_longer/20260529_012409_robomimic_lift_ppo_replan_only_seed0/checkpoints/best.pt --episodes 3 --seed 0 --deterministic true --output_root /tmp/d3p_m4i_lift_ppo_eval
```

Standalone eval output:

```text
/tmp/d3p_m4i_lift_ppo_eval/20260529_012452_robomimic_lift_ppo_replan_only_eval_seed0
```

Fixed/TD reference commands used `episodes=3` and `max_episode_steps=20` under `/tmp/d3p_m4i_references`. The TD-error reference used the M3d critic and p95 percentile calibration.

Reference output directories:

```text
/tmp/d3p_m4i_references/20260529_012558_robomimic_lift_low_compute_seed0
/tmp/d3p_m4i_references/20260529_012527_robomimic_lift_fixed_chunk_seed0
/tmp/d3p_m4i_references/20260529_012629_robomimic_lift_high_compute_seed0
/tmp/d3p_m4i_references/20260529_012708_robomimic_lift_td_error_replan_seed0
```

Same-horizon comparison, `seed=0`, `episodes=3`, `max_episode_steps=20`:

| Method | Success | Return | HL Return | NFE/Action | Replan Rate | Forced Replan | Learned Replan | Discarded/Episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low_compute | 0.0 | 0.0 | 0.000 | 1.000 | 0.250 | 0.250 | 0.000 | 0.0 |
| fixed_chunk | 0.0 | 0.0 | 0.000 | 5.000 | 0.250 | 0.250 | 0.000 | 0.0 |
| td_error_replan p95 | 0.0 | 0.0 | 0.000 | 5.000 | 0.250 | 0.250 | 0.000 | 0.0 |
| ppo_replan internal eval | 0.0 | 0.0 | -0.520 | 7.000 | 0.350 | 0.183 | 0.167 | 6.3 |
| ppo_replan eval_policy | 0.0 | 0.0 | -0.580 | 7.333 | 0.367 | 0.133 | 0.233 | 7.7 |
| high_compute | 0.0 | 0.0 | 0.000 | 20.000 | 1.000 | 1.000 | 0.000 | 0.0 |

Last PPO training update:

```text
update_id=5, env_steps=120, rollout_steps=20, continue_actions=13, replan_actions=7, forced_replans=3, mean_hl_reward=-0.0270, mean_nfe_per_action=7.0, entropy=0.5880
```

Interpretation: compared with M4h, the longer PPO run reduces deterministic eval replanning substantially: from M4h's `0.75` replan rate and `15.0` NFE/action down to about `0.35-0.37` replan rate and `7.0-7.33` NFE/action. It still uses more compute than fixed_chunk (`5.0` NFE/action) and all methods have zero short-horizon raw task return, so this is not performance evidence. It is evidence that the PPO controller can learn a less aggressive replanning policy when trained longer, and that the current next step should compare longer PPO runs against fixed/TD references on longer horizons where task success can appear.

## M4j 80-Step Lift PPO Diagnostic

M4j moves the PPO diagnostic to the same 80-step horizon used by the M3d/M3e Lift references. The goal is not final performance, but to check whether replan-only PPO remains compute-sensitive when task success can start to appear.

PPO training command:

```bash
conda run -n d3p python scripts/train_replan_ppo.py --env robomimic_lift --seed 0 --total_env_steps 240 --rollout_steps 80 --max_episode_steps 80 --eval_episodes 3 --hidden_dims 64 --minibatch_size 40 --update_epochs 2 --lambda_C 0.003 --lambda_D 0.03 --output_root /tmp/d3p_m4j_lift_ppo_h80
```

PPO training output:

```text
/tmp/d3p_m4j_lift_ppo_h80/20260529_013331_robomimic_lift_ppo_replan_only_seed0
```

Standalone checkpoint eval command:

```bash
conda run -n d3p python scripts/eval_policy.py --checkpoint /tmp/d3p_m4j_lift_ppo_h80/20260529_013331_robomimic_lift_ppo_replan_only_seed0/checkpoints/best.pt --episodes 3 --seed 0 --deterministic true --output_root /tmp/d3p_m4j_lift_ppo_eval
```

Standalone eval output:

```text
/tmp/d3p_m4j_lift_ppo_eval/20260529_013406_robomimic_lift_ppo_replan_only_eval_seed0
```

The fixed references are the M3e 3-episode 80-step Lift runs. The TD-error reference is the M3d p95 threshold row, selected because it was the best M3d debug row by raw success/return.

Same-horizon comparison, `seed=0`, `episodes=3`, `max_episode_steps=80`:

| Method | Success | Return | HL Return | NFE/Action | Replan Rate | Forced Replan | Learned Replan | Discarded/Episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low_compute | 0.000 | 0.000 | 0.000 | 1.000 | 0.250 | 0.250 | 0.000 | 0.0 |
| fixed_chunk | 0.000 | 0.000 | 0.000 | 5.000 | 0.250 | 0.250 | 0.000 | 0.0 |
| td_error_replan p95 | 0.333 | 2.333 | n/a | 6.083 | 0.304 | 0.229 | 0.075 | 14.7 |
| ppo_replan internal eval | 0.000 | 0.000 | -1.820 | 6.583 | 0.329 | 0.229 | 0.100 | 23.7 |
| ppo_replan eval_policy | 0.333 | 1.333 | -0.597 | 6.833 | 0.342 | 0.221 | 0.121 | 28.3 |
| high_compute | 0.000 | 0.000 | 0.000 | 20.000 | 1.000 | 1.000 | 0.000 | 0.0 |

PPO training summary:

```text
actual_env_steps=240, updates=3, train_episodes=3, train_success=0.0, train_return=0.0, train_hl_return=-3.830, train_nfe_per_action=11.083, train_replan_rate=0.554
```

Last PPO update:

```text
update_id=2, env_steps=240, rollout_steps=80, continue_actions=36, replan_actions=44, forced_replans=6, mean_hl_reward=-0.0472, mean_nfe_per_action=11.0, entropy=0.6358
```

Interpretation: the 80-step PPO run still learns a much lower replan rate than the early 6-step diagnostics, but it is not yet better than TD-error or fixed_chunk on compute. The standalone checkpoint eval produced `1/3` Lift success with `6.833` NFE/action and `0.342` replan rate, while TD-error p95 produced `1/3` success with lower compute (`6.083` NFE/action, `0.304` replan rate). The internal eval and standalone eval disagree on raw success (`0/3` vs `1/3`), so this is high-variance debug evidence only. The next useful PPO step is a small multi-seed or longer-training comparison, not a larger method surface.

## M4k Longer 80-Step Lift PPO Diagnostic

M4k keeps the M4j 80-step horizon but increases PPO training from `240` to `720` environment steps. This checks whether the M4j PPO policy was mostly limited by very short training.

PPO training command:

```bash
conda run -n d3p python scripts/train_replan_ppo.py --env robomimic_lift --seed 0 --total_env_steps 720 --rollout_steps 80 --max_episode_steps 80 --eval_episodes 3 --hidden_dims 64 --minibatch_size 40 --update_epochs 2 --lambda_C 0.003 --lambda_D 0.03 --output_root /tmp/d3p_m4k_lift_ppo_h80_longer
```

PPO training output:

```text
/tmp/d3p_m4k_lift_ppo_h80_longer/20260529_013936_robomimic_lift_ppo_replan_only_seed0
```

Standalone checkpoint eval command:

```bash
conda run -n d3p python scripts/eval_policy.py --checkpoint /tmp/d3p_m4k_lift_ppo_h80_longer/20260529_013936_robomimic_lift_ppo_replan_only_seed0/checkpoints/best.pt --episodes 3 --seed 0 --deterministic true --output_root /tmp/d3p_m4k_lift_ppo_eval
```

Standalone eval output:

```text
/tmp/d3p_m4k_lift_ppo_eval/20260529_014023_robomimic_lift_ppo_replan_only_eval_seed0
```

Comparison against the M4j PPO run and the 80-step references, `seed=0`, `episodes=3`, `max_episode_steps=80`:

| Method | Success | Return | HL Return | NFE/Action | Replan Rate | Forced Replan | Learned Replan | Discarded/Episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_chunk | 0.000 | 0.000 | 0.000 | 5.000 | 0.250 | 0.250 | 0.000 | 0.0 |
| td_error_replan p95 | 0.333 | 2.333 | n/a | 6.083 | 0.304 | 0.229 | 0.075 | 14.7 |
| M4j ppo_replan eval_policy, 240 steps | 0.333 | 1.333 | -0.597 | 6.833 | 0.342 | 0.221 | 0.121 | 28.3 |
| M4k ppo_replan internal eval, 720 steps | 0.333 | 0.667 | -0.563 | 5.083 | 0.254 | 0.250 | 0.004 | 0.7 |
| M4k ppo_replan eval_policy, 720 steps | 0.000 | 0.000 | -1.200 | 5.000 | 0.250 | 0.250 | 0.000 | 0.0 |
| high_compute | 0.000 | 0.000 | 0.000 | 20.000 | 1.000 | 1.000 | 0.000 | 0.0 |

PPO training summary:

```text
actual_env_steps=720, updates=9, train_episodes=9, train_success=0.0, train_return=0.0, train_hl_return=-3.283, train_nfe_per_action=9.778, train_replan_rate=0.489
```

Last PPO update:

```text
update_id=8, env_steps=720, rollout_steps=80, continue_actions=50, replan_actions=30, forced_replans=12, mean_hl_reward=-0.0293, mean_nfe_per_action=7.5, entropy=0.5275
```

Interpretation: longer PPO training clearly reduces compute. Internal eval reaches almost fixed_chunk compute (`5.083` NFE/action, `0.254` replan rate) with `1/3` success, and standalone eval exactly matches fixed_chunk replan accounting (`5.0` NFE/action, `0.25` replan rate). However, standalone success drops to `0/3`. The useful conclusion is that the current PPO reward can learn to avoid extra replans, but the task-success signal is too high-variance at 3 eval episodes to claim an improvement. The next comparison should increase eval episodes and/or seeds before expanding to Can/Square.

## M4l 10-Episode 80-Step Lift Evaluation

M4l increases the 80-step Lift evaluation from 3 to 10 episodes on seed 0. This is still not final evaluation scale, but it reduces the obvious 3-episode variance seen in M4j/M4k and gives a better local comparison before changing PPO reward or rollout settings.

Commands:

```bash
conda run -n d3p python scripts/run_baseline.py --env robomimic_lift --method fixed_chunk --episodes 10 --seed 0 --max_episode_steps 80 --chunk_horizon 4 --denoise_steps 20 --output_root /tmp/d3p_m4l_lift_eval10
conda run -n d3p python scripts/run_baseline.py --env robomimic_lift --method td_error_replan --episodes 10 --seed 0 --max_episode_steps 80 --chunk_horizon 4 --denoise_steps 20 --td_critic_checkpoint /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt --td_calibration /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json --td_threshold_percentile 95 --output_root /tmp/d3p_m4l_lift_eval10
conda run -n d3p python scripts/eval_policy.py --checkpoint /tmp/d3p_m4k_lift_ppo_h80_longer/20260529_013936_robomimic_lift_ppo_replan_only_seed0/checkpoints/best.pt --episodes 10 --seed 0 --deterministic true --max_episode_steps 80 --output_root /tmp/d3p_m4l_lift_eval10
conda run -n d3p python scripts/eval_policy.py --checkpoint /tmp/d3p_m4j_lift_ppo_h80/20260529_013331_robomimic_lift_ppo_replan_only_seed0/checkpoints/best.pt --episodes 10 --seed 0 --deterministic true --max_episode_steps 80 --output_root /tmp/d3p_m4l_lift_eval10
conda run -n d3p python scripts/run_baseline.py --env robomimic_lift --method low_compute --episodes 10 --seed 0 --max_episode_steps 80 --chunk_horizon 4 --denoise_steps 4 --output_root /tmp/d3p_m4l_lift_eval10
conda run -n d3p python scripts/run_baseline.py --env robomimic_lift --method high_compute --episodes 10 --seed 0 --max_episode_steps 80 --chunk_horizon 1 --denoise_steps 20 --output_root /tmp/d3p_m4l_lift_eval10
```

Output directories:

```text
/tmp/d3p_m4l_lift_eval10/20260529_014340_robomimic_lift_fixed_chunk_seed0
/tmp/d3p_m4l_lift_eval10/20260529_014417_robomimic_lift_td_error_replan_seed0
/tmp/d3p_m4l_lift_eval10/20260529_014453_robomimic_lift_ppo_replan_only_eval_seed0  # M4k checkpoint
/tmp/d3p_m4l_lift_eval10/20260529_014533_robomimic_lift_ppo_replan_only_eval_seed0  # M4j checkpoint
/tmp/d3p_m4l_lift_eval10/20260529_014617_robomimic_lift_low_compute_seed0
/tmp/d3p_m4l_lift_eval10/20260529_014650_robomimic_lift_high_compute_seed0
```

Same-horizon comparison, `seed=0`, `episodes=10`, `max_episode_steps=80`:

| Method | Success | Return | HL Return | NFE/Action | Replan Rate | Forced Replan | Learned Replan | Discarded/Episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low_compute | 0.000 | 0.000 | 0.000 | 1.000 | 0.250 | 0.250 | 0.000 | 0.0 |
| fixed_chunk | 0.100 | 1.500 | 1.500 | 5.000 | 0.250 | 0.250 | 0.000 | 0.0 |
| td_error_replan p95 | 0.300 | 3.900 | 3.900 | 6.025 | 0.301 | 0.231 | 0.070 | 15.0 |
| M4k ppo_replan eval_policy, 720 train steps | 0.000 | 0.000 | -1.209 | 5.025 | 0.251 | 0.250 | 0.001 | 0.2 |
| M4j ppo_replan eval_policy, 240 train steps | 0.000 | 0.000 | -1.902 | 6.775 | 0.339 | 0.224 | 0.115 | 27.2 |
| high_compute | 0.000 | 0.000 | 0.000 | 20.000 | 1.000 | 1.000 | 0.000 | 0.0 |

Interpretation: the larger single-seed Lift eval changes the earlier 3-episode picture. Fixed_chunk reaches `1/10` success, TD-error p95 reaches `3/10` success with modest extra compute, and both PPO checkpoints reach `0/10`. The 720-step PPO checkpoint mostly learns to match fixed compute, while the 240-step checkpoint spends more compute without improving task success. This points to a PPO reward/signal issue: in the short training runs the controller often sees no successful episodes, so the cost terms are much denser than the sparse task reward and the learned policy drifts toward avoiding replans. TD-error remains the strongest current debug method on Lift.

## M4m PPO TD-Uncertainty Bonus Diagnostic

M4m adds an optional PPO reward shaping term, `uncertainty_replan_bonus`, and fixes `scripts/eval_policy.py` so standalone checkpoint eval can recover `td_critic_checkpoint` from checkpoint metadata. The default bonus is `0.0`, so existing baselines and previous PPO runs are unchanged unless the flag is set.

Implementation details:

```text
hl_reward = env_reward + uncertainty_replan_bonus * max(0, uncertainty) * I(replanned) - lambda_C * replan_cost - lambda_D * I(replanned and discarded)
```

The intended use is PPO training with a TD critic available as the uncertainty signal. This gives PPO a dense positive signal for replanning in high TD-error states while preserving the compute/discard penalties.

Fake smoke command:

```bash
conda run -n d3p python scripts/train_replan_ppo.py --env fake --seed 0 --total_env_steps 12 --rollout_steps 6 --max_episode_steps 6 --eval_episodes 1 --hidden_dims 16 --minibatch_size 6 --update_epochs 1 --uncertainty_replan_bonus 0.1 --output_root /tmp/d3p_m4m_fake_bonus_smoke
```

Fake smoke output:

```text
/tmp/d3p_m4m_fake_bonus_smoke/20260529_015453_fake_ppo_replan_only_seed0
```

Lift PPO TD-bonus training command:

```bash
conda run -n d3p python scripts/train_replan_ppo.py --env robomimic_lift --seed 0 --total_env_steps 720 --rollout_steps 80 --max_episode_steps 80 --eval_episodes 3 --hidden_dims 64 --minibatch_size 40 --update_epochs 2 --lambda_C 0.003 --lambda_D 0.03 --td_critic_checkpoint /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt --uncertainty_replan_bonus 0.1 --output_root /tmp/d3p_m4m_lift_ppo_td_bonus
```

Lift PPO TD-bonus output:

```text
/tmp/d3p_m4m_lift_ppo_td_bonus/20260529_015511_robomimic_lift_ppo_replan_only_seed0
```

Standalone checkpoint eval command, intentionally without `--td_critic_checkpoint` to verify checkpoint fallback:

```bash
conda run -n d3p python scripts/eval_policy.py --checkpoint /tmp/d3p_m4m_lift_ppo_td_bonus/20260529_015511_robomimic_lift_ppo_replan_only_seed0/checkpoints/best.pt --episodes 10 --seed 0 --deterministic true --max_episode_steps 80 --output_root /tmp/d3p_m4m_lift_ppo_td_bonus_eval
```

Standalone eval output:

```text
/tmp/d3p_m4m_lift_ppo_td_bonus_eval/20260529_015552_robomimic_lift_ppo_replan_only_eval_seed0
```

Checkpoint fallback verification from standalone eval config:

```text
uncertainty_replan_bonus: 0.1
td_critic_checkpoint: /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt
```

M4m comparison against the previous 10-episode M4l rows, `seed=0`, `episodes=10`, `max_episode_steps=80`:

| Method | Success | Return | HL Return | NFE/Action | Replan Rate | Forced Replan | Learned Replan | Discarded/Episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_chunk | 0.100 | 1.500 | 1.500 | 5.000 | 0.250 | 0.250 | 0.000 | 0.0 |
| td_error_replan p95 | 0.300 | 3.900 | 3.900 | 6.025 | 0.301 | 0.231 | 0.070 | 15.0 |
| M4k ppo_replan, no TD bonus | 0.000 | 0.000 | -1.209 | 5.025 | 0.251 | 0.250 | 0.001 | 0.2 |
| M4m ppo_replan, TD bonus 0.1 | 0.000 | 0.000 | -1.142 | 5.675 | 0.284 | 0.236 | 0.048 | 10.0 |
| M4j ppo_replan, shorter train | 0.000 | 0.000 | -1.902 | 6.775 | 0.339 | 0.224 | 0.115 | 27.2 |

M4m training summary:

```text
actual_env_steps=720, updates=9, train_success=0.0, train_return=0.0, train_hl_return=-3.090, train_nfe_per_action=10.139, train_replan_rate=0.507, learned_replan_rate=0.415
```

M4m standalone eval summary:

```text
success=0.0, return=0.0, hl_return=-1.142, nfe_per_action=5.675, replan_rate=0.284, learned_replan_rate=0.048, discarded_actions_per_episode=10.0
```

Interpretation: the TD-uncertainty bonus changes PPO behavior slightly in eval, increasing learned replans from M4k near-zero rate to `0.048`, but it still gets `0/10` success and remains below the TD-error heuristic. This means the implementation is useful infrastructure, but `bonus=0.1` with 720 training steps is not enough to solve the sparse reward problem. The next PPO step should either sweep the uncertainty bonus/cost scale or try a supervised/behavior-cloning warm start from TD-error decisions before PPO fine-tuning.

## M4n PPO Uncertainty-Bonus Sweep Support And Lift Diagnostic

M4n extends `scripts/run_ppo_cost_sweep.py` so PPO sweeps can vary `uncertainty_replan_bonus` and pass `td_critic_checkpoint`/`td_gamma` through to `scripts/train_replan_ppo.py`. Labels stay backward-compatible: bonus `0.0` keeps the old `c..._d...` label, while nonzero bonus adds `_b...`.

Fake sweep verification command:

```bash
conda run -n d3p python scripts/run_ppo_cost_sweep.py --env fake --lambda_C_values 0.003 --lambda_D_values 0.03 --uncertainty_replan_bonus_values 0.0,0.1 --seed 0 --total_env_steps 12 --rollout_steps 6 --max_episode_steps 3 --eval_episodes 1 --hidden_dims 8 --minibatch_size 3 --update_epochs 1 --device cpu --output_root /tmp/d3p_m4n_fake_bonus_sweep
```

Fake sweep output:

```text
/tmp/d3p_m4n_fake_bonus_sweep/20260529_020317_fake_ppo_cost_sweep_seed0
```

Fake sweep verification result:

```text
num_runs=2, labels=[c0p003_d0p03, c0p003_d0p03_b0p1], artifacts={sweep_config.yaml, sweep_results.csv, sweep_summary.json, sweep_results.md}
```

Lift bonus sweep command:

```bash
conda run -n d3p python scripts/run_ppo_cost_sweep.py --env robomimic_lift --lambda_C_values 0.003 --lambda_D_values 0.03 --uncertainty_replan_bonus_values 0.0,0.1,0.3 --seed 0 --total_env_steps 720 --rollout_steps 80 --max_episode_steps 80 --eval_episodes 3 --hidden_dims 64 --minibatch_size 40 --update_epochs 2 --td_critic_checkpoint /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt --output_root /tmp/d3p_m4n_lift_bonus_sweep
```

Lift bonus sweep output:

```text
/tmp/d3p_m4n_lift_bonus_sweep/20260529_020337_robomimic_lift_ppo_cost_sweep_seed0
```

Internal 3-episode Lift sweep, `seed=0`, `total_env_steps=720`, `max_episode_steps=80`:

| Label | Bonus | Eval Success | Eval Return | Eval HL Return | Eval NFE/Action | Eval Replan | Eval Learned Replan | Eval Discarded/Episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c0p003_d0p03 | 0.0 | 0.000 | 0.000 | -1.260 | 5.167 | 0.258 | 0.008 | 1.7 |
| c0p003_d0p03_b0p1 | 0.1 | 0.000 | 0.000 | -1.039 | 5.000 | 0.250 | 0.000 | 0.0 |
| c0p003_d0p03_b0p3 | 0.3 | 0.000 | 0.000 | 0.666 | 6.833 | 0.342 | 0.125 | 27.3 |

Best internal row by sweep rule:

```text
c0p003_d0p03_b0p3
```

Standalone 10-episode eval command for the best checkpoint:

```bash
conda run -n d3p python scripts/eval_policy.py --checkpoint /tmp/d3p_m4n_lift_bonus_sweep/20260529_020337_robomimic_lift_ppo_cost_sweep_seed0/runs/20260529_020435_robomimic_lift_ppo_replan_only_seed0/checkpoints/best.pt --episodes 10 --seed 0 --deterministic true --max_episode_steps 80 --output_root /tmp/d3p_m4n_lift_bonus_sweep_eval
```

Standalone eval output:

```text
/tmp/d3p_m4n_lift_bonus_sweep_eval/20260529_020515_robomimic_lift_ppo_replan_only_eval_seed0
```

10-episode comparison against M4l references, `seed=0`, `max_episode_steps=80`:

| Method | Success | Return | HL Return | NFE/Action | Replan Rate | Forced Replan | Learned Replan | Discarded/Episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_chunk | 0.100 | 1.500 | 1.500 | 5.000 | 0.250 | 0.250 | 0.000 | 0.0 |
| td_error_replan p95 | 0.300 | 3.900 | 3.900 | 6.025 | 0.301 | 0.231 | 0.070 | 15.0 |
| ppo_replan TD bonus 0.3 | 0.300 | 3.200 | 6.654 | 6.475 | 0.324 | 0.223 | 0.101 | 22.5 |
| high_compute | 0.000 | 0.000 | 0.000 | 20.000 | 1.000 | 1.000 | 0.000 | 0.0 |

Interpretation: the uncertainty bonus sweep is the first PPO result that matches TD-error p95 on Lift debug success (`3/10`). It still has lower task return (`3.2` vs `3.9`) and higher compute (`6.475` vs `6.025` NFE/action), but it is no longer a failed PPO setting. The high-level return is positive because it includes the TD-uncertainty bonus, so raw task return remains the primary task metric. Next PPO work should refine the bonus scale near `0.3` and validate across more seeds before moving to Can/Square.

## M4o 3-Seed Lift Debug Evaluation For PPO Bonus 0.3

M4o checks whether the M4n seed-0 PPO result with `uncertainty_replan_bonus=0.3` survives a small 3-seed debug evaluation. This is still not the final 100-episode evaluation, but it is the first multi-seed check for the current PPO setting.

Seed 0 references reused from M4l/M4n:

```text
fixed_chunk: /tmp/d3p_m4l_lift_eval10/20260529_014340_robomimic_lift_fixed_chunk_seed0
td_error_p95: /tmp/d3p_m4l_lift_eval10/20260529_014417_robomimic_lift_td_error_replan_seed0
ppo_bonus0p3: /tmp/d3p_m4n_lift_bonus_sweep_eval/20260529_020515_robomimic_lift_ppo_replan_only_eval_seed0
```

Seed 1/2 fixed and TD commands used this pattern:

```bash
conda run -n d3p python scripts/run_baseline.py --env robomimic_lift --method fixed_chunk --episodes 10 --seed <seed> --max_episode_steps 80 --chunk_horizon 4 --denoise_steps 20 --output_root /tmp/d3p_m4o_lift_multiseed
conda run -n d3p python scripts/run_baseline.py --env robomimic_lift --method td_error_replan --episodes 10 --seed <seed> --max_episode_steps 80 --chunk_horizon 4 --denoise_steps 20 --td_critic_checkpoint /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt --td_calibration /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json --td_threshold_percentile 95 --output_root /tmp/d3p_m4o_lift_multiseed
```

Seed 1/2 PPO commands used this pattern:

```bash
conda run -n d3p python scripts/train_replan_ppo.py --env robomimic_lift --seed <seed> --total_env_steps 720 --rollout_steps 80 --max_episode_steps 80 --eval_episodes 3 --hidden_dims 64 --minibatch_size 40 --update_epochs 2 --lambda_C 0.003 --lambda_D 0.03 --td_critic_checkpoint /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt --uncertainty_replan_bonus 0.3 --output_root /tmp/d3p_m4o_lift_ppo_bonus03_train
conda run -n d3p python scripts/eval_policy.py --checkpoint <best.pt> --episodes 10 --seed <seed> --deterministic true --max_episode_steps 80 --output_root /tmp/d3p_m4o_lift_ppo_bonus03_eval
```

New seed 1/2 output directories:

```text
/tmp/d3p_m4o_lift_multiseed/20260529_020911_robomimic_lift_fixed_chunk_seed1
/tmp/d3p_m4o_lift_multiseed/20260529_020950_robomimic_lift_td_error_replan_seed1
/tmp/d3p_m4o_lift_ppo_bonus03_train/20260529_021028_robomimic_lift_ppo_replan_only_seed1
/tmp/d3p_m4o_lift_ppo_bonus03_eval/20260529_021104_robomimic_lift_ppo_replan_only_eval_seed1
/tmp/d3p_m4o_lift_multiseed/20260529_021139_robomimic_lift_fixed_chunk_seed2
/tmp/d3p_m4o_lift_multiseed/20260529_021217_robomimic_lift_td_error_replan_seed2
/tmp/d3p_m4o_lift_ppo_bonus03_train/20260529_021255_robomimic_lift_ppo_replan_only_seed2
/tmp/d3p_m4o_lift_ppo_bonus03_eval/20260529_021331_robomimic_lift_ppo_replan_only_eval_seed2
```

Per-seed comparison, `episodes=10`, `max_episode_steps=80`:

| Method | Seed | Success | Return | HL Return | NFE/Action | Replan Rate | Learned Replan | Discarded/Episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_chunk | 0 | 0.100 | 1.500 | 1.500 | 5.000 | 0.250 | 0.000 | 0.0 |
| td_error_p95 | 0 | 0.300 | 3.900 | 3.900 | 6.025 | 0.301 | 0.070 | 15.0 |
| ppo_bonus0p3 | 0 | 0.300 | 3.200 | 6.654 | 6.475 | 0.324 | 0.101 | 22.5 |
| fixed_chunk | 1 | 0.000 | 0.000 | 0.000 | 5.000 | 0.250 | 0.000 | 0.0 |
| td_error_p95 | 1 | 0.000 | 0.000 | 0.000 | 5.625 | 0.281 | 0.048 | 9.2 |
| ppo_bonus0p3 | 1 | 0.000 | 0.000 | -0.497 | 5.000 | 0.250 | 0.000 | 0.0 |
| fixed_chunk | 2 | 0.000 | 0.000 | 0.000 | 5.000 | 0.250 | 0.000 | 0.0 |
| td_error_p95 | 2 | 0.200 | 2.300 | 2.300 | 5.775 | 0.289 | 0.049 | 10.7 |
| ppo_bonus0p3 | 2 | 0.000 | 0.000 | -1.158 | 6.425 | 0.321 | 0.199 | 21.2 |

3-seed mean comparison:

| Method | Seeds | Success Mean | Return Mean | HL Return Mean | NFE/Action Mean | Replan Mean | Learned Replan Mean | Discarded/Episode Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_chunk | 3 | 0.033 | 0.500 | 0.500 | 5.000 | 0.250 | 0.000 | 0.0 |
| td_error_p95 | 3 | 0.167 | 2.067 | 2.067 | 5.808 | 0.290 | 0.055 | 11.6 |
| ppo_bonus0p3 | 3 | 0.100 | 1.067 | 1.667 | 5.967 | 0.298 | 0.100 | 14.6 |

Interpretation: the M4n seed-0 PPO result is not robust yet. PPO bonus `0.3` matches TD-error p95 on seed 0, but seed 1 collapses to fixed-like compute with no success and seed 2 spends extra compute without success. TD-error p95 remains the stronger debug method across 3 seeds: higher mean success, higher raw return, and slightly lower compute. PPO bonus shaping is still useful because it can learn uncertainty-conditioned replans, but it needs a stronger training signal, longer training, or a TD-error behavior-cloning warm start before it is ready for final multi-task evaluation.

## M4p TD-Error Behavior-Cloning Warm Start For PPO

M4p adds an opt-in behavior-cloning warm start to `scripts/train_replan_ppo.py`. The default behavior is unchanged. When `--bc_warmstart_steps > 0`, the script collects high-level replan labels from a teacher controller, trains the PPO actor with masked cross-entropy, writes `bc_warmstart_examples.npz` and `bc_warmstart_summary.json`, then runs the original PPO loop.

Implementation details:

```text
--bc_teacher fixed|td_error
--bc_warmstart_steps N
--bc_epochs E
--bc_minibatch_size B
--bc_learning_rate LR
--bc_td_calibration <td_threshold_calibration.json>
--bc_td_threshold_percentile 95
```

For the TD-error teacher, M4p uses the same p95 calibration as the current TD-error debug baseline. The actor sees the same PPO feature vector and action mask as online PPO, so this warm start initializes the actual policy used by later PPO updates.

Fake BC smoke command:

```bash
conda run -n d3p python scripts/train_replan_ppo.py --env fake --seed 0 --total_env_steps 12 --rollout_steps 6 --max_episode_steps 6 --eval_episodes 1 --hidden_dims 16 --minibatch_size 6 --update_epochs 1 --bc_teacher fixed --bc_warmstart_steps 6 --bc_epochs 5 --bc_minibatch_size 6 --bc_learning_rate 0.01 --output_root /tmp/d3p_m4p_fake_bc_smoke
```

Fake BC smoke output:

```text
/tmp/d3p_m4p_fake_bc_smoke/20260529_022607_fake_ppo_replan_only_seed0
```

Fake BC smoke verification:

```text
bc_warmstart_examples.npz
bc_warmstart_summary.json
bc_warmstart: examples=6, accuracy=1.0, replan_fraction=0.333, forced_replan_fraction=0.333
```

Lift BC+PPO command:

```bash
conda run -n d3p python scripts/train_replan_ppo.py --env robomimic_lift --seed 0 --total_env_steps 720 --rollout_steps 80 --max_episode_steps 80 --eval_episodes 3 --hidden_dims 64 --minibatch_size 40 --update_epochs 2 --lambda_C 0.003 --lambda_D 0.03 --td_critic_checkpoint /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt --uncertainty_replan_bonus 0.3 --bc_teacher td_error --bc_warmstart_steps 720 --bc_epochs 10 --bc_minibatch_size 80 --bc_learning_rate 0.001 --bc_td_calibration /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json --bc_td_threshold_percentile 95 --output_root /tmp/d3p_m4p_lift_bc_td_ppo
```

Lift BC+PPO output:

```text
/tmp/d3p_m4p_lift_bc_td_ppo/20260529_022649_robomimic_lift_ppo_replan_only_seed0
```

Lift BC+PPO standalone eval command:

```bash
conda run -n d3p python scripts/eval_policy.py --checkpoint /tmp/d3p_m4p_lift_bc_td_ppo/20260529_022649_robomimic_lift_ppo_replan_only_seed0/checkpoints/best.pt --episodes 10 --seed 0 --deterministic true --max_episode_steps 80 --output_root /tmp/d3p_m4p_lift_bc_td_ppo_eval
```

Lift BC+PPO standalone eval output:

```text
/tmp/d3p_m4p_lift_bc_td_ppo_eval/20260529_022730_robomimic_lift_ppo_replan_only_eval_seed0
```

BC-only-ish diagnostic command, using `learning_rate=0.0` for the PPO phase to preserve the BC actor while still producing a checkpoint through the same training script:

```bash
conda run -n d3p python scripts/train_replan_ppo.py --env robomimic_lift --seed 0 --total_env_steps 80 --rollout_steps 80 --max_episode_steps 80 --eval_episodes 3 --hidden_dims 64 --learning_rate 0.0 --minibatch_size 40 --update_epochs 1 --lambda_C 0.003 --lambda_D 0.03 --td_critic_checkpoint /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt --uncertainty_replan_bonus 0.3 --bc_teacher td_error --bc_warmstart_steps 720 --bc_epochs 10 --bc_minibatch_size 80 --bc_learning_rate 0.001 --bc_td_calibration /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json --bc_td_threshold_percentile 95 --output_root /tmp/d3p_m4p_lift_bc_td_only
```

BC-only-ish standalone eval output:

```text
/tmp/d3p_m4p_lift_bc_td_only_eval/20260529_022902_robomimic_lift_ppo_replan_only_eval_seed0
```

Seed-0 10-episode comparison, `max_episode_steps=80`:

| Method | Success | Return | HL Return | NFE/Action | Replan Rate | Learned Replan | Discarded/Episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_chunk | 0.100 | 1.500 | 1.500 | 5.000 | 0.250 | 0.000 | 0.0 |
| td_error_p95 | 0.300 | 3.900 | 3.900 | 6.025 | 0.301 | 0.070 | 15.0 |
| ppo_bonus0p3 | 0.300 | 3.200 | 6.654 | 6.475 | 0.324 | 0.101 | 22.5 |
| bc_p95_only-ish | 0.100 | 0.900 | 1.462 | 5.150 | 0.258 | 0.009 | 2.0 |
| bc_p95_plus_ppo | 0.200 | 3.900 | 4.430 | 5.100 | 0.255 | 0.006 | 1.2 |

BC warm-start stats for both Lift runs:

```text
examples=720, accuracy=0.983, replan_fraction=0.264, forced_replan_fraction=0.244
```

Interpretation: the BC infrastructure works technically, but p95 teacher cloning is not enough. The cloned dataset contains very few learned replans beyond forced buffer-empty replans (`replan_fraction=0.264`, `forced_replan_fraction=0.244`), and both BC-only-ish and BC+PPO policies behave close to fixed_chunk in eval. BC+PPO gets `2/10` success and raw return `3.9`, but it does so with almost no learned replans and does not beat TD-error p95. The next PPO step should either clone a more aggressive teacher, such as TD p90/p80, or weight learned replan examples instead of cloning p95 labels directly.

## M4q More Aggressive TD-Error BC Teachers

M4q uses the M4p BC warm-start path without new code changes, but changes the TD-error teacher percentile from p95 to p90 and p80. The goal is to increase learned-replan labels beyond forced buffer-empty replans and check whether that improves PPO robustness.

P90 BC+PPO command:

```bash
conda run -n d3p python scripts/train_replan_ppo.py --env robomimic_lift --seed 0 --total_env_steps 720 --rollout_steps 80 --max_episode_steps 80 --eval_episodes 3 --hidden_dims 64 --minibatch_size 40 --update_epochs 2 --lambda_C 0.003 --lambda_D 0.03 --td_critic_checkpoint /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt --uncertainty_replan_bonus 0.3 --bc_teacher td_error --bc_warmstart_steps 720 --bc_epochs 10 --bc_minibatch_size 80 --bc_learning_rate 0.001 --bc_td_calibration /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json --bc_td_threshold_percentile 90 --output_root /tmp/d3p_m4q_lift_bc_td_p90_ppo
```

P90 output:

```text
/tmp/d3p_m4q_lift_bc_td_p90_ppo/20260529_023404_robomimic_lift_ppo_replan_only_seed0
/tmp/d3p_m4q_lift_bc_td_p90_eval/20260529_023446_robomimic_lift_ppo_replan_only_eval_seed0
```

P80 BC+PPO command:

```bash
conda run -n d3p python scripts/train_replan_ppo.py --env robomimic_lift --seed 0 --total_env_steps 720 --rollout_steps 80 --max_episode_steps 80 --eval_episodes 3 --hidden_dims 64 --minibatch_size 40 --update_epochs 2 --lambda_C 0.003 --lambda_D 0.03 --td_critic_checkpoint /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt --uncertainty_replan_bonus 0.3 --bc_teacher td_error --bc_warmstart_steps 720 --bc_epochs 10 --bc_minibatch_size 80 --bc_learning_rate 0.001 --bc_td_calibration /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json --bc_td_threshold_percentile 80 --output_root /tmp/d3p_m4q_lift_bc_td_p80_ppo
```

P80 output:

```text
/tmp/d3p_m4q_lift_bc_td_p80_ppo/20260529_023530_robomimic_lift_ppo_replan_only_seed0
/tmp/d3p_m4q_lift_bc_td_p80_eval/20260529_023613_robomimic_lift_ppo_replan_only_eval_seed0
```

BC label statistics:

| Teacher | BC Accuracy | Replan Fraction | Forced Replan Fraction | Learned Label Fraction |
| --- | ---: | ---: | ---: | ---: |
| p95 | 0.983 | 0.264 | 0.244 | 0.019 |
| p90 | 0.963 | 0.293 | 0.229 | 0.064 |
| p80 | 0.913 | 0.343 | 0.207 | 0.136 |

Training and internal-eval diagnostics:

| Teacher | Train Success | Train Return | Train NFE/Action | Train Learned Replan | Internal Eval Success | Internal Eval Return | Internal Eval NFE/Action | Internal Learned Replan |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p95 | 0.000 | 0.000 | 5.250 | 0.021 | 0.000 | 0.000 | 5.000 | 0.000 |
| p90 | 0.222 | 4.111 | 6.028 | 0.074 | 0.000 | 0.000 | 5.000 | 0.000 |
| p80 | 0.333 | 2.222 | 7.194 | 0.169 | 0.333 | 5.667 | 6.667 | 0.108 |

Standalone 10-episode eval, `seed=0`, `max_episode_steps=80`:

| Method | Success | Return | HL Return | NFE/Action | Replan Rate | Learned Replan | Discarded/Episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| td_error_p95 | 0.300 | 3.900 | 3.900 | 6.025 | 0.301 | 0.070 | 15.0 |
| ppo_bonus0p3 | 0.300 | 3.200 | 6.654 | 6.475 | 0.324 | 0.101 | 22.5 |
| bc_p95_plus_ppo | 0.200 | 3.900 | 4.430 | 5.100 | 0.255 | 0.006 | 1.2 |
| bc_p90_plus_ppo | 0.200 | 2.800 | 3.884 | 5.550 | 0.278 | 0.036 | 8.5 |
| bc_p80_plus_ppo | 0.100 | 1.300 | 2.909 | 6.825 | 0.341 | 0.118 | 27.4 |

Interpretation: lowering the teacher threshold does create the intended behavior: learned-replan labels increase from `0.019` at p95 to `0.064` at p90 and `0.136` at p80. That translates into more learned replans during rollout/eval, but not into better 10-episode task return. p90 is the best aggressive-BC row by standalone return/compute, but it still does not beat direct TD-error p95 or the previous PPO bonus-only row. p80 over-replans and loses task return. The practical next step is not a broader percentile sweep; it is either longer PPO training from the p90/p80 warm starts or explicit weighting/regularization so PPO preserves useful learned replans without over-triggering.


## M4r Learned-Replan Weighted BC Warm Start

M4r adds one targeted PPO warm-start knob from the M4q conclusion: `--bc_learned_replan_weight`. The weighted BC loss only upweights teacher labels where the high-level action is replan and `forced_replan=False`; forced buffer-empty replans keep weight `1.0`. The loss is normalized by the minibatch weight sum, so the knob changes class emphasis without scaling the whole actor gradient. The run config and `bc_warmstart_summary.json` now record `learned_replan_fraction`, `learned_replan_weight`, and `mean_sample_weight`.

Verification commands:

```bash
conda run -n d3p python -m pytest -q tests/test_bc_warmstart.py
conda run -n d3p python scripts/train_replan_ppo.py --env fake --seed 0 --total_env_steps 12 --rollout_steps 6 --max_episode_steps 6 --eval_episodes 1 --hidden_dims 16 --minibatch_size 6 --update_epochs 1 --bc_teacher fixed --bc_warmstart_steps 6 --bc_epochs 5 --bc_minibatch_size 6 --bc_learning_rate 0.01 --bc_learned_replan_weight 4.0 --output_root /tmp/d3p_m4r_fake_weighted_bc_smoke
conda run -n d3p python -m pytest -q tests
```

Verification results:

```text
tests/test_bc_warmstart.py: 5 passed
fake smoke: /tmp/d3p_m4r_fake_weighted_bc_smoke/20260529_024631_fake_ppo_replan_only_seed0
fake BC summary: examples=6, accuracy=1.0, replan_fraction=0.333, forced_replan_fraction=0.333, learned_replan_fraction=0.0, learned_replan_weight=4.0, mean_sample_weight=1.0
full tests: 64 passed
```

Lift p95 weight8 train command:

```bash
conda run -n d3p python scripts/train_replan_ppo.py --env robomimic_lift --seed 0 --total_env_steps 720 --rollout_steps 80 --max_episode_steps 80 --eval_episodes 3 --hidden_dims 64 --minibatch_size 40 --update_epochs 2 --lambda_C 0.003 --lambda_D 0.03 --td_critic_checkpoint /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt --uncertainty_replan_bonus 0.3 --bc_teacher td_error --bc_warmstart_steps 720 --bc_epochs 10 --bc_minibatch_size 80 --bc_learning_rate 0.001 --bc_learned_replan_weight 8.0 --bc_td_calibration /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json --bc_td_threshold_percentile 95 --output_root /tmp/d3p_m4r_lift_bc_p95_weight8_ppo
```

Lift p95 weight8 outputs:

```text
/tmp/d3p_m4r_lift_bc_p95_weight8_ppo/20260529_024722_robomimic_lift_ppo_replan_only_seed0
/tmp/d3p_m4r_lift_bc_p95_weight8_eval/20260529_024809_robomimic_lift_ppo_replan_only_eval_seed0
```

Lift p90 weight4 train command:

```bash
conda run -n d3p python scripts/train_replan_ppo.py --env robomimic_lift --seed 0 --total_env_steps 720 --rollout_steps 80 --max_episode_steps 80 --eval_episodes 3 --hidden_dims 64 --minibatch_size 40 --update_epochs 2 --lambda_C 0.003 --lambda_D 0.03 --td_critic_checkpoint /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt --uncertainty_replan_bonus 0.3 --bc_teacher td_error --bc_warmstart_steps 720 --bc_epochs 10 --bc_minibatch_size 80 --bc_learning_rate 0.001 --bc_learned_replan_weight 4.0 --bc_td_calibration /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json --bc_td_threshold_percentile 90 --output_root /tmp/d3p_m4r_lift_bc_p90_weight4_ppo
```

Lift p90 weight4 outputs:

```text
/tmp/d3p_m4r_lift_bc_p90_weight4_ppo/20260529_024852_robomimic_lift_ppo_replan_only_seed0
/tmp/d3p_m4r_lift_bc_p90_weight4_eval/20260529_024932_robomimic_lift_ppo_replan_only_eval_seed0
```

BC and train diagnostics:

| Method | BC Acc | Replan Labels | Forced Labels | Learned Labels | BC Weight | Mean Sample Weight | Train Success | Train Return | Train NFE | Train Learned | Internal Eval Success | Internal Eval Learned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bc_p95_plus_ppo | 0.983 | 0.264 | 0.244 | 0.019 | 1.0 | 1.000 | 0.000 | 0.000 | 5.250 | 0.021 | 0.000 | 0.000 |
| bc_p95_weight8 | 0.996 | 0.264 | 0.244 | 0.019 | 8.0 | 1.136 | 0.000 | 0.000 | 5.444 | 0.040 | 0.000 | 0.000 |
| bc_p90_plus_ppo | 0.963 | 0.293 | 0.229 | 0.064 | 1.0 | 1.000 | 0.222 | 4.111 | 6.028 | 0.074 | 0.000 | 0.000 |
| bc_p90_weight4 | 0.949 | 0.293 | 0.229 | 0.064 | 4.0 | 1.192 | 0.111 | 0.778 | 6.833 | 0.140 | 0.000 | 0.025 |

Standalone 10-episode eval, `seed=0`, `max_episode_steps=80`:

| Method | Success | Return | HL Return | NFE/Action | Replan Rate | Learned Replan | Discarded/Episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_chunk | 0.100 | 1.500 | 1.500 | 5.000 | 0.250 | 0.000 | 0.0 |
| td_error_p95 | 0.300 | 3.900 | 3.900 | 6.025 | 0.301 | 0.070 | 15.0 |
| ppo_bonus0p3 | 0.300 | 3.200 | 6.654 | 6.475 | 0.324 | 0.101 | 22.5 |
| bc_p95_plus_ppo | 0.200 | 3.900 | 4.430 | 5.100 | 0.255 | 0.006 | 1.2 |
| bc_p95_weight8 | 0.100 | 2.800 | 5.374 | 5.725 | 0.286 | 0.050 | 10.5 |
| bc_p90_plus_ppo | 0.200 | 2.800 | 3.884 | 5.550 | 0.278 | 0.036 | 8.5 |
| bc_p90_weight4 | 0.000 | 0.000 | 0.578 | 6.325 | 0.316 | 0.085 | 19.7 |

Interpretation: the weighted BC knob works mechanically and changes behavior, but it does not solve task performance. p95 weight8 increases standalone learned replans from `0.006` to `0.050`, but success drops from `2/10` to `1/10`. p90 weight4 increases learned replans from `0.036` to `0.085`, but drops to `0/10` success and uses more compute. The failure mode is no longer only label imbalance; stronger learned-replan cloning can over-trigger discards or fail to preserve task-helpful replan timing. The next PPO improvement should preserve teacher behavior more directly during PPO updates, for example with a teacher-KL/BC regularizer, or run longer from the best unweighted p90 warm start with stricter evaluation gates.


## M4s PPO BC Regularizer During Updates

M4s moves beyond warm-start-only cloning. It adds `PPOBCRegularizerBatch` and two PPO config fields, `bc_regularizer_coef` and `bc_regularizer_minibatch_size`. When enabled, every PPO minibatch also samples a teacher minibatch from the BC warm-start examples and adds a masked cross-entropy term to the actor loss. The regularizer uses the same sample weights as M4r, so learned replans can still be upweighted, but this M4s diagnostic keeps `learned_replan_weight=1.0` to isolate the effect of preserving the p90 teacher during PPO updates.

Implementation notes:

- `bc_regularizer_coef=0.0` preserves the previous PPO update path.
- `bc_regularizer_coef > 0` requires `--bc_warmstart_steps > 0`, so the teacher batch is explicit and saved as `bc_warmstart_examples.npz`.
- `training_updates.csv` now includes `bc_regularizer_loss` and `bc_regularizer_accuracy`.
- The run config records the regularizer fields under `ppo` and `bc_warmstart`.

Verification commands:

```bash
conda run -n d3p python -m pytest -q tests/test_ppo_update.py
conda run -n d3p python scripts/train_replan_ppo.py --env fake --seed 0 --total_env_steps 12 --rollout_steps 6 --max_episode_steps 6 --eval_episodes 1 --hidden_dims 16 --minibatch_size 6 --update_epochs 1 --bc_teacher fixed --bc_warmstart_steps 6 --bc_epochs 5 --bc_minibatch_size 6 --bc_learning_rate 0.01 --bc_regularizer_coef 0.2 --bc_regularizer_minibatch_size 6 --output_root /tmp/d3p_m4s_fake_bc_regularizer_smoke
conda run -n d3p python -m pytest -q tests
```

Verification results:

```text
tests/test_ppo_update.py: 6 passed
fake smoke: /tmp/d3p_m4s_fake_bc_regularizer_smoke/20260529_025842_fake_ppo_replan_only_seed0
fake last_update: bc_regularizer_loss=0.136, bc_regularizer_accuracy=1.0
full tests: 67 passed
```

Lift p90 regularizer `coef=0.05` train command:

```bash
conda run -n d3p python scripts/train_replan_ppo.py --env robomimic_lift --seed 0 --total_env_steps 720 --rollout_steps 80 --max_episode_steps 80 --eval_episodes 3 --hidden_dims 64 --minibatch_size 40 --update_epochs 2 --lambda_C 0.003 --lambda_D 0.03 --td_critic_checkpoint /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt --uncertainty_replan_bonus 0.3 --bc_teacher td_error --bc_warmstart_steps 720 --bc_epochs 10 --bc_minibatch_size 80 --bc_learning_rate 0.001 --bc_td_calibration /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json --bc_td_threshold_percentile 90 --bc_regularizer_coef 0.05 --bc_regularizer_minibatch_size 80 --output_root /tmp/d3p_m4s_lift_bc_p90_reg005_ppo
```

Lift p90 regularizer `coef=0.2` train command:

```bash
conda run -n d3p python scripts/train_replan_ppo.py --env robomimic_lift --seed 0 --total_env_steps 720 --rollout_steps 80 --max_episode_steps 80 --eval_episodes 3 --hidden_dims 64 --minibatch_size 40 --update_epochs 2 --lambda_C 0.003 --lambda_D 0.03 --td_critic_checkpoint /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt --uncertainty_replan_bonus 0.3 --bc_teacher td_error --bc_warmstart_steps 720 --bc_epochs 10 --bc_minibatch_size 80 --bc_learning_rate 0.001 --bc_td_calibration /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json --bc_td_threshold_percentile 90 --bc_regularizer_coef 0.2 --bc_regularizer_minibatch_size 80 --output_root /tmp/d3p_m4s_lift_bc_p90_reg02_ppo
```

Lift outputs:

```text
coef=0.05 train: /tmp/d3p_m4s_lift_bc_p90_reg005_ppo/20260529_025925_robomimic_lift_ppo_replan_only_seed0
coef=0.05 eval:  /tmp/d3p_m4s_lift_bc_p90_reg005_eval/20260529_030007_robomimic_lift_ppo_replan_only_eval_seed0
coef=0.2 train:  /tmp/d3p_m4s_lift_bc_p90_reg02_ppo/20260529_030050_robomimic_lift_ppo_replan_only_seed0
coef=0.2 eval:   /tmp/d3p_m4s_lift_bc_p90_reg02_eval/20260529_030131_robomimic_lift_ppo_replan_only_eval_seed0
```

Training diagnostics:

| Method | BC Acc | Teacher Learned Labels | Reg Coef | Last Reg Loss | Last Reg Acc | Train Success | Train Return | Train NFE | Train Learned | Internal Eval Success | Internal Eval Learned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bc_p90_plus_ppo | 0.963 | 0.064 | 0.00 | 0.000 | 0.000 | 0.222 | 4.111 | 6.028 | 0.074 | 0.000 | 0.000 |
| bc_p90_reg005 | 0.963 | 0.064 | 0.05 | 0.144 | 0.944 | 0.222 | 4.111 | 6.028 | 0.074 | 0.000 | 0.000 |
| bc_p90_reg02 | 0.963 | 0.064 | 0.20 | 0.143 | 0.944 | 0.222 | 4.111 | 6.083 | 0.079 | 0.000 | 0.013 |

Standalone 10-episode eval, `seed=0`, `max_episode_steps=80`:

| Method | Success | Return | HL Return | NFE/Action | Replan Rate | Learned Replan | Discarded/Episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| td_error_p95 | 0.300 | 3.900 | 3.900 | 6.025 | 0.301 | 0.070 | 15.0 |
| ppo_bonus0p3 | 0.300 | 3.200 | 6.654 | 6.475 | 0.324 | 0.101 | 22.5 |
| bc_p90_plus_ppo | 0.200 | 2.800 | 3.884 | 5.550 | 0.278 | 0.036 | 8.5 |
| bc_p90_reg005 | 0.000 | 0.000 | -0.569 | 5.050 | 0.253 | 0.003 | 0.4 |
| bc_p90_reg02 | 0.200 | 3.400 | 5.900 | 5.675 | 0.284 | 0.043 | 9.8 |

Interpretation: the PPO BC regularizer is functional and `coef=0.2` is the best p90 BC variant so far by raw return and high-level return. It improves over unregularized p90 (`return 3.4` vs `2.8`) without the large discard cost of p90 weight4. It still does not beat direct TD-error p95 on success (`2/10` vs `3/10`) or raw return (`3.4` vs `3.9`). A weak coefficient (`0.05`) effectively collapses deterministic eval back toward fixed_chunk. The next PPO step, if pursued, should be a small coefficient sweep around `0.1-0.3` with multiple seeds; otherwise TD-error p95 remains the strongest debug baseline for moving toward final evaluations.


## M5a Summary Tables And Pareto Plot Tooling

M5a implements the first reusable analysis/plotting slice required by Agents.md priority 9. The new `adaptive_diffusion.plots` module and `scripts/make_plots.py` convert existing `eval_summary.json` / PPO `train_summary.json` artifacts into summary tables, aggregate tables, and dependency-free SVG Pareto plots.

Implemented artifacts:

- `adaptive_diffusion/plots.py`: loads labeled runs, discovers eval summaries, aggregates seed groups, writes CSV/Markdown summaries, and writes SVG Pareto plots for success and return vs NFE/action.
- `scripts/make_plots.py`: CLI wrapper supporting repeated `--run label=path`, recursive `--input` discovery, and `--plot pareto,summary_table` selection. `replan_timeline` is intentionally rejected until M5b implements it.
- `tests/test_make_plots.py`: validates row loading, multi-seed aggregation, SVG/table outputs, and the CLI path.

Verification commands:

```bash
conda run -n d3p python -m pytest -q tests/test_make_plots.py
conda run -n d3p python -m pytest -q tests
conda run -n d3p python scripts/make_plots.py --plot pareto,summary_table --run fixed_chunk=/tmp/d3p_m4l_lift_eval10/20260529_014340_robomimic_lift_fixed_chunk_seed0 --run td_error_p95=/tmp/d3p_m4l_lift_eval10/20260529_014417_robomimic_lift_td_error_replan_seed0 --run ppo_bonus0p3=/tmp/d3p_m4n_lift_bonus_sweep_eval/20260529_020515_robomimic_lift_ppo_replan_only_eval_seed0 --run bc_p90_plus_ppo=/tmp/d3p_m4q_lift_bc_td_p90_eval/20260529_023446_robomimic_lift_ppo_replan_only_eval_seed0 --run bc_p90_reg02=/tmp/d3p_m4s_lift_bc_p90_reg02_eval/20260529_030131_robomimic_lift_ppo_replan_only_eval_seed0 --output /tmp/d3p_m5a_lift_debug_plots
```

Verification results:

```text
tests/test_make_plots.py: 3 passed
full tests: 70 passed
analysis output: /tmp/d3p_m5a_lift_debug_plots
files: summary_table.csv, summary_table.md, aggregate_summary.csv, aggregate_summary.md, analysis_summary.json, pareto_lift_success.svg, pareto_lift_return.svg
```

M5a Lift debug aggregate table, generated from the existing 10-episode seed-0 eval runs:

| Label | Success | Return | NFE/Action | Replan Rate | Learned Replan |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed_chunk | 0.100 | 1.500 | 5.000 | 0.250 | 0.000 |
| td_error_p95 | 0.300 | 3.900 | 6.025 | 0.301 | 0.070 |
| ppo_bonus0p3 | 0.300 | 3.200 | 6.475 | 0.324 | 0.101 |
| bc_p90_plus_ppo | 0.200 | 2.800 | 5.550 | 0.278 | 0.036 |
| bc_p90_reg02 | 0.200 | 3.400 | 5.675 | 0.284 | 0.043 |

Interpretation: the plotting pipeline is now ready for final multi-seed evaluations. On the current Lift debug set, `td_error_p95` remains the strongest raw-performance point, while `bc_p90_reg02` moves PPO closer to TD-error at slightly lower NFE/action. The current plot is still seed-0 / 10-episode diagnostic evidence, not a final Pareto claim. The remaining M5 work is to feed this same tool with 3-seed Lift/Can/Square eval outputs and add replan timeline visualizations.


## M5b Replan Timeline Visualization

M5b extends the M5a plotting pipeline with `--plot replan_timeline`. The timeline renderer reads `metrics_step.csv` or `eval_metrics_step.csv` for each labeled run and writes one SVG per selected episode. Each SVG has two panels: uncertainty / TD error / reward on top, and buffer remaining / plan age / discarded actions on the bottom. Forced replans and learned replans are drawn as separate vertical markers.

Implementation details:

- `adaptive_diffusion.plots.resolve_step_metrics`: resolves step metrics from run directories, eval directories, summary JSON paths, or direct CSV paths.
- `adaptive_diffusion.plots.write_replan_timeline_svg`: creates a dependency-free SVG for one episode.
- `adaptive_diffusion.plots.write_timeline_outputs`: writes `timelines/replan_timeline_{label}_ep{episode}.svg` and `timeline_manifest.csv`.
- `scripts/make_plots.py --plot pareto,summary_table,replan_timeline --timeline_episodes 1`: now supports the Agents.md plot CLI shape.

Verification commands:

```bash
conda run -n d3p python -m pytest -q tests/test_make_plots.py
conda run -n d3p python -m pytest -q tests
conda run -n d3p python scripts/make_plots.py --plot pareto,summary_table,replan_timeline --timeline_episodes 1 --run fixed_chunk=/tmp/d3p_m4l_lift_eval10/20260529_014340_robomimic_lift_fixed_chunk_seed0 --run td_error_p95=/tmp/d3p_m4l_lift_eval10/20260529_014417_robomimic_lift_td_error_replan_seed0 --run bc_p90_reg02=/tmp/d3p_m4s_lift_bc_p90_reg02_eval/20260529_030131_robomimic_lift_ppo_replan_only_eval_seed0 --output /tmp/d3p_m5b_lift_debug_timelines
```

Verification results:

```text
tests/test_make_plots.py: 5 passed
full tests: 72 passed
analysis output: /tmp/d3p_m5b_lift_debug_timelines
timeline manifest: /tmp/d3p_m5b_lift_debug_timelines/timeline_manifest.csv
replan timeline SVGs:
- /tmp/d3p_m5b_lift_debug_timelines/timelines/replan_timeline_fixed_chunk_ep0.svg
- /tmp/d3p_m5b_lift_debug_timelines/timelines/replan_timeline_td_error_p95_ep0.svg
- /tmp/d3p_m5b_lift_debug_timelines/timelines/replan_timeline_bc_p90_reg02_ep0.svg
```

Interpretation: M5 now has the core analysis artifacts needed by Agents.md: summary tables, Pareto plots, and replan timeline/buffer behavior plots. The current examples are still Lift seed-0 debug artifacts; final evidence still requires 3-seed Lift/Can/Square evaluation outputs. The next M5 step should focus on running or orchestrating those evals, not adding more plotting infrastructure.


## M5c Final Eval Suite Orchestration

M5c adds a repeatable suite runner for the final evaluation stage. `scripts/run_final_eval_suite.py` wraps existing baseline and PPO evaluation scripts, records every command, loads each run's summary, writes suite-level result CSVs, and then calls the M5a/M5b analysis pipeline to produce summary tables, Pareto SVGs, and replan timeline SVGs.

Implemented behavior:

- Baseline jobs call `scripts/run_baseline.py` for `fixed_chunk`, `high_compute`, `low_compute`, and `td_error_replan` on supported Robomimic state envs. TD-error jobs can use global `--td_critic_checkpoint/--td_calibration` or env-specific repeated `--td_artifact env:critic.pt=calibration.json` entries.
- PPO jobs call `scripts/eval_policy.py` from repeated `--ppo_checkpoint label=path` or env-specific `--ppo_checkpoint env:label=path` entries.
- The suite writes `suite_config.yaml`, `suite_commands.csv`, `suite_results.csv`, `suite_summary.json`, and a nested `plots/` directory with M5a/M5b artifacts.
- `--dry_run` writes the exact commands without launching evaluations.
- Duplicate timeline filenames from multi-seed runs are disambiguated with a `_run{index}` suffix.

Example final-eval command shape, once final checkpoints and TD critic artifacts are selected:

```bash
conda run -n d3p python scripts/run_final_eval_suite.py \
  --envs robomimic_lift,robomimic_can \
  --seeds 0,1,2 \
  --episodes 100 \
  --baseline_methods fixed_chunk,high_compute,low_compute,td_error_replan \
  --td_artifact robomimic_lift:/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json \
  --td_artifact robomimic_can:/path/to/can_td_critic.pt=/path/to/can_td_threshold_calibration.json \
  --td_threshold_percentile 95 \
  --ppo_checkpoint robomimic_lift:bc_p90_reg02=/tmp/d3p_m4s_lift_bc_p90_reg02_ppo/20260529_030050_robomimic_lift_ppo_replan_only_seed0/checkpoints/best.pt \
  --timeline_episodes 1 \
  --output_root /tmp/d3p_final_eval_suite
```

Square is still not runnable through the local scripts because current `run_baseline.py` / `train_replan_ppo.py` task maps only include Lift and Can, and this workspace currently has `square.json` env metadata but no discovered Square checkpoint or normalization artifacts under local `log/` and `data/`. The suite fails clearly if `robomimic_square` is requested before those paths are added.

Verification commands:

```bash
conda run -n d3p python -m pytest -q tests/test_final_eval_suite.py
conda run -n d3p python -m pytest -q tests
conda run -n d3p python scripts/run_final_eval_suite.py --envs fake --seeds 0,1 --episodes 2 --max_episode_steps 3 --baseline_methods "" --ppo_checkpoint fake:ppo_fake=/tmp/d3p_m5c_fake_ppo.pt --timeline_episodes 1 --output_root /tmp/d3p_m5c_fake_suite
```

Verification results:

```text
tests/test_final_eval_suite.py: 3 passed
full tests: 76 passed
fake suite: /tmp/d3p_m5c_fake_suite/20260529_033537_fake_final_eval_suite
num_jobs=2, num_completed=2
suite_results.csv rows: ppo_fake seed 0 and ppo_fake seed 1
aggregate: success=1.000, return=1.294, NFE/action=20.000, replan_rate=1.000, learned_replan_rate=0.667
timeline SVGs:
- /tmp/d3p_m5c_fake_suite/20260529_033537_fake_final_eval_suite/plots/timelines/replan_timeline_ppo_fake_ep0.svg
- /tmp/d3p_m5c_fake_suite/20260529_033537_fake_final_eval_suite/plots/timelines/replan_timeline_ppo_fake_ep0_run1.svg
```

Interpretation: final-eval execution is now scripted rather than manual. The remaining bottleneck is experimental runtime and artifact availability: run the suite for real Lift/Can 3-seed evals, add Square task paths if local checkpoint/normalization become available, then use the generated suite `plots/` directory as the final analysis bundle.


## M5d Can TD Artifact And Baseline Suite Slice

M5d starts the Can transfer work required by Agents.md Stage 4. Before this slice, Lift had a TD critic/calibration artifact and several Lift PPO checkpoints, but Can only had the pretrained diffusion policy checkpoint and normalization file. This slice creates the missing Can TD-error artifact, runs a short Can percentile sweep, and feeds a Can baseline suite through the M5c final-suite runner.

Can artifact generation commands:

```bash
conda run -n d3p python scripts/run_baseline.py \
  --env robomimic_can \
  --method fixed_chunk \
  --episodes 3 \
  --max_episode_steps 80 \
  --save_transitions true \
  --output_root /tmp/d3p_m5d_can_td_data

conda run -n d3p python scripts/train_td_critic.py \
  --dataset /tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/transitions.npz \
  --epochs 15 \
  --batch_size 128 \
  --hidden_dims 256,256 \
  --device cpu

conda run -n d3p python scripts/calibrate_td_threshold.py \
  --dataset /tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/transitions.npz \
  --critic_checkpoint /tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_critic/td_critic.pt \
  --device cpu
```

Can TD artifact:

```text
fixed data run: /tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0
transitions: 240
obs_dim: 23
critic: /tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_critic/td_critic.pt
critic final_loss: 0.0009586150
calibration: /tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_threshold_calibration.json
calibrated uncertainty thresholds: p70=-0.0080, p80=0.3243, p90=0.7213, p95=1.5011
abs TD-error percentiles: p70=0.01194, p80=0.01470, p90=0.02222, p95=0.03367
```

Can percentile sweep command:

```bash
conda run -n d3p python scripts/run_td_threshold_sweep.py \
  --env robomimic_can \
  --threshold_mode percentile \
  --percentiles 70,80,90,95 \
  --critic_checkpoint /tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_critic/td_critic.pt \
  --calibration /tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_threshold_calibration.json \
  --episodes 3 \
  --seed 0 \
  --max_episode_steps 80 \
  --output_root /tmp/d3p_m5d_can_td_sweep
```

Can TD sweep output: `/tmp/d3p_m5d_can_td_sweep/20260529_034625_robomimic_can_td_threshold_sweep_seed0`

| Threshold | Success | Return | NFE/Action | Replan Rate | Learned Replan | Discarded/Episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| p70 | 0.0 | 0.0 | 9.500 | 0.475 | 0.308 | 69.333 |
| p80 | 0.0 | 0.0 | 9.000 | 0.450 | 0.275 | 61.667 |
| p90 | 0.0 | 0.0 | 7.417 | 0.371 | 0.183 | 38.333 |
| p95 | 0.0 | 0.0 | 6.583 | 0.329 | 0.104 | 22.333 |

The short Can sweep does not show task success at 3 episodes and 80 steps, but the TD controller is not degenerate. Lower percentiles spend substantially more compute and discard more buffered actions. With all rows tied on task performance, p95 is the current Can TD candidate because it is the lowest-compute non-fixed TD setting.

Can baseline suite command:

```bash
conda run -n d3p python scripts/run_final_eval_suite.py \
  --envs robomimic_can \
  --seeds 0 \
  --episodes 3 \
  --max_episode_steps 80 \
  --baseline_methods fixed_chunk,high_compute,low_compute,td_error_replan \
  --td_artifact robomimic_can:/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_threshold_calibration.json \
  --td_threshold_percentile 95 \
  --timeline_episodes 1 \
  --output_root /tmp/d3p_m5d_can_baseline_suite
```

Can baseline suite output: `/tmp/d3p_m5d_can_baseline_suite/20260529_034831_robomimic_can_final_eval_suite`

| Method | Success | Return | NFE/Action | Total Wall Time/Action | Replan Rate | Learned Replan | Discarded/Episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_chunk | 0.0 | 0.0 | 5.000 | 0.001129 | 0.250 | 0.000 | 0.000 |
| high_compute | 0.0 | 0.0 | 20.000 | 0.003827 | 1.000 | 0.000 | 0.000 |
| low_compute | 0.0 | 0.0 | 1.000 | 0.000441 | 0.250 | 0.000 | 0.000 |
| td_error_p95 | 0.0 | 0.0 | 6.333 | 0.001374 | 0.317 | 0.092 | 20.333 |

Generated analysis artifacts:

```text
summary table: /tmp/d3p_m5d_can_baseline_suite/20260529_034831_robomimic_can_final_eval_suite/plots/summary_table.csv
aggregate table: /tmp/d3p_m5d_can_baseline_suite/20260529_034831_robomimic_can_final_eval_suite/plots/aggregate_summary.csv
Pareto success plot: /tmp/d3p_m5d_can_baseline_suite/20260529_034831_robomimic_can_final_eval_suite/plots/pareto_can_success.svg
Pareto return plot: /tmp/d3p_m5d_can_baseline_suite/20260529_034831_robomimic_can_final_eval_suite/plots/pareto_can_return.svg
timeline manifest: /tmp/d3p_m5d_can_baseline_suite/20260529_034831_robomimic_can_final_eval_suite/plots/timeline_manifest.csv
timeline SVGs: fixed_chunk, high_compute, low_compute, td_error_p95 episode 0
```

Interpretation: the Can TD baseline path is now ready for larger evaluation and for Can PPO warm-start experiments. This short slice is negative task-performance evidence, not a final Can claim: all methods have zero success/return at 3 episodes and 80 steps. It still verifies the Can DP checkpoint, Can normalization, TD critic, calibrated TD replanning, high/low compute references, final-suite aggregation, Pareto outputs, and timeline outputs under the same scripts that will be used for the full evaluation.


## M5e First Can PPO Checkpoint And Comparison Suite

M5e trains the first Can-compatible binary PPO replanning checkpoint. The point of this slice is to close the `obs_dim=23` gap from M5d: previous PPO checkpoints were Lift-only and could not be evaluated on Can. The Can run intentionally reuses the best debug-scale Lift recipe from M4s: TD-error p90 teacher, `uncertainty_replan_bonus=0.3`, and BC regularizer coefficient `0.2`.

Can PPO train command:

```bash
conda run -n d3p python scripts/train_replan_ppo.py \
  --env robomimic_can \
  --seed 0 \
  --total_env_steps 720 \
  --rollout_steps 80 \
  --max_episode_steps 80 \
  --eval_episodes 3 \
  --hidden_dims 64 \
  --minibatch_size 40 \
  --update_epochs 2 \
  --lambda_C 0.003 \
  --lambda_D 0.03 \
  --td_critic_checkpoint /tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_critic/td_critic.pt \
  --uncertainty_replan_bonus 0.3 \
  --bc_teacher td_error \
  --bc_warmstart_steps 720 \
  --bc_epochs 10 \
  --bc_minibatch_size 80 \
  --bc_learning_rate 0.001 \
  --bc_td_calibration /tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_threshold_calibration.json \
  --bc_td_threshold_percentile 90 \
  --bc_regularizer_coef 0.2 \
  --bc_regularizer_minibatch_size 80 \
  --output_root /tmp/d3p_m5e_can_bc_p90_reg02_ppo
```

Can PPO training output:

```text
train run: /tmp/d3p_m5e_can_bc_p90_reg02_ppo/20260529_035456_robomimic_can_ppo_replan_only_seed0
best checkpoint: /tmp/d3p_m5e_can_bc_p90_reg02_ppo/20260529_035456_robomimic_can_ppo_replan_only_seed0/checkpoints/best.pt
actual_env_steps: 720
updates: 9
episodes: 9
```

BC warm-start and last PPO update diagnostics:

| Metric | Value |
| --- | ---: |
| BC examples | 720 |
| BC accuracy | 0.949 |
| Teacher replan labels | 0.415 |
| Teacher forced replan labels | 0.186 |
| Teacher learned replan labels | 0.229 |
| Last update continue actions | 43 |
| Last update replan actions | 37 |
| Last update forced replans | 13 |
| Last update mean NFE/action | 9.250 |
| Last update BC regularizer loss | 0.145 |
| Last update BC regularizer accuracy | 0.950 |

Training summary, `seed=0`, 9 train episodes at `max_episode_steps=80`:

| Success | Return | HL Return | NFE/Action | Replan Rate | Learned Replan | Discarded/Episode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.0 | 6.209 | 8.222 | 0.411 | 0.235 | 50.111 |

Internal deterministic eval, 3 episodes:

| Success | Return | HL Return | NFE/Action | Replan Rate | Learned Replan | Discarded/Episode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.0 | 3.906 | 8.000 | 0.400 | 0.200 | 47.333 |

Standalone deterministic eval command:

```bash
conda run -n d3p python scripts/eval_policy.py \
  --env robomimic_can \
  --checkpoint /tmp/d3p_m5e_can_bc_p90_reg02_ppo/20260529_035456_robomimic_can_ppo_replan_only_seed0/checkpoints/best.pt \
  --episodes 10 \
  --seed 0 \
  --max_episode_steps 80 \
  --deterministic true \
  --output_root /tmp/d3p_m5e_can_bc_p90_reg02_eval
```

Standalone eval output: `/tmp/d3p_m5e_can_bc_p90_reg02_eval/20260529_035558_robomimic_can_ppo_replan_only_eval_seed0`

The standalone 10-episode eval produced `success=0.0`, `return=0.0`, `NFE/action=9.125`, `replan_rate=0.456`, and `learned_replan_rate=0.273`.

Can 10-episode comparison suite command:

```bash
conda run -n d3p python scripts/run_final_eval_suite.py \
  --envs robomimic_can \
  --seeds 0 \
  --episodes 10 \
  --max_episode_steps 80 \
  --baseline_methods fixed_chunk,high_compute,low_compute,td_error_replan \
  --td_artifact robomimic_can:/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_threshold_calibration.json \
  --td_threshold_percentile 95 \
  --ppo_checkpoint robomimic_can:can_bc_p90_reg02=/tmp/d3p_m5e_can_bc_p90_reg02_ppo/20260529_035456_robomimic_can_ppo_replan_only_seed0/checkpoints/best.pt \
  --timeline_episodes 1 \
  --output_root /tmp/d3p_m5e_can_comparison_suite
```

Can comparison suite output: `/tmp/d3p_m5e_can_comparison_suite/20260529_035645_robomimic_can_final_eval_suite`

| Method | Success | Return | HL Return | NFE/Action | Total Wall Time/Action | Replan Rate | Learned Replan | Discarded/Episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low_compute | 0.0 | 0.0 | 0.000 | 1.000 | 0.000288 | 0.250 | 0.000 | 0.000 |
| fixed_chunk | 0.0 | 0.0 | 0.000 | 5.000 | 0.000977 | 0.250 | 0.000 | 0.000 |
| td_error_p95 | 0.0 | 0.0 | 0.000 | 6.275 | 0.001189 | 0.314 | 0.086 | 18.400 |
| can_bc_p90_reg02 | 0.0 | 0.0 | 5.502 | 8.300 | 0.002001 | 0.415 | 0.219 | 50.800 |
| high_compute | 0.0 | 0.0 | 0.000 | 20.000 | 0.003721 | 1.000 | 0.000 | 0.000 |

Generated comparison artifacts:

```text
suite results: /tmp/d3p_m5e_can_comparison_suite/20260529_035645_robomimic_can_final_eval_suite/suite_results.csv
aggregate table: /tmp/d3p_m5e_can_comparison_suite/20260529_035645_robomimic_can_final_eval_suite/plots/aggregate_summary.csv
Pareto success plot: /tmp/d3p_m5e_can_comparison_suite/20260529_035645_robomimic_can_final_eval_suite/plots/pareto_can_success.svg
Pareto return plot: /tmp/d3p_m5e_can_comparison_suite/20260529_035645_robomimic_can_final_eval_suite/plots/pareto_can_return.svg
timeline manifest: /tmp/d3p_m5e_can_comparison_suite/20260529_035645_robomimic_can_final_eval_suite/plots/timeline_manifest.csv
timeline SVGs: fixed_chunk, high_compute, low_compute, td_error_p95, can_bc_p90_reg02 episode 0
```

Interpretation: Can PPO is now mechanically complete for the main replan-only method, but this debug-scale checkpoint is not useful on raw task performance. It learns non-forced replans and preserves the p90 teacher behavior better than fixed_chunk, yet all Can methods still have zero raw success/return at 10 episodes and 80 steps. Compared with TD-error p95, the PPO checkpoint spends more compute (`8.3` vs `6.275` NFE/action), replans more often (`0.415` vs `0.314`), and discards more actions (`50.8` vs `18.4`) without raw task gain. This is useful negative evidence: Can needs either longer-horizon/longer-training evaluation, a stronger reward signal, or a better base success regime before PPO improvements can be measured.


## M5f Lift/Can 3-Seed Debug Suite

M5f is the first cross-task multi-seed suite that evaluates the current fixed references, TD-error p95, and env-specific PPO checkpoints in one run. This is still debug-scale evaluation, not final reporting: `episodes=10`, `seeds=0,1,2`, and `max_episode_steps=80`. It is useful because it exercises the same final-suite machinery and produces cross-env aggregate tables, Pareto plots, and timeline plots from 30 completed jobs.

Command:

```bash
conda run -n d3p python scripts/run_final_eval_suite.py \
  --envs robomimic_lift,robomimic_can \
  --seeds 0,1,2 \
  --episodes 10 \
  --max_episode_steps 80 \
  --baseline_methods fixed_chunk,high_compute,low_compute,td_error_replan \
  --td_artifact robomimic_lift:/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json \
  --td_artifact robomimic_can:/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_threshold_calibration.json \
  --td_threshold_percentile 95 \
  --ppo_checkpoint robomimic_lift:lift_bc_p90_reg02=/tmp/d3p_m4s_lift_bc_p90_reg02_ppo/20260529_030050_robomimic_lift_ppo_replan_only_seed0/checkpoints/best.pt \
  --ppo_checkpoint robomimic_can:can_bc_p90_reg02=/tmp/d3p_m5e_can_bc_p90_reg02_ppo/20260529_035456_robomimic_can_ppo_replan_only_seed0/checkpoints/best.pt \
  --timeline_episodes 1 \
  --output_root /tmp/d3p_m5f_lift_can_multiseed_suite
```

Suite output:

```text
suite: /tmp/d3p_m5f_lift_can_multiseed_suite/20260529_040243_robomimic_lift_robomimic_can_final_eval_suite
num_jobs: 30
num_completed: 30
suite results: /tmp/d3p_m5f_lift_can_multiseed_suite/20260529_040243_robomimic_lift_robomimic_can_final_eval_suite/suite_results.csv
aggregate table: /tmp/d3p_m5f_lift_can_multiseed_suite/20260529_040243_robomimic_lift_robomimic_can_final_eval_suite/plots/aggregate_summary.csv
summary table: /tmp/d3p_m5f_lift_can_multiseed_suite/20260529_040243_robomimic_lift_robomimic_can_final_eval_suite/plots/summary_table.csv
Pareto plots: pareto_lift_success.svg, pareto_lift_return.svg, pareto_can_success.svg, pareto_can_return.svg
timeline manifest: /tmp/d3p_m5f_lift_can_multiseed_suite/20260529_040243_robomimic_lift_robomimic_can_final_eval_suite/plots/timeline_manifest.csv
timeline SVGs: 30 one-episode timelines, one per run/method/seed with duplicate names disambiguated by _runN
```

Aggregate results, 3 seeds x 10 episodes per row:

| Env | Method | Success Mean | Success SEM | Return Mean | Return SEM | NFE/Action | Replan Rate | Learned Replan | Discarded/Episode |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lift | low_compute | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.250 | 0.000 | 0.000 |
| lift | fixed_chunk | 0.067 | 0.033 | 0.867 | 0.467 | 5.000 | 0.250 | 0.000 | 0.000 |
| lift | td_error_p95 | 0.033 | 0.033 | 0.933 | 0.933 | 5.650 | 0.283 | 0.047 | 9.567 |
| lift | lift_bc_p90_reg02 | 0.100 | 0.000 | 1.633 | 0.584 | 5.450 | 0.273 | 0.029 | 6.467 |
| lift | high_compute | 0.000 | 0.000 | 0.000 | 0.000 | 20.000 | 1.000 | 0.000 | 0.000 |
| can | low_compute | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.250 | 0.000 | 0.000 |
| can | fixed_chunk | 0.000 | 0.000 | 0.000 | 0.000 | 5.000 | 0.250 | 0.000 | 0.000 |
| can | td_error_p95 | 0.000 | 0.000 | 0.000 | 0.000 | 6.283 | 0.314 | 0.087 | 18.567 |
| can | can_bc_p90_reg02 | 0.000 | 0.000 | 0.000 | 0.000 | 8.617 | 0.431 | 0.239 | 55.933 |
| can | high_compute | 0.000 | 0.000 | 0.000 | 0.000 | 20.000 | 1.000 | 0.000 | 0.000 |

Seed-level raw task results:

```text
Lift fixed_chunk success/return: seed0 0.1/1.6, seed1 0.1/1.0, seed2 0.0/0.0
Lift td_error_p95 success/return: seed0 0.1/2.8, seed1 0.0/0.0, seed2 0.0/0.0
Lift lift_bc_p90_reg02 success/return: seed0 0.1/1.1, seed1 0.1/1.0, seed2 0.1/2.8
Can all methods success/return: 0.0/0.0 for seeds 0, 1, and 2
```

Interpretation: on Lift at this debug scale, the current PPO checkpoint is the best raw-performance point and a better Pareto candidate than TD-error p95: `lift_bc_p90_reg02` reaches `0.10` mean success and `1.633` mean return at `5.45` NFE/action, while `td_error_p95` reaches `0.033` success and `0.933` return at `5.65` NFE/action. Fixed_chunk remains competitive at `0.067` success and exactly `5.0` NFE/action. High compute performs poorly in this short horizon despite using `20.0` NFE/action. On Can, this suite confirms the issue from M5d/M5e: the current 80-step debug setting gives no raw task signal for any method, so Can cannot yet support a performance claim. Can PPO learns frequent non-forced replans, but those replans increase cost and discards without raw task return.


## M5g Lift 100-Episode/Seed Best-Method Suite

M5g checks whether the M5f Lift conclusion survives a larger evaluation. M5f used 10 episodes per seed and favored `lift_bc_p90_reg02`; this suite keeps the same 80-step horizon but increases Lift evaluation to 100 episodes per seed for the three most important methods: `fixed_chunk`, `td_error_p95`, and `lift_bc_p90_reg02`. This is closer to Agents.md's default episode count, but still not the final full-horizon Robomimic evaluation because `max_episode_steps=80` is retained for comparability with prior PPO training/eval.

Command:

```bash
conda run -n d3p python scripts/run_final_eval_suite.py \
  --envs robomimic_lift \
  --seeds 0,1,2 \
  --episodes 100 \
  --max_episode_steps 80 \
  --baseline_methods fixed_chunk,td_error_replan \
  --td_artifact robomimic_lift:/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json \
  --td_threshold_percentile 95 \
  --ppo_checkpoint robomimic_lift:lift_bc_p90_reg02=/tmp/d3p_m4s_lift_bc_p90_reg02_ppo/20260529_030050_robomimic_lift_ppo_replan_only_seed0/checkpoints/best.pt \
  --timeline_episodes 1 \
  --output_root /tmp/d3p_m5g_lift_100ep_suite
```

Suite output:

```text
suite: /tmp/d3p_m5g_lift_100ep_suite/20260529_042149_robomimic_lift_final_eval_suite
num_jobs: 9
num_completed: 9
suite results: /tmp/d3p_m5g_lift_100ep_suite/20260529_042149_robomimic_lift_final_eval_suite/suite_results.csv
aggregate table: /tmp/d3p_m5g_lift_100ep_suite/20260529_042149_robomimic_lift_final_eval_suite/plots/aggregate_summary.csv
summary table: /tmp/d3p_m5g_lift_100ep_suite/20260529_042149_robomimic_lift_final_eval_suite/plots/summary_table.csv
Pareto plots: pareto_lift_success.svg, pareto_lift_return.svg
timeline manifest: /tmp/d3p_m5g_lift_100ep_suite/20260529_042149_robomimic_lift_final_eval_suite/plots/timeline_manifest.csv
timeline SVGs: 9 one-episode timelines, one per run/method/seed
```

Aggregate results, 3 seeds x 100 episodes per row:

| Method | Success Mean | Success SEM | Return Mean | Return SEM | HL Return Mean | NFE/Action | Replan Rate | Learned Replan | Discarded/Episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_chunk | 0.053 | 0.028 | 0.673 | 0.339 | 0.673 | 5.000 | 0.250 | 0.000 | 0.000 |
| td_error_p95 | 0.073 | 0.015 | 0.693 | 0.129 | 0.693 | 5.753 | 0.288 | 0.053 | 10.830 |
| lift_bc_p90_reg02 | 0.060 | 0.017 | 0.607 | 0.119 | 1.212 | 5.250 | 0.263 | 0.016 | 3.413 |

Seed-level raw task results:

```text
fixed_chunk success/return: seed0 0.02/0.45, seed1 0.11/1.34, seed2 0.03/0.23
td_error_p95 success/return: seed0 0.07/0.58, seed1 0.10/0.95, seed2 0.05/0.55
lift_bc_p90_reg02 success/return: seed0 0.03/0.39, seed1 0.09/0.80, seed2 0.06/0.63
```

Interpretation: the 10-episode M5f Lift PPO advantage does not survive the larger 100-episode-per-seed check. TD-error p95 is now the best raw task point by mean success and mean return, but only slightly above fixed_chunk in return while using more compute and discarding about 10.8 actions per episode. Fixed_chunk remains a strong low-compute baseline at `5.0` NFE/action. `lift_bc_p90_reg02` is not the best raw-performance point; it has slightly higher success than fixed_chunk but lower raw return and higher compute. Its high-level return remains higher because the PPO reward includes the uncertainty bonus and cost terms, so raw task metrics should remain the primary comparison. This larger Lift evidence pushes the current best final candidate back toward TD-error p95 for raw performance, with fixed_chunk as the hard-to-beat efficiency reference.


## M5h Lift Full-Horizon 100-Episode/Seed Suite

M5h repeats the M5g Lift comparison at the Robomimic full 300-step horizon. This removes the main limitation of M5g while keeping the same three methods and 3 seeds x 100 episodes per method: `fixed_chunk`, `td_error_p95`, and `lift_bc_p90_reg02`.

Command:

```bash
conda run -n d3p python scripts/run_final_eval_suite.py \
  --envs robomimic_lift \
  --seeds 0,1,2 \
  --episodes 100 \
  --max_episode_steps 300 \
  --baseline_methods fixed_chunk,td_error_replan \
  --td_artifact robomimic_lift:/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json \
  --td_threshold_percentile 95 \
  --ppo_checkpoint robomimic_lift:lift_bc_p90_reg02=/tmp/d3p_m4s_lift_bc_p90_reg02_ppo/20260529_030050_robomimic_lift_ppo_replan_only_seed0/checkpoints/best.pt \
  --timeline_episodes 1 \
  --output_root /tmp/d3p_m5h_lift_full_horizon_suite
```

Suite output:

```text
suite: /tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite
num_jobs: 9
num_completed: 9
suite results: /tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite/suite_results.csv
aggregate table: /tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite/plots/aggregate_summary.csv
summary table: /tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite/plots/summary_table.csv
Pareto plots: pareto_lift_success.svg, pareto_lift_return.svg
timeline manifest: /tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite/plots/timeline_manifest.csv
timeline SVGs: 9 one-episode timelines, one per run/method/seed
```

Aggregate results, 3 seeds x 100 episodes per row:

| Method | Success Mean | Success SEM | Return Mean | Return SEM | HL Return Mean | NFE/Action | Replan Rate | Learned Replan | Discarded/Episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_chunk | 0.790 | 0.020 | 36.213 | 1.071 | 36.213 | 5.000 | 0.250 | 0.000 | 0.000 |
| td_error_p95 | 0.693 | 0.007 | 46.067 | 0.939 | 46.067 | 8.331 | 0.417 | 0.233 | 198.077 |
| lift_bc_p90_reg02 | 0.753 | 0.022 | 30.710 | 0.798 | 60.985 | 6.259 | 0.313 | 0.087 | 73.973 |

Seed-level raw task results:

```text
fixed_chunk success/return: seed0 0.81/37.11, seed1 0.81/37.45, seed2 0.75/34.08
td_error_p95 success/return: seed0 0.70/45.34, seed1 0.70/44.93, seed2 0.68/47.93
lift_bc_p90_reg02 success/return: seed0 0.77/31.78, seed1 0.78/31.20, seed2 0.71/29.15
```

Interpretation: full-horizon Lift changes the 80-step picture. `fixed_chunk` is now the best success-rate and compute Pareto point: `0.790` success at exactly `5.0` NFE/action with no discarded actions. `td_error_p95` gets the highest mean raw return (`46.067` versus fixed_chunk's `36.213`), but success drops to `0.693` and compute rises to `8.331` NFE/action with about `198` discarded actions per episode. The current PPO checkpoint is between them on success and compute, but it is below fixed_chunk on raw success and below both fixed_chunk and TD-error on raw return. Its high-level return is still largest because it is the PPO-shaped reward, not the task return. For final Lift claims, fixed_chunk is the strongest success/efficiency baseline, while TD-error p95 is only attractive if raw return is prioritized over success and compute.


## M5i Can Full-Horizon 100-Episode/Seed Suite

M5i repeats the Can comparison at the Robomimic full 300-step horizon. Earlier Can suites used `max_episode_steps=80` and produced all-zero task returns, so this run is the first Can evaluation that can support a raw task comparison. The suite uses the same three core methods and 3 seeds x 100 episodes per method: `fixed_chunk`, `td_error_p95`, and `can_bc_p90_reg02`.

Command:

```bash
conda run -n d3p python scripts/run_final_eval_suite.py \
  --envs robomimic_can \
  --seeds 0,1,2 \
  --episodes 100 \
  --max_episode_steps 300 \
  --baseline_methods fixed_chunk,td_error_replan \
  --td_artifact robomimic_can:/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_threshold_calibration.json \
  --td_threshold_percentile 95 \
  --ppo_checkpoint robomimic_can:can_bc_p90_reg02=/tmp/d3p_m5e_can_bc_p90_reg02_ppo/20260529_035456_robomimic_can_ppo_replan_only_seed0/checkpoints/best.pt \
  --timeline_episodes 1 \
  --output_root /tmp/d3p_m5i_can_full_horizon_suite
```

Suite output:

```text
suite: /tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite
num_jobs: 9
num_completed: 9
suite results: /tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite/suite_results.csv
aggregate table: /tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite/plots/aggregate_summary.csv
summary table: /tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite/plots/summary_table.csv
Pareto plots: pareto_can_success.svg, pareto_can_return.svg
timeline manifest: /tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite/plots/timeline_manifest.csv
timeline SVGs: 9 one-episode timelines, one per run/method/seed
```

Aggregate results, 3 seeds x 100 episodes per row:

| Method | Success Mean | Success SEM | Return Mean | Return SEM | HL Return Mean | NFE/Action | Replan Rate | Learned Replan | Discarded/Episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_chunk | 0.703 | 0.052 | 82.053 | 4.452 | 82.053 | 5.000 | 0.250 | 0.000 | 0.000 |
| td_error_p95 | 0.713 | 0.024 | 84.047 | 2.188 | 84.047 | 8.846 | 0.442 | 0.263 | 228.903 |
| can_bc_p90_reg02 | 0.717 | 0.013 | 88.970 | 0.431 | 163.869 | 12.338 | 0.617 | 0.493 | 437.747 |

Seed-level raw task results:

```text
fixed_chunk success/return: seed0 0.61/74.03, seed1 0.79/89.41, seed2 0.71/82.72
td_error_p95 success/return: seed0 0.76/88.37, seed1 0.70/82.47, seed2 0.68/81.30
can_bc_p90_reg02 success/return: seed0 0.73/88.42, seed1 0.73/89.82, seed2 0.69/88.67
```

Interpretation: full-horizon Can resolves the all-zero short-horizon issue. `can_bc_p90_reg02` has the highest mean success (`0.717`) and mean return (`88.970`), but only by a small margin over TD-error and fixed_chunk while using much more compute (`12.338` NFE/action), replanning more often (`0.617`), and discarding about `438` actions per episode. `td_error_p95` is a middle point: slightly higher success and return than fixed_chunk, but at `8.846` NFE/action and about `229` discarded actions per episode. `fixed_chunk` remains the efficiency baseline with competitive raw task performance at exactly `5.0` NFE/action. For Can, the current PPO checkpoint is the best raw-performance point, but its cost is high enough that the Pareto story depends on whether the objective values raw return or compute efficiency.


## M5j Square Runner Support And Lift/Can Combined Snapshot

M5j removes the code-level Square blocker and records the remaining artifact blocker. `robomimic_square` is now accepted by the baseline runner, PPO trainer/evaluator path through the shared task table, TD threshold sweep, PPO cost sweep, and final eval suite orchestration. The Square entry follows the existing low-dimensional config: `env_name=square`, `obs_dim=23`, `action_dim=7`, `policy_horizon=4`, `max_episode_steps=400`, and the configured diffusion MLP checkpoint path `robomimic-pretrain/square/square_pre_diffusion_mlp_ta4_td20/2024-07-10_01-46-16/checkpoint/state_8000.pt`.

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_final_eval_suite.py tests/test_td_threshold_sweep.py
conda run -n d3p python scripts/run_final_eval_suite.py \
  --envs robomimic_square \
  --seeds 0 \
  --episodes 1 \
  --max_episode_steps 400 \
  --baseline_methods fixed_chunk \
  --dry_run \
  --output_root /tmp/d3p_m5j_square_dry_run
conda run -n d3p python scripts/run_baseline.py \
  --env robomimic_square \
  --method fixed_chunk \
  --episodes 1 \
  --debug true \
  --output_root /tmp/d3p_m5j_square_missing_probe
```

Results:

```text
pytest: 10 passed in 1.99s
Square dry-run suite: /tmp/d3p_m5j_square_dry_run/20260529_054234_robomimic_square_final_eval_suite
Square dry-run jobs: 1 command, 0 completed because dry_run=true
Square dry-run command: run_baseline.py --env robomimic_square --method fixed_chunk --episodes 1 --seed 0 --max_episode_steps 400
Square probe result: expected FileNotFoundError for /home/koukanni/Documents/D3P/log/robomimic-pretrain/square/square_pre_diffusion_mlp_ta4_td20/2024-07-10_01-46-16/checkpoint/state_8000.pt
Square normalization audit: data/robomimic/square/normalization.npz is also absent locally
```

Current full-horizon Lift/Can summary, 3 seeds x 100 episodes per row:

| Env | Method | Success Mean | Success SEM | Return Mean | Return SEM | NFE/Action | Replan Rate | Learned Replan | Discarded/Episode | Primary Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| lift | fixed_chunk | 0.790 | 0.020 | 36.213 | 1.071 | 5.000 | 0.250 | 0.000 | 0.000 | Best Lift success/efficiency point |
| lift | td_error_p95 | 0.693 | 0.007 | 46.067 | 0.939 | 8.331 | 0.417 | 0.233 | 198.077 | Best Lift raw return, worse success/compute |
| lift | lift_bc_p90_reg02 | 0.753 | 0.022 | 30.710 | 0.798 | 6.259 | 0.313 | 0.087 | 73.973 | Not a robust Lift raw-task improvement |
| can | fixed_chunk | 0.703 | 0.052 | 82.053 | 4.452 | 5.000 | 0.250 | 0.000 | 0.000 | Competitive Can efficiency point |
| can | td_error_p95 | 0.713 | 0.024 | 84.047 | 2.188 | 8.846 | 0.442 | 0.263 | 228.903 | Small Can raw gain, substantial cost |
| can | can_bc_p90_reg02 | 0.717 | 0.013 | 88.970 | 0.431 | 12.338 | 0.617 | 0.493 | 437.747 | Best Can raw point, highest cost |

Combined interpretation: fixed_chunk remains the strongest cross-task efficiency reference. Adaptive replanning is not uniformly better: TD-error buys Lift return and small Can gains at substantial compute/discard cost, while PPO helps Can raw metrics but not Lift. Square is now a data/artifact problem rather than an unsupported-runner problem; once Square checkpoint and normalization are present, the next sequence is fixed Square rollouts, Square TD critic/calibration, Square PPO training/eval, then a final 3-task suite.


## M5k Runtime Artifact Overrides For Square

M5k makes the Square path less brittle by adding explicit runtime artifact overrides. The baseline runner, PPO trainer, and PPO evaluator now accept `--base_policy_checkpoint`, `--normalization_path`, and `--env_meta_path`. The final eval suite can pass global overrides or repeated env-specific `--runtime_artifact ENV:base_policy_checkpoint=normalization_npz` entries through to both baseline and PPO eval jobs. This does not complete Square evaluation, but it means Square artifacts no longer have to be copied into the hard-coded DPPO directory layout before running the final suite.

Verification:

```bash
conda run -n d3p python -m py_compile \
  scripts/run_baseline.py \
  scripts/train_replan_ppo.py \
  scripts/eval_policy.py \
  scripts/run_final_eval_suite.py \
  scripts/run_td_threshold_sweep.py \
  scripts/run_ppo_cost_sweep.py
conda run -n d3p python -m pytest -q tests/test_final_eval_suite.py tests/test_td_threshold_sweep.py
conda run -n d3p python scripts/run_final_eval_suite.py \
  --envs robomimic_square \
  --seeds 0 \
  --episodes 1 \
  --max_episode_steps 400 \
  --baseline_methods fixed_chunk \
  --runtime_artifact robomimic_square:/tmp/d3p_m5k_square_runtime_artifacts/square_policy.pt=/tmp/d3p_m5k_square_runtime_artifacts/normalization.npz \
  --dry_run \
  --output_root /tmp/d3p_m5k_square_runtime_dry_run
```

Results:

```text
py_compile: passed
pytest: 11 passed in 1.99s
Square runtime dry-run suite: /tmp/d3p_m5k_square_runtime_dry_run/20260529_055321_robomimic_square_final_eval_suite
Square runtime dry-run jobs: 1 command, 0 completed because dry_run=true
Generated command includes: --env robomimic_square --base_policy_checkpoint /tmp/d3p_m5k_square_runtime_artifacts/square_policy.pt --normalization_path /tmp/d3p_m5k_square_runtime_artifacts/normalization.npz
```

Interpretation: the remaining Square blocker is now only artifact availability and model/data validity. The final suite can be launched with real Square artifacts using `--runtime_artifact robomimic_square:/path/to/state_8000.pt=/path/to/normalization.npz`. If an alternate Square env meta file is required, add `--env_meta_path /path/to/square.json`; otherwise the checked-in `cfg/robomimic/env_meta/square.json` is used.


## M5l Final Readiness Report

M5l adds `scripts/check_final_readiness.py`, a small readiness reporter for the final Agents.md evidence path. It checks each requested Robomimic env for the runtime artifacts needed by the pretrained diffusion policy (`base_policy_checkpoint`, `normalization`, `env_meta`) and the adaptive comparison artifacts (`td_critic`, `td_calibration`, `ppo_checkpoint`). It writes both JSON and Markdown, and for Square it also emits the exact command sequence for fixed rollouts, TD critic training, TD calibration, PPO replan training, and the final suite.

Verification:

```bash
conda run -n d3p python -m py_compile scripts/check_final_readiness.py
conda run -n d3p python -m pytest -q tests/test_final_readiness.py tests/test_final_eval_suite.py tests/test_td_threshold_sweep.py
conda run -n d3p python scripts/check_final_readiness.py \
  --td_artifact robomimic_lift:/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json \
  --td_artifact robomimic_can:/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_threshold_calibration.json \
  --ppo_checkpoint robomimic_lift:lift_bc_p90_reg02=/tmp/d3p_m4s_lift_bc_p90_reg02_ppo/20260529_030050_robomimic_lift_ppo_replan_only_seed0/checkpoints/best.pt \
  --ppo_checkpoint robomimic_can:can_bc_p90_reg02=/tmp/d3p_m5e_can_bc_p90_reg02_ppo/20260529_035456_robomimic_can_ppo_replan_only_seed0/checkpoints/best.pt \
  --output_dir /tmp/d3p_m5l_readiness
```

Report output:

```text
readiness dir: /tmp/d3p_m5l_readiness/20260529_060902_final_readiness
readiness json: /tmp/d3p_m5l_readiness/20260529_060902_final_readiness/readiness.json
readiness markdown: /tmp/d3p_m5l_readiness/20260529_060902_final_readiness/readiness.md
pytest: 13 passed in 2.37s
all_final_eval_ready: false
```

Readiness summary:

| Env | Runtime Ready | Final Eval Ready | Missing |
| --- | ---: | ---: | --- |
| robomimic_lift | true | true | none |
| robomimic_can | true | true | none |
| robomimic_square | false | false | base_policy_checkpoint, normalization, td_critic, td_calibration, ppo_checkpoint |

Interpretation: the final readiness state is now machine-readable. Lift and Can have all artifacts needed for the current fixed/TD/PPO final-suite path. Square has checked-in env metadata and runner support, but still lacks the pretrained diffusion checkpoint, normalization file, TD critic/calibration, and PPO checkpoint. The generated Square command sequence is available in the readiness Markdown and should be used once real Square artifacts are provided.


## M5m Readiness Artifact Integrity Guard

M5m strengthens `scripts/check_final_readiness.py` so final-readiness checks require artifacts to be real non-empty files, not merely existing paths. Each artifact check now records `exists`, `valid`, `size_bytes`, and `reason`. This matters because the local Square dry-run directory contains placeholder files that exist but are zero bytes, and those must not be accepted as evidence that Square is ready.

Local Square artifact search:

```text
Default Square checkpoint: missing at /home/koukanni/Documents/D3P/log/robomimic-pretrain/square/square_pre_diffusion_mlp_ta4_td20/2024-07-10_01-46-16/checkpoint/state_8000.pt
Default Square normalization: missing at /home/koukanni/Documents/D3P/data/robomimic/square/normalization.npz
Square env meta: present at /home/koukanni/Documents/D3P/cfg/robomimic/env_meta/square.json
Dummy dry-run Square files: /tmp/d3p_m5k_square_runtime_artifacts/square_policy.pt and normalization.npz are 0 bytes
```

Verification:

```bash
conda run -n d3p python -m py_compile scripts/check_final_readiness.py
conda run -n d3p python -m pytest -q tests/test_final_readiness.py tests/test_final_eval_suite.py tests/test_td_threshold_sweep.py
conda run -n d3p python scripts/check_final_readiness.py \
  --td_artifact robomimic_lift:/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json \
  --td_artifact robomimic_can:/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_threshold_calibration.json \
  --ppo_checkpoint robomimic_lift:lift_bc_p90_reg02=/tmp/d3p_m4s_lift_bc_p90_reg02_ppo/20260529_030050_robomimic_lift_ppo_replan_only_seed0/checkpoints/best.pt \
  --ppo_checkpoint robomimic_can:can_bc_p90_reg02=/tmp/d3p_m5e_can_bc_p90_reg02_ppo/20260529_035456_robomimic_can_ppo_replan_only_seed0/checkpoints/best.pt \
  --output_dir /tmp/d3p_m5m_readiness
conda run -n d3p python scripts/check_final_readiness.py \
  --envs robomimic_square \
  --runtime_artifact robomimic_square:/tmp/d3p_m5k_square_runtime_artifacts/square_policy.pt=/tmp/d3p_m5k_square_runtime_artifacts/normalization.npz \
  --output_dir /tmp/d3p_m5m_dummy_square_readiness
```

Report output:

```text
full readiness dir: /tmp/d3p_m5m_readiness/20260529_061431_final_readiness
dummy Square readiness dir: /tmp/d3p_m5m_dummy_square_readiness/20260529_061442_final_readiness
pytest: 14 passed in 2.36s
all_final_eval_ready: false
```

Readiness summary:

| Env | Runtime Ready | Final Eval Ready | Missing / Invalid |
| --- | ---: | ---: | --- |
| robomimic_lift | true | true | none |
| robomimic_can | true | true | none |
| robomimic_square | false | false | base_policy_checkpoint, normalization, td_critic, td_calibration, ppo_checkpoint |
| robomimic_square with dummy runtime artifacts | false | false | base_policy_checkpoint `empty_file`, normalization `empty_file`, td_critic, td_calibration, ppo_checkpoint |

Interpretation: the final readiness gate is now stricter and closer to the actual completion requirement. Lift and Can still have valid non-empty runtime/adaptive artifacts. Square remains incomplete, and the only discovered local Square runtime files are placeholders that the checker now rejects.


## M5n Final Evidence Audit

M5n adds `scripts/audit_final_evidence.py`, a completion-oriented audit over the artifacts needed to close the Agents.md objective. It checks final eval suites directly instead of relying on notes alone: suite summary, non-dry-run status, completed job count, result rows for expected method x seed combinations, per-run `eval_summary.json` episode counts, plot/table outputs, Pareto SVGs, timeline SVGs, readiness JSON, and markdown coverage.

Verification:

```bash
conda run -n d3p python -m py_compile scripts/audit_final_evidence.py scripts/check_final_readiness.py scripts/run_final_eval_suite.py scripts/make_plots.py
conda run -n d3p python -m pytest -q tests/test_final_evidence_audit.py tests/test_final_readiness.py tests/test_final_eval_suite.py tests/test_make_plots.py
conda run -n d3p python scripts/audit_final_evidence.py \
  --suite robomimic_lift=/tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite \
  --expected_label robomimic_lift:fixed_chunk \
  --expected_label robomimic_lift:td_error_p95 \
  --expected_label robomimic_lift:lift_bc_p90_reg02 \
  --expected_label robomimic_can:fixed_chunk \
  --expected_label robomimic_can:td_error_p95 \
  --expected_label robomimic_can:can_bc_p90_reg02 \
  --expected_label robomimic_square:fixed_chunk \
  --expected_label robomimic_square:td_error_p95 \
  --expected_label robomimic_square:square_bc_p90_reg02 \
  --readiness_json /tmp/d3p_m5m_readiness/20260529_061431_final_readiness/readiness.json \
  --output_dir /tmp/d3p_m5n_final_evidence_audit
```

Report output:

```text
audit dir: /tmp/d3p_m5n_final_evidence_audit/20260529_062732_final_evidence_audit
audit json: /tmp/d3p_m5n_final_evidence_audit/20260529_062732_final_evidence_audit/final_evidence_audit.json
audit markdown: /tmp/d3p_m5n_final_evidence_audit/20260529_062732_final_evidence_audit/final_evidence_audit.md
pytest: 18 passed in 3.78s
all_goal_evidence_ready: false
all_suites_ready: false
readiness_ready: false
markdown_ready: true
```

Audit summary:

| Env | Suite Ready | Jobs | Rows | Failed Checks |
| --- | ---: | ---: | ---: | --- |
| robomimic_lift | true | 9/9 | 9 | none |
| robomimic_can | true | 9/9 | 9 | none |
| robomimic_square | false | none | none | suite_provided |

Interpretation: the final evidence gate is now explicit. Current Lift and Can full-horizon suites have the expected 3 methods x 3 seeds, at least 100 episodes per run, and all suite analysis outputs. The experiment markdown contains the required analysis sections. The goal still cannot close because Square has no suite and the readiness report still marks `robomimic_square` incomplete.


## M5o Reproducible Final Report Generator

M5o adds `scripts/build_final_report.py`, which turns a `final_evidence_audit.json` into reproducible analysis artifacts: `final_results_matrix.csv`, `final_report.json`, and `final_report.md`. It reads each audited suite's `plots/aggregate_summary.csv`, preserves the numeric means/SEMs, identifies best-success, best-return, and lowest-compute methods per environment, and writes explicit missing rows for environments that are not ready. This makes the final analysis report a generated artifact rather than only hand-written notes.

Verification:

```bash
conda run -n d3p python -m py_compile scripts/build_final_report.py scripts/audit_final_evidence.py scripts/check_final_readiness.py scripts/run_final_eval_suite.py scripts/make_plots.py
conda run -n d3p python -m pytest -q tests/test_final_report.py tests/test_final_evidence_audit.py tests/test_final_readiness.py tests/test_final_eval_suite.py tests/test_make_plots.py
conda run -n d3p python scripts/build_final_report.py \
  --audit_json /tmp/d3p_m5n_final_evidence_audit/20260529_062732_final_evidence_audit/final_evidence_audit.json \
  --output_dir /tmp/d3p_m5o_final_report
```

Report output:

```text
report dir: /tmp/d3p_m5o_final_report/20260529_063416_final_report
report json: /tmp/d3p_m5o_final_report/20260529_063416_final_report/final_report.json
report csv: /tmp/d3p_m5o_final_report/20260529_063416_final_report/final_results_matrix.csv
report markdown: /tmp/d3p_m5o_final_report/20260529_063416_final_report/final_report.md
pytest: 21 passed in 3.79s
all_goal_evidence_ready: false
```

Generated environment summary:

| Env | Ready | Best Success | Best Return | Lowest Compute | Status |
| --- | ---: | --- | --- | --- | --- |
| can | true | can_bc_p90_reg02 (0.717) | can_bc_p90_reg02 (88.970) | fixed_chunk (5.000) | complete |
| lift | true | fixed_chunk (0.790) | td_error_p95 (46.067) | fixed_chunk (5.000) | complete |
| square | false | n/a | n/a | n/a | missing:suite_provided |

Interpretation: the analysis dataset is now reproducible from the audit and suite outputs. The generated report makes the current cross-task story explicit: fixed_chunk is the Lift success/efficiency winner, TD-error is Lift's return winner at higher compute, PPO is Can's raw success/return winner at much higher compute, and Square is an explicit missing row. The generated report is still incomplete because `all_goal_evidence_ready=false` until Square artifacts and suite outputs exist.


## M5p Final Evidence Pipeline Wrapper

M5p adds `scripts/run_final_evidence_pipeline.py`, a thin wrapper that runs the M5n audit and M5o report generation in one command. The pipeline writes `audit/final_evidence_audit.json`, `audit/final_evidence_audit.md`, `report/final_report.json`, `report/final_results_matrix.csv`, `report/final_report.md`, and top-level `pipeline_summary.json` / `pipeline_summary.md`. This is the command to rerun after Square artifacts and Square suite outputs are available.

Verification:

```bash
conda run -n d3p python -m py_compile scripts/run_final_evidence_pipeline.py scripts/build_final_report.py scripts/audit_final_evidence.py scripts/check_final_readiness.py scripts/run_final_eval_suite.py scripts/make_plots.py
conda run -n d3p python -m pytest -q tests/test_final_evidence_pipeline.py tests/test_final_report.py tests/test_final_evidence_audit.py tests/test_final_readiness.py tests/test_final_eval_suite.py tests/test_make_plots.py
conda run -n d3p python scripts/run_final_evidence_pipeline.py \
  --suite robomimic_lift=/tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite \
  --expected_label robomimic_lift:fixed_chunk \
  --expected_label robomimic_lift:td_error_p95 \
  --expected_label robomimic_lift:lift_bc_p90_reg02 \
  --expected_label robomimic_can:fixed_chunk \
  --expected_label robomimic_can:td_error_p95 \
  --expected_label robomimic_can:can_bc_p90_reg02 \
  --expected_label robomimic_square:fixed_chunk \
  --expected_label robomimic_square:td_error_p95 \
  --expected_label robomimic_square:square_bc_p90_reg02 \
  --readiness_json /tmp/d3p_m5m_readiness/20260529_061431_final_readiness/readiness.json \
  --output_dir /tmp/d3p_m5p_final_pipeline
```

Pipeline output:

```text
pipeline dir: /tmp/d3p_m5p_final_pipeline/20260529_064149_final_evidence_pipeline
pipeline summary: /tmp/d3p_m5p_final_pipeline/20260529_064149_final_evidence_pipeline/pipeline_summary.md
audit json: /tmp/d3p_m5p_final_pipeline/20260529_064149_final_evidence_pipeline/audit/final_evidence_audit.json
final report: /tmp/d3p_m5p_final_pipeline/20260529_064149_final_evidence_pipeline/report/final_report.md
final matrix: /tmp/d3p_m5p_final_pipeline/20260529_064149_final_evidence_pipeline/report/final_results_matrix.csv
pytest: 23 passed in 3.81s
all_goal_evidence_ready: false
all_suites_ready: false
readiness_ready: false
markdown_ready: true
```

Pipeline summary:

| Env | Suite Ready | Failed Checks |
| --- | ---: | --- |
| robomimic_lift | true | none |
| robomimic_can | true | none |
| robomimic_square | false | suite_provided |

Interpretation: the final close-out path is now a single reproducible command once Square is available. The current run still intentionally fails the final gate because the Square suite is missing and readiness marks `robomimic_square` incomplete.


## M5q Require-Complete Pipeline Gate

M5q adds `--require_complete` to `scripts/run_final_evidence_pipeline.py`. With this flag, the pipeline still writes the audit, report, matrix, and summary artifacts, but exits with status `1` if `all_goal_evidence_ready=false`. This makes the final objective closure mechanically checkable: once Square is complete, the same command should exit `0`; until then it fails clearly while preserving diagnostic outputs.

Verification:

```bash
conda run -n d3p python -m py_compile scripts/run_final_evidence_pipeline.py scripts/build_final_report.py scripts/audit_final_evidence.py scripts/check_final_readiness.py scripts/run_final_eval_suite.py scripts/make_plots.py
conda run -n d3p python -m pytest -q tests/test_final_evidence_pipeline.py tests/test_final_report.py tests/test_final_evidence_audit.py tests/test_final_readiness.py tests/test_final_eval_suite.py tests/test_make_plots.py
conda run -n d3p python scripts/run_final_evidence_pipeline.py \
  --suite robomimic_lift=/tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite \
  --expected_label robomimic_lift:fixed_chunk \
  --expected_label robomimic_lift:td_error_p95 \
  --expected_label robomimic_lift:lift_bc_p90_reg02 \
  --expected_label robomimic_can:fixed_chunk \
  --expected_label robomimic_can:td_error_p95 \
  --expected_label robomimic_can:can_bc_p90_reg02 \
  --expected_label robomimic_square:fixed_chunk \
  --expected_label robomimic_square:td_error_p95 \
  --expected_label robomimic_square:square_bc_p90_reg02 \
  --readiness_json /tmp/d3p_m5m_readiness/20260529_061431_final_readiness/readiness.json \
  --output_dir /tmp/d3p_m5q_final_gate \
  --require_complete
```

Gate output:

```text
expected exit code: 1
pipeline dir: /tmp/d3p_m5q_final_gate/20260529_064733_final_evidence_pipeline
pipeline summary: /tmp/d3p_m5q_final_gate/20260529_064733_final_evidence_pipeline/pipeline_summary.md
pytest: 24 passed in 4.94s
all_goal_evidence_ready: false
all_suites_ready: false
readiness_ready: false
markdown_ready: true
require_complete: true
```

Gate summary:

| Env | Suite Ready | Failed Checks |
| --- | ---: | --- |
| robomimic_lift | true | none |
| robomimic_can | true | none |
| robomimic_square | false | suite_provided |

Interpretation: the completion gate is active and behaving correctly. The current failure is expected and useful: it proves that the final goal cannot be accidentally marked complete while Square evidence is missing, but all diagnostic artifacts are still generated for analysis.

## M5r Generated-Readiness Final Pipeline

M5r lets `scripts/run_final_evidence_pipeline.py` generate `readiness.json` and `readiness.md` internally with `--generate_readiness`, using the same artifact arguments as `scripts/check_final_readiness.py`. This reduces final close-out to one pipeline command: generate readiness, audit suites, build report/matrix, write summary, and optionally enforce `--require_complete`.

Verification:

```bash
conda run -n d3p python -m py_compile scripts/run_final_evidence_pipeline.py scripts/build_final_report.py scripts/audit_final_evidence.py scripts/check_final_readiness.py scripts/run_final_eval_suite.py scripts/make_plots.py
conda run -n d3p python -m pytest -q tests/test_final_evidence_pipeline.py tests/test_final_report.py tests/test_final_evidence_audit.py tests/test_final_readiness.py tests/test_final_eval_suite.py tests/test_make_plots.py
conda run -n d3p python scripts/run_final_evidence_pipeline.py \
  --suite robomimic_lift=/tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite \
  --expected_label robomimic_lift:fixed_chunk \
  --expected_label robomimic_lift:td_error_p95 \
  --expected_label robomimic_lift:lift_bc_p90_reg02 \
  --expected_label robomimic_can:fixed_chunk \
  --expected_label robomimic_can:td_error_p95 \
  --expected_label robomimic_can:can_bc_p90_reg02 \
  --expected_label robomimic_square:fixed_chunk \
  --expected_label robomimic_square:td_error_p95 \
  --expected_label robomimic_square:square_bc_p90_reg02 \
  --generate_readiness \
  --td_artifact robomimic_lift:/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json \
  --td_artifact robomimic_can:/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_threshold_calibration.json \
  --ppo_checkpoint robomimic_lift:lift_bc_p90_reg02=/tmp/d3p_m4s_lift_bc_p90_reg02_ppo/20260529_030050_robomimic_lift_ppo_replan_only_seed0/checkpoints/best.pt \
  --ppo_checkpoint robomimic_can:can_bc_p90_reg02=/tmp/d3p_m5e_can_bc_p90_reg02_ppo/20260529_035456_robomimic_can_ppo_replan_only_seed0/checkpoints/best.pt \
  --output_dir /tmp/d3p_m5r_generated_readiness_pipeline \
  --require_complete
```

Generated-readiness pipeline output:

```text
expected exit code: 1
pipeline dir: /tmp/d3p_m5r_generated_readiness_pipeline/20260529_065609_final_evidence_pipeline
readiness: /tmp/d3p_m5r_generated_readiness_pipeline/20260529_065609_final_evidence_pipeline/readiness/readiness.md
pipeline summary: /tmp/d3p_m5r_generated_readiness_pipeline/20260529_065609_final_evidence_pipeline/pipeline_summary.md
pytest: 25 passed in 4.96s
all_goal_evidence_ready: false
all_suites_ready: false
readiness_ready: false
markdown_ready: true
generated_readiness: true
require_complete: true
```

Readiness summary:

| Env | Runtime Ready | Final Eval Ready | Missing |
| --- | ---: | ---: | --- |
| robomimic_lift | true | true | none |
| robomimic_can | true | true | none |
| robomimic_square | false | false | base_policy_checkpoint, normalization, td_critic, td_calibration, ppo_checkpoint |

Pipeline suite summary:

| Env | Suite Ready | Failed Checks |
| --- | ---: | --- |
| robomimic_lift | true | none |
| robomimic_can | true | none |
| robomimic_square | false | suite_provided |

Interpretation: the final evidence pipeline now owns readiness generation, so the eventual final command only needs the Square runtime, TD, PPO, and suite artifacts added as arguments. The current `--require_complete` failure is still expected because the local Square runtime/training/eval evidence is missing.

## M6a Rule-Based Denoise-Only Smoke

M6a starts the secondary dynamic-denoising extension without touching PPO training or full hierarchical control. The implemented path keeps fixed-interval replanning (`H=4`) but lets the controller choose the denoising budget on each forced replan from `[4, 8, 12, 20]` using normalized uncertainty thresholds `[0.5, 1.0, 1.5]`.

Implementation notes:

- `DenoiseOnlyController` replans only when the buffer is empty and attaches the selected `denoise_steps` to the controller decision.
- `UncertaintyRuleDenoiseAdaptor` maps the latest uncertainty estimate to candidate denoising steps.
- `AdaptiveDiffusionExecutor` now honors `decision.denoise_steps` when sampling a new chunk.
- `TorchDiffusionPolicyAdapter` supports fewer DDPM denoising steps than the configured maximum and restores the wrapped model's original loop length after each sample.
- `scripts/run_baseline.py --method denoise_only` and `scripts/run_final_eval_suite.py --baseline_methods denoise_only` are wired for debug and suite orchestration.

Verification:

```bash
conda run -n d3p python -m py_compile adaptive_diffusion/controllers/denoise_adaptor.py adaptive_diffusion/controllers/fixed.py adaptive_diffusion/executor.py adaptive_diffusion/diffusion_interface.py adaptive_diffusion/reference_configs.py scripts/run_baseline.py scripts/run_final_eval_suite.py
conda run -n d3p python -m pytest -q tests/test_denoise_adaptor.py tests/test_diffusion_policy_adapter.py tests/test_executor_smoke.py tests/test_reference_configs.py tests/test_final_eval_suite.py
conda run -n d3p python -m pytest -q tests
conda run -n d3p python scripts/run_baseline.py --env robomimic_lift --method denoise_only --episodes 2 --debug true --td_critic_checkpoint /tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt --output_root /tmp/d3p_m6a_denoise_only_smoke
conda run -n d3p python scripts/run_final_eval_suite.py --envs robomimic_lift --seeds 0 --episodes 1 --max_episode_steps 5 --baseline_methods denoise_only --td_artifact robomimic_lift:/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json --dry_run --output_root /tmp/d3p_m6a_denoise_only_suite_dry_run
```

Test and smoke output:

```text
focused pytest: 20 passed in 2.01s
full tests: 99 passed in 9.89s
smoke dir: /tmp/d3p_m6a_denoise_only_smoke/20260529_071629_robomimic_lift_denoise_only_seed0
suite dry-run dir: /tmp/d3p_m6a_denoise_only_suite_dry_run/20260529_071727_robomimic_lift_final_eval_suite
success_rate: 0.0
mean_episode_return: 0.0
mean_episode_length: 20.0
mean_nfe_per_action: 1.2
replan_rate: 0.25
forced_replan_rate: 0.25
denoise_steps_on_replan: [4, 4, 4, 4, 4, 4, 12, 4, 4, 4]
nfe_values_per_step: [0, 4, 12]
```

Interpretation: this is a software-path milestone, not a task-performance claim. The denoise-only ablation now runs on real Lift state observations, produces normal metrics, and can be included in final-suite orchestration. The debug run mostly chose 4-step denoising and once escalated to 12 steps when uncertainty reached `1.4685`, reducing mean NFE/action to `1.2` versus fixed chunk's `5.0` debug accounting. Longer Lift/Can/Square evaluations are still needed before making any Pareto claim.

## M6b Denoise-Only Short Lift/Can Suite

M6b runs a small comparable suite for the denoise-only ablation against fixed_chunk on Lift and Can. This is still a short-horizon diagnostic (`max_episode_steps=80`, `episodes=5`, `seeds=0,1,2`), not final evidence. The goal is to verify multi-seed suite orchestration, plots, timeline generation, and cost accounting for the secondary dynamic-denoising path.

Command:

```bash
conda run -n d3p python scripts/run_final_eval_suite.py \
  --envs robomimic_lift,robomimic_can \
  --seeds 0,1,2 \
  --episodes 5 \
  --max_episode_steps 80 \
  --baseline_methods fixed_chunk,denoise_only \
  --td_artifact robomimic_lift:/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json \
  --td_artifact robomimic_can:/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_threshold_calibration.json \
  --timeline_episodes 1 \
  --output_root /tmp/d3p_m6b_denoise_only_short_suite
```

Suite output:

```text
suite dir: /tmp/d3p_m6b_denoise_only_short_suite/20260529_072508_robomimic_lift_robomimic_can_final_eval_suite
num_jobs: 12
num_completed: 12
summary_table: /tmp/d3p_m6b_denoise_only_short_suite/20260529_072508_robomimic_lift_robomimic_can_final_eval_suite/plots/summary_table.md
aggregate_summary: /tmp/d3p_m6b_denoise_only_short_suite/20260529_072508_robomimic_lift_robomimic_can_final_eval_suite/plots/aggregate_summary.md
pareto plots: 4 SVGs
replan timelines: 12 SVGs plus timeline_manifest.csv
```

Aggregate table:

| Label | Env | Seeds | Success | Return | NFE/action | Replan Rate | Learned Replan Rate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| fixed_chunk | lift | 0,1,2 | 0.000 | 0.000 | 5.000 | 0.250 | 0.000 |
| denoise_only | lift | 0,1,2 | 0.000 | 0.000 | 1.547 | 0.250 | 0.000 |
| fixed_chunk | can | 0,1,2 | 0.000 | 0.000 | 5.000 | 0.250 | 0.000 |
| denoise_only | can | 0,1,2 | 0.000 | 0.000 | 1.827 | 0.250 | 0.000 |

Denoise-only budget distribution over replans:

| Env | Total Replans | Mean Steps/Replan | 4 | 8 | 12 | 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lift | 300 | 6.187 | 230 | 26 | 19 | 25 |
| can | 300 | 7.307 | 201 | 34 | 23 | 42 |

Interpretation: denoise-only now has multi-seed Lift/Can software-path evidence and produces the required plots/timelines. The short 80-step horizon is too weak for task-performance conclusions because both fixed_chunk and denoise_only have zero raw return/success. The useful result is cost accounting: denoise-only preserved fixed interval replanning while reducing NFE/action by about `69.1%` on Lift and `63.5%` on Can relative to fixed_chunk under the same short-suite protocol. Full-horizon Lift/Can and Square runtime artifacts are still required before deciding whether denoise-only belongs in the final audited table.

## M6c Multi-Suite Final Audit Merge

M6c closes a tooling gap created by adding denoise-only after the original full-horizon Lift/Can suites were already generated. `scripts/audit_final_evidence.py` now accepts multiple `--suite ENV=...` values for the same environment and audits their env-specific rows together. `scripts/build_final_report.py` reads aggregate summaries from every suite path, filters rows to the target environment, and deduplicates repeated labels by keeping the row with the larger `episodes_total`. This allows a future denoise-only full-horizon suite to be added to the final report without rerunning already valid fixed/TD/PPO suites.

Verification:

```bash
conda run -n d3p python -m py_compile scripts/audit_final_evidence.py scripts/build_final_report.py tests/test_final_evidence_audit.py tests/test_final_report.py
conda run -n d3p python -m pytest -q tests/test_final_evidence_audit.py tests/test_final_report.py tests/test_final_evidence_pipeline.py
conda run -n d3p python scripts/run_final_evidence_pipeline.py \
  --envs robomimic_lift,robomimic_can \
  --min_episodes 5 \
  --suite robomimic_lift=/tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite \
  --suite robomimic_lift=/tmp/d3p_m6b_denoise_only_short_suite/20260529_072508_robomimic_lift_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m6b_denoise_only_short_suite/20260529_072508_robomimic_lift_robomimic_can_final_eval_suite \
  --expected_label robomimic_lift:fixed_chunk \
  --expected_label robomimic_lift:td_error_p95 \
  --expected_label robomimic_lift:lift_bc_p90_reg02 \
  --expected_label robomimic_lift:denoise_only \
  --expected_label robomimic_can:fixed_chunk \
  --expected_label robomimic_can:td_error_p95 \
  --expected_label robomimic_can:can_bc_p90_reg02 \
  --expected_label robomimic_can:denoise_only \
  --readiness_json /tmp/d3p_m5r_generated_readiness_pipeline/20260529_065609_final_evidence_pipeline/readiness/readiness.json \
  --output_dir /tmp/d3p_m6c_multi_suite_pipeline_check
```

Tooling check output:

```text
pytest: 12 passed in 2.07s
pipeline dir: /tmp/d3p_m6c_multi_suite_pipeline_check/20260529_074131_final_evidence_pipeline
all_goal_evidence_ready: true
all_suites_ready: true
readiness_ready: true
markdown_ready: true
matrix rows after duplicate-label merge: 8
```

Merged matrix check:

| Env | Label | Episodes Total | NFE/action | Source Suite |
| --- | --- | ---: | ---: | --- |
| lift | fixed_chunk | 300 | 5.000 | M5h Lift full-horizon suite |
| lift | lift_bc_p90_reg02 | 300 | 6.259 | M5h Lift full-horizon suite |
| lift | td_error_p95 | 300 | 8.331 | M5h Lift full-horizon suite |
| lift | denoise_only | 15 | 1.547 | M6b short denoise-only suite |
| can | can_bc_p90_reg02 | 300 | 12.338 | M5i Can full-horizon suite |
| can | fixed_chunk | 300 | 5.000 | M5i Can full-horizon suite |
| can | td_error_p95 | 300 | 8.846 | M5i Can full-horizon suite |
| can | denoise_only | 15 | 1.827 | M6b short denoise-only suite |

Interpretation: this is a tooling verification, not a final Lift/Can claim, because `--min_episodes 5` allows the short M6b denoise-only rows to pass. The important result is structural: final evidence can now merge additive method suites per environment, preserve the `source_suite` for each row, and avoid duplicate fixed_chunk rows when a short method suite includes a reference baseline.

## M6d Full-Horizon Denoise-Only Lift/Can Suite

M6d runs the denoise-only ablation at the same full 300-step horizon and 3 seeds x 100 episodes per environment used by the M5h/M5i final Lift/Can suites. This removes the main limitation of M6b: the denoise-only rows are now comparable with the audited full-horizon fixed, TD-error, and PPO rows for Lift and Can. Square remains excluded because local Square checkpoint and normalization artifacts are still missing.

Command:

```bash
conda run -n d3p python scripts/run_final_eval_suite.py \
  --envs robomimic_lift,robomimic_can \
  --seeds 0,1,2 \
  --episodes 100 \
  --baseline_methods denoise_only \
  --td_artifact robomimic_lift:/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json \
  --td_artifact robomimic_can:/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_threshold_calibration.json \
  --timeline_episodes 1 \
  --output_root /tmp/d3p_m6d_denoise_only_full_suite
```

Suite output:

```text
suite dir: /tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite
num_jobs: 6
num_completed: 6
summary_table: /tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite/plots/summary_table.md
aggregate_summary: /tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite/plots/aggregate_summary.md
pareto plots: 4 SVGs
replan timelines: 6 SVGs plus timeline_manifest.csv
```

Per-seed results:

| Label | Env | Seed | Episodes | Success | Return | NFE/action | Replan Rate | Learned Replan Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| denoise_only | lift | 0 | 100 | 0.060 | 1.400 | 1.717 | 0.250 | 0.000 |
| denoise_only | lift | 1 | 100 | 0.110 | 3.270 | 1.728 | 0.250 | 0.000 |
| denoise_only | lift | 2 | 100 | 0.130 | 5.770 | 1.740 | 0.250 | 0.000 |
| denoise_only | can | 0 | 100 | 0.040 | 3.070 | 1.725 | 0.250 | 0.000 |
| denoise_only | can | 1 | 100 | 0.020 | 0.630 | 1.773 | 0.250 | 0.000 |
| denoise_only | can | 2 | 100 | 0.010 | 0.920 | 1.668 | 0.250 | 0.000 |

Aggregate table:

| Label | Env | Seeds | Episodes Total | Success | Success SEM | Return | NFE/action | Replan Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| denoise_only | lift | 0,1,2 | 300 | 0.100 | 0.021 | 3.480 | 1.728 | 0.250 |
| denoise_only | can | 0,1,2 | 300 | 0.023 | 0.009 | 1.540 | 1.722 | 0.250 |

Denoise-only budget distribution over replans:

| Env | Total Replans | Mean Steps/Replan | 4 | 8 | 12 | 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lift | 22500 | 6.914 | 16129 | 2063 | 1453 | 2855 |
| can | 22500 | 6.888 | 15977 | 2270 | 1518 | 2735 |

Multi-suite final pipeline check:

```bash
conda run -n d3p python scripts/run_final_evidence_pipeline.py \
  --envs robomimic_lift,robomimic_can \
  --min_episodes 100 \
  --suite robomimic_lift=/tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite \
  --suite robomimic_lift=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --expected_label robomimic_lift:fixed_chunk \
  --expected_label robomimic_lift:td_error_p95 \
  --expected_label robomimic_lift:lift_bc_p90_reg02 \
  --expected_label robomimic_lift:denoise_only \
  --expected_label robomimic_can:fixed_chunk \
  --expected_label robomimic_can:td_error_p95 \
  --expected_label robomimic_can:can_bc_p90_reg02 \
  --expected_label robomimic_can:denoise_only \
  --readiness_json /tmp/d3p_m5r_generated_readiness_pipeline/20260529_065609_final_evidence_pipeline/readiness/readiness.json \
  --output_dir /tmp/d3p_m6d_multi_suite_pipeline_check
```

Pipeline output:

```text
pipeline dir: /tmp/d3p_m6d_multi_suite_pipeline_check/20260529_080422_final_evidence_pipeline
all_goal_evidence_ready: true
all_suites_ready: true
readiness_ready: true
markdown_ready: true
matrix rows after duplicate-label merge: 8
```

Merged full-horizon matrix:

| Env | Label | Episodes Total | Success | Return | NFE/action | Replan Rate | Source Suite |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| lift | fixed_chunk | 300 | 0.790 | 36.213 | 5.000 | 0.250 | M5h Lift full-horizon suite |
| lift | lift_bc_p90_reg02 | 300 | 0.753 | 30.710 | 6.259 | 0.313 | M5h Lift full-horizon suite |
| lift | td_error_p95 | 300 | 0.693 | 46.067 | 8.331 | 0.417 | M5h Lift full-horizon suite |
| lift | denoise_only | 300 | 0.100 | 3.480 | 1.728 | 0.250 | M6d denoise-only full-horizon suite |
| can | can_bc_p90_reg02 | 300 | 0.717 | 88.970 | 12.338 | 0.617 | M5i Can full-horizon suite |
| can | fixed_chunk | 300 | 0.703 | 82.053 | 5.000 | 0.250 | M5i Can full-horizon suite |
| can | td_error_p95 | 300 | 0.713 | 84.047 | 8.846 | 0.442 | M5i Can full-horizon suite |
| can | denoise_only | 300 | 0.023 | 1.540 | 1.722 | 0.250 | M6d denoise-only full-horizon suite |

Interpretation: denoise-only is now a full-horizon Lift/Can ablation rather than a short-suite placeholder. It is the lowest-compute method in both environments, using about `34.6%` of fixed_chunk NFE/action on Lift and `34.4%` on Can. That compute reduction is not free: full-horizon task performance collapses relative to fixed_chunk (`0.100` vs `0.790` Lift success, `0.023` vs `0.703` Can success). The current rule-based uncertainty-to-denoise mapping is therefore useful negative evidence for dynamic denoising by itself: it can reduce sampling cost substantially, but the aggressive low-step budget degrades raw task performance enough that it should not be presented as a Pareto improvement.

## M6e Square Readiness Denoise-Only Coverage

M6e closes a small but important planning gap for the remaining Square work. After M6d, the final Square sequence should not stop at `fixed_chunk` and `td_error_replan`; it also needs to run `denoise_only` so the final evidence table can include the same dynamic-denoising ablation shape as Lift/Can. `scripts/check_final_readiness.py` now emits `--baseline_methods fixed_chunk,td_error_replan,denoise_only` in the generated Square `run_final_suite` command.

Implementation change:

```text
scripts/check_final_readiness.py: Square final-suite command now includes denoise_only.
tests/test_final_readiness.py: asserts the generated Square final-suite command contains fixed_chunk, td_error_replan, and denoise_only.
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_final_readiness.py
conda run -n d3p python -m pytest -q tests/test_final_evidence_pipeline.py tests/test_final_eval_suite.py
conda run -n d3p python scripts/check_final_readiness.py \
  --envs robomimic_lift,robomimic_can,robomimic_square \
  --td_artifact robomimic_lift:/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json \
  --td_artifact robomimic_can:/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_threshold_calibration.json \
  --ppo_checkpoint robomimic_lift:lift_bc_p90_reg02=/tmp/d3p_m4s_lift_bc_p90_reg02_ppo/20260529_030050_robomimic_lift_ppo_replan_only_seed0/checkpoints/best.pt \
  --ppo_checkpoint robomimic_can:can_bc_p90_reg02=/tmp/d3p_m5e_can_bc_p90_reg02_ppo/20260529_035456_robomimic_can_ppo_replan_only_seed0/checkpoints/best.pt \
  --episodes 100 \
  --seeds 0,1,2 \
  --output_dir /tmp/d3p_m6e_square_readiness_valid_lift_can
```

Verification output:

```text
tests/test_final_readiness.py: 3 passed in 0.90s
pipeline/final-suite adjacent tests: 11 passed in 3.50s
readiness dir: /tmp/d3p_m6e_square_readiness_valid_lift_can/20260529_081129_final_readiness
all_final_eval_ready: false
robomimic_lift: runtime_ready=true, final_eval_ready=true, missing=none
robomimic_can: runtime_ready=true, final_eval_ready=true, missing=none
robomimic_square: runtime_ready=false, final_eval_ready=false, missing=base_policy_checkpoint, normalization, td_critic, td_calibration, ppo_checkpoint
Square run_final_suite includes: --baseline_methods fixed_chunk,td_error_replan,denoise_only
```

Interpretation: this does not make Square runnable; it makes the future Square close-out command complete for the M6 denoise-only ablation once real Square runtime artifacts exist. The local artifact search still only found zero-byte Square placeholders at `/tmp/d3p_m5k_square_runtime_artifacts/square_policy.pt` and `/tmp/d3p_m5k_square_runtime_artifacts/normalization.npz`, and the readiness checker correctly rejects them as invalid.

## M6f Final Expected-Label Preset

M6f closes the matching audit-side gap after M6e. The readiness command can now run Square `denoise_only`, but the final evidence pipeline previously relied on repeated manual `--expected_label` arguments. If a final command forgot `robomimic_square:denoise_only`, the audit would not require that row. `scripts/run_final_evidence_pipeline.py` now supports `--expected_label_preset final_full`, which expands the expected rows for each requested environment:

```text
robomimic_lift: fixed_chunk, td_error_p95, lift_bc_p90_reg02, denoise_only
robomimic_can: fixed_chunk, td_error_p95, can_bc_p90_reg02, denoise_only
robomimic_square: fixed_chunk, td_error_p95, square_bc_p90_reg02, denoise_only
```

Implementation change:

```text
scripts/run_final_evidence_pipeline.py: adds final_full expected-label preset and merges it with explicit labels without duplicates.
tests/test_final_evidence_pipeline.py: asserts final_full fails when denoise_only/PPO/TD rows are absent from a suite.
```

Verification:

```bash
conda run -n d3p python -m py_compile scripts/run_final_evidence_pipeline.py tests/test_final_evidence_pipeline.py
conda run -n d3p python -m pytest -q tests/test_final_evidence_pipeline.py
conda run -n d3p python -m pytest -q tests/test_final_evidence_audit.py tests/test_final_report.py tests/test_final_evidence_pipeline.py
conda run -n d3p python scripts/run_final_evidence_pipeline.py \
  --envs robomimic_lift,robomimic_can \
  --min_episodes 100 \
  --suite robomimic_lift=/tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite \
  --suite robomimic_lift=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --expected_label_preset final_full \
  --readiness_json /tmp/d3p_m6e_square_readiness_valid_lift_can/20260529_081129_final_readiness/readiness.json \
  --output_dir /tmp/d3p_m6f_expected_label_preset_lift_can
conda run -n d3p python scripts/run_final_evidence_pipeline.py \
  --envs robomimic_lift,robomimic_can,robomimic_square \
  --min_episodes 100 \
  --suite robomimic_lift=/tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite \
  --suite robomimic_lift=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --expected_label_preset final_full \
  --readiness_json /tmp/d3p_m6e_square_readiness_valid_lift_can/20260529_081129_final_readiness/readiness.json \
  --output_dir /tmp/d3p_m6f_expected_label_preset_full_gate \
  --require_complete
```

Verification output:

```text
tests/test_final_evidence_pipeline.py: 5 passed in 2.05s
final audit/report/pipeline tests: 13 passed in 2.12s
Lift/Can preset pipeline: /tmp/d3p_m6f_expected_label_preset_lift_can/20260529_081716_final_evidence_pipeline, all_goal_evidence_ready=true, matrix rows=8
Full gate with Square: /tmp/d3p_m6f_expected_label_preset_full_gate/20260529_081732_final_evidence_pipeline, exit=1 as expected
Full gate Square missing rows include fixed_chunk, td_error_p95, square_bc_p90_reg02, and denoise_only for seeds 0,1,2
Full gate failed checks: robomimic_square suite_provided; readiness_ready=false
```

Interpretation: `final_full` makes the completion gate stricter and less error-prone. Lift/Can pass with their merged M5h/M5i + M6d suites, including denoise-only rows. The full three-task gate still correctly fails because Square runtime artifacts and Square suite outputs are missing, and it now explicitly shows that Square `denoise_only` is one of the missing expected rows.

## M6g Default Markdown Gate Tightening

M6g tightens the final audit default markdown-token gate. Before this change, `scripts/audit_final_evidence.py` required the experiment log to mention the M6b short denoise-only suite, but it did not require the later full-horizon denoise-only evidence or the final expected-label preset. The default token set now also requires:

```text
M6d Full-Horizon Denoise-Only Lift/Can Suite
M6f Final Expected-Label Preset
```

Implementation change:

```text
scripts/audit_final_evidence.py: DEFAULT_MARKDOWN_TOKENS includes M6d and M6f sections.
tests/test_final_evidence_audit.py: asserts old markdown content fails when M6d/M6f tokens are absent.
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_final_evidence_audit.py tests/test_final_evidence_pipeline.py
conda run -n d3p python scripts/run_final_evidence_pipeline.py \
  --envs robomimic_lift,robomimic_can \
  --min_episodes 100 \
  --suite robomimic_lift=/tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite \
  --suite robomimic_lift=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --expected_label_preset final_full \
  --readiness_json /tmp/d3p_m6e_square_readiness_valid_lift_can/20260529_081129_final_readiness/readiness.json \
  --output_dir /tmp/d3p_m6g_default_markdown_tokens_lift_can
conda run -n d3p python -m pytest -q tests
```

Verification output:

```text
focused audit/pipeline tests: 10 passed in 2.06s
Lift/Can default-token pipeline: /tmp/d3p_m6g_default_markdown_tokens_lift_can/20260529_082209_final_evidence_pipeline
Lift/Can default-token pipeline status: all_goal_evidence_ready=true, markdown_ready=true, matrix rows=8
tests: 103 passed in 9.85s
```

Interpretation: the default final gate now checks that the markdown analysis contains the current full-horizon denoise-only evidence and the final expected-label preset documentation. This is still not Square completion, but it removes another way the final goal could be marked complete with stale experiment notes.

## M6h Final-Full Default Gate

M6h makes the stricter completion behavior the default. M6f added `--expected_label_preset final_full`, but the CLI default remained `none`, so a final pipeline command could still omit the preset and avoid checking expected fixed, TD-error, PPO, and denoise-only rows. `scripts/run_final_evidence_pipeline.py` now defaults to `final_full`; `--expected_label_preset none` remains available for narrow diagnostics.

Implementation change:

```text
scripts/run_final_evidence_pipeline.py: expected_label_preset default changes from none to final_full.
tests/test_final_evidence_pipeline.py: parser test asserts the CLI default is final_full.
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_final_evidence_pipeline.py
conda run -n d3p python scripts/run_final_evidence_pipeline.py \
  --envs robomimic_lift,robomimic_can \
  --min_episodes 100 \
  --suite robomimic_lift=/tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite \
  --suite robomimic_lift=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --readiness_json /tmp/d3p_m6e_square_readiness_valid_lift_can/20260529_081129_final_readiness/readiness.json \
  --output_dir /tmp/d3p_m6h_default_final_full_lift_can
conda run -n d3p python -m pytest -q tests
```

Verification output:

```text
tests/test_final_evidence_pipeline.py: 6 passed in 2.06s
Lift/Can default-final_full pipeline: /tmp/d3p_m6h_default_final_full_lift_can/20260529_082556_final_evidence_pipeline
Lift/Can default-final_full status: all_goal_evidence_ready=true, markdown_ready=true, matrix rows=8
tests: 104 passed in 9.89s
```

Interpretation: the final pipeline now fails closed by default on missing expected rows. The eventual close-out command no longer has to remember `--expected_label_preset final_full`; it is still documented explicitly for clarity, but the default behavior already requires the full row set for any requested Robomimic env.

## M6i Final-Eval Default Baselines

M6i aligns the suite producer with the stricter final gate. After M6h, `scripts/run_final_evidence_pipeline.py` expects denoise-only rows by default, but `scripts/run_final_eval_suite.py` still defaulted to only `fixed_chunk,td_error_replan`. That mismatch meant a user could generate a default suite that would not satisfy the default final gate. The final eval suite runner now defaults to:

```text
fixed_chunk,td_error_replan,denoise_only
```

`--baseline_methods fixed_chunk,td_error_replan` or `--baseline_methods ""` still allow narrower diagnostics when needed.

Implementation change:

```text
scripts/run_final_eval_suite.py: default baseline_methods includes denoise_only.
tests/test_final_eval_suite.py: parser test asserts the default baseline methods include fixed_chunk, td_error_replan, and denoise_only.
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_final_eval_suite.py
conda run -n d3p python scripts/run_final_eval_suite.py \
  --envs robomimic_lift \
  --seeds 0 \
  --episodes 1 \
  --max_episode_steps 5 \
  --td_artifact robomimic_lift:/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json \
  --dry_run \
  --output_root /tmp/d3p_m6i_final_eval_default_dry_run
conda run -n d3p python -m pytest -q tests
```

Verification output:

```text
tests/test_final_eval_suite.py: 8 passed in 2.00s
dry-run suite: /tmp/d3p_m6i_final_eval_default_dry_run/20260529_083044_robomimic_lift_final_eval_suite
dry-run jobs: 3
labels: fixed_chunk, td_error_p95, denoise_only
tests: 105 passed in 9.89s
```

Interpretation: final-suite generation and final-suite auditing now agree by default. Once Square runtime artifacts exist, the default suite runner will generate the denoise-only baseline row needed by the default final evidence pipeline. Square itself remains blocked by missing real Square artifacts.

## M6j Generated Final-Pipeline Close-Out Command

M6j extends the Square readiness report beyond data collection and suite generation. Before this change, `scripts/check_final_readiness.py` generated commands through `run_final_suite`, but the final audit/report close-out still had to be assembled manually. The Square readiness report now also emits `run_final_pipeline`, a command that calls `scripts/run_final_evidence_pipeline.py` with `--generate_readiness`, `--require_complete`, suite placeholders for every requested environment, and the Square runtime/TD/PPO artifacts needed after the Square suite has been generated.

Implementation change:

```text
scripts/check_final_readiness.py: adds run_final_pipeline to Square next_commands.
tests/test_final_readiness.py: asserts the generated command includes run_final_evidence_pipeline.py, --generate_readiness, --require_complete, Square suite placeholder, runtime artifact, and PPO checkpoint.
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_final_readiness.py tests/test_final_evidence_pipeline.py
conda run -n d3p python scripts/check_final_readiness.py \
  --envs robomimic_lift,robomimic_can,robomimic_square \
  --td_artifact robomimic_lift:/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json \
  --td_artifact robomimic_can:/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_threshold_calibration.json \
  --ppo_checkpoint robomimic_lift:lift_bc_p90_reg02=/tmp/d3p_m4s_lift_bc_p90_reg02_ppo/20260529_030050_robomimic_lift_ppo_replan_only_seed0/checkpoints/best.pt \
  --ppo_checkpoint robomimic_can:can_bc_p90_reg02=/tmp/d3p_m5e_can_bc_p90_reg02_ppo/20260529_035456_robomimic_can_ppo_replan_only_seed0/checkpoints/best.pt \
  --episodes 100 \
  --seeds 0,1,2 \
  --output_dir /tmp/d3p_m6j_square_readiness_pipeline_command
conda run -n d3p python -m pytest -q tests
```

Verification output:

```text
readiness/final-pipeline tests: 9 passed in 2.06s
readiness dir: /tmp/d3p_m6j_square_readiness_pipeline_command/20260529_083810_final_readiness
run_final_pipeline includes: scripts/run_final_evidence_pipeline.py, --generate_readiness, --require_complete
run_final_pipeline suites: robomimic_lift=<robomimic_lift_final_eval_suite_dir>, robomimic_can=<robomimic_can_final_eval_suite_dir>, robomimic_square=/tmp/d3p_square_final/final_suite/<square_final_eval_suite_dir>
tests: 105 passed in 9.89s
```

Interpretation: the readiness artifact now gives the complete post-artifact command chain: collect Square transitions, train/calibrate Square TD, train Square PPO, run Square final suite, then run the final evidence pipeline. The final command still cannot succeed until real Square runtime artifacts and suite outputs exist.

## M6k Env-Meta Forwarding In Generated Close-Out Commands

M6k fixes a robustness gap in the M6j generated commands. `scripts/check_final_readiness.py --env_meta_path ...` already affected the fixed-transition and PPO-training commands, but the generated `run_final_suite` command did not pass `--env_meta_path`, and the generated `run_final_pipeline` command did not forward the override into its internal `--generate_readiness` step. If Square requires a non-default env metadata file, the close-out chain could silently switch back to the default env metadata at the final suite or final audit stage.

Implementation change:

```text
scripts/check_final_readiness.py: run_final_suite includes --env_meta_path for Square.
scripts/check_final_readiness.py: run_final_pipeline forwards --env_meta_path when the readiness command received one.
tests/test_final_readiness.py: asserts both generated commands include the explicit env meta path.
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_final_readiness.py tests/test_final_evidence_pipeline.py tests/test_final_eval_suite.py
conda run -n d3p python scripts/check_final_readiness.py \
  --envs robomimic_square \
  --env_meta_path cfg/robomimic/env_meta/square.json \
  --episodes 100 \
  --seeds 0,1,2 \
  --output_dir /tmp/d3p_m6k_square_env_meta_forwarding
conda run -n d3p python -m pytest -q tests
```

Verification output:

```text
focused readiness/final-suite tests: 17 passed in 3.50s
readiness dir: /tmp/d3p_m6k_square_env_meta_forwarding/20260529_084508_final_readiness
run_final_suite includes: --env_meta_path cfg/robomimic/env_meta/square.json
run_final_pipeline includes: --env_meta_path cfg/robomimic/env_meta/square.json
tests: 105 passed in 9.86s
```

Interpretation: generated Square close-out commands now preserve runtime env-meta overrides through collection, PPO training, final suite generation, and final audit/report generation. This still does not remove the real Square artifact blocker.

## M6l Default Markdown Gate Refresh

M6l refreshes the final audit default markdown-token gate after the M6h-M6k close-out hardening work. Before this change, the default gate required M6d and M6f but did not require the later sections that make `final_full` the default, align final-suite baseline defaults, generate the final pipeline close-out command, and preserve env-meta overrides. The default token set now also requires:

```text
M6h Final-Full Default Gate
M6i Final-Eval Default Baselines
M6j Generated Final-Pipeline Close-Out Command
M6k Env-Meta Forwarding In Generated Close-Out Commands
```

Implementation change:

```text
scripts/audit_final_evidence.py: DEFAULT_MARKDOWN_TOKENS includes M6h-M6k sections.
tests/test_final_evidence_audit.py: asserts markdown missing M6h-M6k fails the default token audit.
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_final_evidence_audit.py tests/test_final_evidence_pipeline.py
conda run -n d3p python scripts/run_final_evidence_pipeline.py \
  --envs robomimic_lift,robomimic_can \
  --min_episodes 100 \
  --suite robomimic_lift=/tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite \
  --suite robomimic_lift=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --readiness_json /tmp/d3p_m6e_square_readiness_valid_lift_can/20260529_081129_final_readiness/readiness.json \
  --output_dir /tmp/d3p_m6l_default_markdown_tokens_lift_can
conda run -n d3p python -m pytest -q tests
```

Verification output:

```text
focused audit/pipeline tests: 11 passed in 2.07s
Lift/Can default-token pipeline: /tmp/d3p_m6l_default_markdown_tokens_lift_can/20260529_084930_final_evidence_pipeline
Lift/Can default-token pipeline status: all_goal_evidence_ready=true, markdown_ready=true, matrix rows=8
tests: 105 passed in 9.86s
```

Interpretation: the default final gate now requires the experiment log to include the current close-out mechanics through M6k, not just the earlier denoise-only and expected-label preset sections. This still does not remove the real Square artifact blocker.

## M6m Pipeline Summary Blocker Details

M6m makes the final evidence pipeline summary directly actionable for the remaining Square close-out. Before this change, `pipeline_summary.json` and `pipeline_summary.md` only listed suite readiness and failed suite checks per environment. To see which Square labels/seeds or artifacts were still missing, the next step had to inspect the nested audit and readiness outputs. The pipeline summary now records each environment's `suite_paths`, `missing_rows`, and `readiness_missing` entries.

Implementation change:

```text
scripts/run_final_evidence_pipeline.py: per-env summary entries include suite_paths, missing_rows, readiness_missing, and the Markdown summary table prints the same blockers.
tests/test_final_evidence_pipeline.py: asserts missing Square rows and readiness blockers remain visible in the top-level pipeline summary.
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_final_evidence_pipeline.py
conda run -n d3p python scripts/run_final_evidence_pipeline.py \
  --envs robomimic_lift,robomimic_can,robomimic_square \
  --min_episodes 100 \
  --suite robomimic_lift=/tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite \
  --suite robomimic_lift=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --readiness_json /tmp/d3p_m6e_square_readiness_valid_lift_can/20260529_081129_final_readiness/readiness.json \
  --output_dir /tmp/d3p_m6m_summary_blockers_full_gate
conda run -n d3p python -m pytest -q tests
```

Verification output:

```text
focused pipeline tests: 6 passed in 2.07s
full gate diagnostic pipeline: /tmp/d3p_m6m_summary_blockers_full_gate/20260529_090252_final_evidence_pipeline
full gate status: all_goal_evidence_ready=false, all_suites_ready=false, readiness_ready=false, markdown_ready=true
Lift/Can summary blockers: missing_rows=[], readiness_missing=[]
Square summary blockers: 12 missing rows (fixed_chunk, td_error_p95, square_bc_p90_reg02, denoise_only for seeds 0,1,2); readiness_missing=base_policy_checkpoint, normalization, td_critic, td_calibration, ppo_checkpoint
tests: 105 passed in 9.87s
```

Interpretation: the top-level final-pipeline output now identifies the exact Square rows and artifacts that still block completion. This does not make Square runnable, but it shortens the final artifact handoff loop because `pipeline_summary.json` alone shows what remains to satisfy the completion gate.

## M6n Default Markdown Gate Includes Summary Blockers

M6n keeps the final evidence audit aligned with M6m. The default markdown gate now requires `M6m Pipeline Summary Blocker Details`, so a final audit cannot pass with an experiment log that omits the top-level blocker-summary behavior added in M6m.

Implementation change:

```text
scripts/audit_final_evidence.py: DEFAULT_MARKDOWN_TOKENS includes M6m Pipeline Summary Blocker Details.
tests/test_final_evidence_audit.py: verifies markdown that includes M6h-M6k but omits M6m fails only on the M6m token.
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_final_evidence_audit.py tests/test_final_evidence_pipeline.py
conda run -n d3p python scripts/run_final_evidence_pipeline.py \
  --envs robomimic_lift,robomimic_can \
  --min_episodes 100 \
  --suite robomimic_lift=/tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite \
  --suite robomimic_lift=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --readiness_json /tmp/d3p_m6e_square_readiness_valid_lift_can/20260529_081129_final_readiness/readiness.json \
  --output_dir /tmp/d3p_m6n_default_markdown_tokens_lift_can
conda run -n d3p python -m pytest -q tests
```

Verification output:

```text
focused audit/pipeline tests: 11 passed in 2.07s
Lift/Can default-token pipeline: /tmp/d3p_m6n_default_markdown_tokens_lift_can/20260529_090922_final_evidence_pipeline
Lift/Can default-token pipeline status: all_goal_evidence_ready=true, all_suites_ready=true, readiness_ready=true, markdown_ready=true, missing_rows=[]
tests: 105 passed in 9.89s
```

Interpretation: the default final audit now enforces that the experiment log includes the current top-level blocker-summary behavior. Lift/Can still pass the default final_full gate with M6m documented; Square remains excluded from this pass because its real artifacts and suite rows are still missing.

## M6o Final Report Blocker Details

M6o carries the M6m blocker detail into the generated final analysis report. Before this change, `pipeline_summary.json` listed exact missing Square rows and readiness artifacts, but `final_report.json` and `final_report.md` summarized Square only as `missing:suite_provided`. The generated final report now includes per-environment `missing_rows`, `readiness_missing`, and `failed_checks` in `env_summaries`, and the Markdown environment table prints the same blocker columns. The default markdown gate also requires this M6o section.

Implementation change:

```text
scripts/build_final_report.py: env_summaries include missing_rows, readiness_missing, and failed_checks; final_report.md prints these blockers.
tests/test_final_report.py: asserts Square blocker details appear in final_report.json-derived summaries and generated Markdown.
scripts/audit_final_evidence.py: DEFAULT_MARKDOWN_TOKENS includes M6o Final Report Blocker Details.
tests/test_final_evidence_audit.py: verifies markdown that includes M6m but omits M6o fails only on the M6o token.
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_final_report.py tests/test_final_evidence_pipeline.py
conda run -n d3p python scripts/run_final_evidence_pipeline.py \
  --envs robomimic_lift,robomimic_can,robomimic_square \
  --min_episodes 100 \
  --suite robomimic_lift=/tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite \
  --suite robomimic_lift=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --readiness_json /tmp/d3p_m6e_square_readiness_valid_lift_can/20260529_081129_final_readiness/readiness.json \
  --output_dir /tmp/d3p_m6o_final_report_blockers_full_gate
conda run -n d3p python -m pytest -q tests/test_final_evidence_audit.py tests/test_final_report.py tests/test_final_evidence_pipeline.py
conda run -n d3p python -m pytest -q tests
```

Verification output:

```text
focused report/pipeline tests: 11 passed in 2.06s
focused audit/report/pipeline tests: 16 passed in 2.08s
full gate diagnostic pipeline: /tmp/d3p_m6o_final_report_blockers_full_gate/20260529_091635_final_evidence_pipeline
full gate status: all_goal_evidence_ready=false, all_suites_ready=false, readiness_ready=false, markdown_ready=true
final_report blockers: Square missing_rows includes fixed_chunk, td_error_p95, square_bc_p90_reg02, and denoise_only for seeds 0,1,2; readiness_missing=base_policy_checkpoint, normalization, td_critic, td_calibration, ppo_checkpoint
final_report markdown blocker row: Readiness Missing and Missing Rows columns contain the same Square blockers
tests: 106 passed in 9.91s
```

Interpretation: the generated final analysis report now preserves the same Square blocker detail as the pipeline summary, so the report artifact is useful for both metric comparison and close-out diagnosis. This still does not make Square runnable; it makes the missing evidence explicit in the final report until real Square artifacts and suites exist.

## M6p Duplicate Suite Path Guard

M6p prevents accidental duplicate suite inputs from polluting final audit/report accounting. The final audit still supports multiple suites for the same environment, which is required to merge base fixed/TD/PPO suites with denoise-only suites, but it now rejects the exact same suite path repeated for the same environment. This matters because a duplicated `--suite ENV=/path` can inflate row/check counts and make copied final commands harder to audit.

Implementation change:

```text
scripts/audit_final_evidence.py: env_suite_map tracks normalized `(env, suite_path)` pairs and raises on exact duplicates.
tests/test_final_evidence_audit.py: asserts duplicate suite paths are rejected while the existing multi-suite merge test continues to pass.
scripts/audit_final_evidence.py: DEFAULT_MARKDOWN_TOKENS includes M6p Duplicate Suite Path Guard.
tests/test_final_evidence_audit.py: verifies markdown that includes M6o but omits M6p fails only on the M6p token.
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_final_evidence_audit.py tests/test_final_evidence_pipeline.py tests/test_final_report.py
conda run -n d3p python scripts/run_final_evidence_pipeline.py \
  --envs robomimic_lift \
  --min_episodes 100 \
  --suite robomimic_lift=/tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite \
  --suite robomimic_lift=/tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite \
  --readiness_json /tmp/d3p_m6e_square_readiness_valid_lift_can/20260529_081129_final_readiness/readiness.json \
  --output_dir /tmp/d3p_m6p_duplicate_suite_guard
conda run -n d3p python scripts/run_final_evidence_pipeline.py \
  --envs robomimic_lift,robomimic_can \
  --min_episodes 100 \
  --suite robomimic_lift=/tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite \
  --suite robomimic_lift=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --readiness_json /tmp/d3p_m6e_square_readiness_valid_lift_can/20260529_081129_final_readiness/readiness.json \
  --output_dir /tmp/d3p_m6p_valid_multi_suite_lift_can
conda run -n d3p python -m pytest -q tests
```

Verification output:

```text
focused audit/pipeline/report tests: 17 passed in 2.07s
duplicate-suite pipeline: exit=1 as expected; ValueError duplicate suite path for robomimic_lift
valid multi-suite Lift/Can pipeline: /tmp/d3p_m6p_valid_multi_suite_lift_can/20260529_092549_final_evidence_pipeline
valid multi-suite status: all_goal_evidence_ready=true, all_suites_ready=true, readiness_ready=true, markdown_ready=true, missing_rows=[]
tests: 107 passed in 9.93s
```

Interpretation: the final audit now distinguishes intended multi-suite merges from accidental duplicate suite arguments. The valid Lift/Can base+denoise merge still passes, while an exact repeated suite path fails before producing misleading final evidence artifacts.

## M6q Final Environment Summary CSV

M6q adds a compact environment-level CSV to the generated final report artifacts. Before this change, `final_results_matrix.csv` captured method-level rows and `final_report.json` / `final_report.md` captured env-level best-method and blocker summaries. The new `final_env_summary.csv` makes those env-level summaries directly usable in spreadsheet/Pandas analysis without parsing JSON or Markdown. It records `ready`, `num_methods`, best-success, best-return, lowest-compute labels and values, plus semicolon-separated `readiness_missing`, `missing_rows`, and `failed_checks`.

Implementation change:

```text
scripts/build_final_report.py: writes report/final_env_summary.csv from env_summaries without changing final_results_matrix.csv.
scripts/run_final_evidence_pipeline.py: exposes final_env_summary_csv in pipeline_summary.json and pipeline_summary.md.
tests/test_final_report.py: asserts the env summary CSV exists and contains Square blocker fields.
tests/test_final_evidence_pipeline.py: asserts the pipeline exposes and writes final_env_summary.csv.
scripts/audit_final_evidence.py: DEFAULT_MARKDOWN_TOKENS includes M6q Final Environment Summary CSV.
tests/test_final_evidence_audit.py: verifies markdown that includes M6p but omits M6q fails only on the M6q token.
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_final_report.py tests/test_final_evidence_pipeline.py
conda run -n d3p python -m pytest -q tests/test_final_evidence_audit.py tests/test_final_report.py tests/test_final_evidence_pipeline.py
conda run -n d3p python scripts/run_final_evidence_pipeline.py \
  --envs robomimic_lift,robomimic_can,robomimic_square \
  --min_episodes 100 \
  --suite robomimic_lift=/tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite \
  --suite robomimic_lift=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --readiness_json /tmp/d3p_m6e_square_readiness_valid_lift_can/20260529_081129_final_readiness/readiness.json \
  --output_dir /tmp/d3p_m6q_env_summary_csv_full_gate
conda run -n d3p python -m pytest -q tests
```

Verification output:

```text
focused report/pipeline tests: 11 passed in 2.08s
focused audit/report/pipeline tests: 17 passed in 2.09s
full gate diagnostic pipeline: /tmp/d3p_m6q_env_summary_csv_full_gate/20260529_093329_final_evidence_pipeline
full gate status: all_goal_evidence_ready=false, all_suites_ready=false, readiness_ready=false, markdown_ready=true
final_env_summary.csv: 3 rows; Can/Lift ready=true, Square ready=false; Square readiness_missing=base_policy_checkpoint, normalization, td_critic, td_calibration, ppo_checkpoint; Square missing_rows include fixed_chunk, td_error_p95, square_bc_p90_reg02, denoise_only for seeds 0,1,2
pipeline summary exposes final_env_summary_csv in JSON and Markdown
tests: 107 passed in 9.90s
```

Interpretation: the final analysis bundle now includes both method-level rows (`final_results_matrix.csv`) and environment-level status/blocker rows (`final_env_summary.csv`). This makes the current Lift/Can/Square state easier to load into a spreadsheet or Pandas without parsing the Markdown report.

## M6r Complete Final Matrix Metrics

M6r fills a data-preservation gap in the generated final analysis matrix. The source suite `aggregate_summary.csv` files already contain high-level return, wall-time/action, and SEM columns for replanning, learned replanning, and discarded actions, but `final_results_matrix.csv` previously emitted only a subset of those metrics. The matrix now preserves the full aggregate metric set used by the suite analysis: high-level return mean/SEM, total wall-time/action mean/SEM, replan-rate SEM, learned-replan SEM, and discarded-actions SEM are included alongside the existing success, return, NFE, replan, learned-replan, and discarded-action means.

Implementation change:

```text
scripts/build_final_report.py: REPORT_FIELDS and NUMERIC_FIELDS now include the missing aggregate metric columns; final_report.md result matrix prints high-level return and wall-time/action with SEMs.
tests/test_final_report.py: asserts report JSON rows, final_results_matrix.csv, and final_report.md preserve the added metrics.
scripts/audit_final_evidence.py: DEFAULT_MARKDOWN_TOKENS includes M6r Complete Final Matrix Metrics.
tests/test_final_evidence_audit.py: verifies markdown that includes M6q but omits M6r fails only on the M6r token.
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_final_report.py
conda run -n d3p python -m pytest -q tests/test_final_evidence_audit.py tests/test_final_report.py tests/test_final_evidence_pipeline.py
conda run -n d3p python scripts/run_final_evidence_pipeline.py \
  --envs robomimic_lift,robomimic_can,robomimic_square \
  --min_episodes 100 \
  --suite robomimic_lift=/tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite \
  --suite robomimic_lift=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --readiness_json /tmp/d3p_m6e_square_readiness_valid_lift_can/20260529_081129_final_readiness/readiness.json \
  --output_dir /tmp/d3p_m6r_complete_matrix_metrics_full_gate
conda run -n d3p python -m pytest -q tests
```

Verification output:

```text
focused final-report tests: 5 passed in 0.91s
focused audit/report/pipeline tests: 17 passed in 2.08s
full gate diagnostic pipeline: /tmp/d3p_m6r_complete_matrix_metrics_full_gate/20260529_093914_final_evidence_pipeline
full gate status: all_goal_evidence_ready=false, all_suites_ready=false, readiness_ready=false, markdown_ready=true
final_results_matrix.csv fields: includes mean_high_level_return_mean/sem, mean_total_wall_time_per_action_mean/sem, replan_rate_sem, learned_replan_rate_sem, mean_discarded_actions_per_episode_sem
sample matrix values: can_bc_p90_reg02 high_level_return=163.86917779156065, wall_time_per_action=0.0025864842500485895, learned_replan_rate_sem=0.006259777537166532, discarded_actions_sem=5.687120927538331
final_report.md result matrix: includes High-Level Return and Wall Time/action columns with mean +/- SEM values
tests: 107 passed in 9.91s
```

Interpretation: `final_results_matrix.csv` now preserves the same core metric columns as the suite aggregate summaries, so downstream analysis can use the final matrix without reopening each suite directory. The generated Markdown report also exposes high-level return and wall-time/action, making compute/performance tradeoffs easier to inspect.

## M6s Env Summary High-Level And Wall-Time Winners

M6s extends the environment-level final summary added in M6q. The summary already recorded best raw success, best raw task return, and lowest NFE/action. After M6r made high-level return and wall-time/action first-class method-level matrix columns, the env summary also needs to expose the corresponding per-environment winners. `env_summaries`, `final_env_summary.csv`, and the Environment Summary table now include best high-level return and lowest wall-time/action labels and values.

Implementation change:

```text
scripts/build_final_report.py: summarize_env adds best_high_level_return and lowest_wall_time; final_env_summary.csv and final_report.md print those fields.
tests/test_final_report.py: asserts JSON env_summaries, final_env_summary.csv, and final_report.md include the added winners.
scripts/audit_final_evidence.py: DEFAULT_MARKDOWN_TOKENS includes M6s Env Summary High-Level And Wall-Time Winners.
tests/test_final_evidence_audit.py: verifies markdown that includes M6r but omits M6s fails only on the M6s token.
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_final_report.py
conda run -n d3p python -m pytest -q tests/test_final_evidence_audit.py tests/test_final_report.py tests/test_final_evidence_pipeline.py
conda run -n d3p python scripts/run_final_evidence_pipeline.py \
  --envs robomimic_lift,robomimic_can,robomimic_square \
  --min_episodes 100 \
  --suite robomimic_lift=/tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite \
  --suite robomimic_lift=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --readiness_json /tmp/d3p_m6e_square_readiness_valid_lift_can/20260529_081129_final_readiness/readiness.json \
  --output_dir /tmp/d3p_m6s_env_summary_winners_full_gate
conda run -n d3p python -m pytest -q tests
```

Verification output:

```text
focused final-report tests: 5 passed in 0.92s
focused audit/report/pipeline tests: 17 passed in 2.07s
full gate diagnostic pipeline: /tmp/d3p_m6s_env_summary_winners_full_gate/20260529_094716_final_evidence_pipeline
full gate status: all_goal_evidence_ready=false, all_suites_ready=false, readiness_ready=false, markdown_ready=true
final_env_summary.csv winners: Can best_high_level_return=can_bc_p90_reg02, lowest_wall_time=denoise_only; Lift best_high_level_return=lift_bc_p90_reg02, lowest_wall_time=denoise_only; Square remains empty/missing
final_report.md env summary: includes Best High-Level Return and Lowest Wall Time/action columns
full tests: 107 passed in 9.92s
```

Interpretation: env-level artifacts now summarize raw task winners, high-level-return winners, NFE efficiency, and wall-time efficiency separately. This keeps high-level PPO reward signals visible without replacing raw task return as the main task metric.

## M6t Fixed Baseline Comparison Columns

M6t makes the final method-level matrix directly comparable against each environment's fixed-chunk baseline. `final_results_matrix.csv` now includes additive delta/ratio columns for success, raw return, high-level return, NFE/action, wall-time/action, replan rate, learned-replan rate, and discarded actions per episode. The generated Markdown report stays compact; the new comparison fields are preserved in machine-readable CSV/JSON output for downstream analysis.

Implementation change:

```text
scripts/build_final_report.py: annotates complete rows with same-env fixed_chunk comparison fields before report serialization.
tests/test_final_report.py: asserts JSON rows and final_results_matrix.csv include the fixed baseline deltas/ratios.
scripts/audit_final_evidence.py: DEFAULT_MARKDOWN_TOKENS includes M6t Fixed Baseline Comparison Columns.
tests/test_final_evidence_audit.py: verifies markdown that includes M6s but omits M6t fails only on the M6t token.
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_final_report.py
conda run -n d3p python -m pytest -q tests/test_final_evidence_audit.py tests/test_final_report.py tests/test_final_evidence_pipeline.py
conda run -n d3p python scripts/run_final_evidence_pipeline.py \
  --envs robomimic_lift,robomimic_can,robomimic_square \
  --min_episodes 100 \
  --suite robomimic_lift=/tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite \
  --suite robomimic_lift=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --readiness_json /tmp/d3p_m6e_square_readiness_valid_lift_can/20260529_081129_final_readiness/readiness.json \
  --output_dir /tmp/d3p_m6t_fixed_comparison_columns_full_gate
conda run -n d3p python -m pytest -q tests
```

Verification output:

```text
focused final-report tests: 5 passed in 0.91s
focused audit/report/pipeline tests: 17 passed in 2.08s
full gate diagnostic pipeline: /tmp/d3p_m6t_fixed_comparison_columns_full_gate/20260529_095607_final_evidence_pipeline
full gate status: all_goal_evidence_ready=false, all_suites_ready=false, readiness_ready=false, markdown_ready=true
final_results_matrix.csv comparison fields: success_delta_vs_fixed, return_delta_vs_fixed, high_level_return_delta_vs_fixed, nfe_ratio_vs_fixed, wall_time_ratio_vs_fixed, replan_rate_delta_vs_fixed, learned_replan_rate_delta_vs_fixed, discarded_actions_delta_vs_fixed
sample Lift comparisons: fixed_chunk success_delta=0.0, nfe_ratio=1.0; lift_bc_p90_reg02 success_delta=-0.036666666666666736, return_delta=-5.503333333333334, nfe_ratio=1.2517333333333334; denoise_only success_delta=-0.6900000000000001, nfe_ratio=0.34568
sample Can comparisons: fixed_chunk success_delta=0.0, nfe_ratio=1.0; can_bc_p90_reg02 success_delta=0.01333333333333342, return_delta=6.916666666666686, nfe_ratio=2.4676; denoise_only success_delta=-0.6799999999999999, nfe_ratio=0.3444088888888889
full tests: 107 passed in 9.90s
```

Interpretation: the final CSV/JSON matrix can now answer "how much better or more expensive is this method than fixed_chunk in the same environment" without reopening suite directories or hand-joining baseline rows.

## M6u Final Pipeline Readiness Details

M6u makes the final evidence pipeline summary self-contained when readiness is generated inside the pipeline. `pipeline_summary.json` now forwards each environment's readiness artifact checks and any generated `next_commands`; `pipeline_summary.md` prints the artifact labels, status/reason, paths, and the Square suggested command sequence. This keeps the final blocker diagnosis and the commands needed to close Square in the same artifact as the audit/report paths.

Implementation change:

```text
scripts/run_final_evidence_pipeline.py: forwards readiness_checks and next_commands into env summaries and prints them in pipeline_summary.md.
tests/test_final_evidence_pipeline.py: asserts generated-readiness Square summaries retain missing artifact reasons and suggested commands.
scripts/audit_final_evidence.py: DEFAULT_MARKDOWN_TOKENS includes M6u Final Pipeline Readiness Details.
tests/test_final_evidence_audit.py: verifies markdown that includes M6t but omits M6u fails only on the M6u token.
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_final_evidence_pipeline.py tests/test_final_evidence_audit.py
conda run -n d3p python scripts/run_final_evidence_pipeline.py \
  --envs robomimic_lift,robomimic_can,robomimic_square \
  --min_episodes 100 \
  --suite robomimic_lift=/tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite \
  --suite robomimic_lift=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --generate_readiness \
  --td_artifact robomimic_lift:/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json \
  --td_artifact robomimic_can:/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_threshold_calibration.json \
  --ppo_checkpoint robomimic_lift:lift_bc_p90_reg02=/tmp/d3p_m4s_lift_bc_p90_reg02_ppo/20260529_030050_robomimic_lift_ppo_replan_only_seed0/checkpoints/best.pt \
  --ppo_checkpoint robomimic_can:can_bc_p90_reg02=/tmp/d3p_m5e_can_bc_p90_reg02_ppo/20260529_035456_robomimic_can_ppo_replan_only_seed0/checkpoints/best.pt \
  --output_dir /tmp/d3p_m6u_pipeline_readiness_details_full_gate
conda run -n d3p python -m pytest -q tests
```

Verification output:

```text
focused pipeline/audit tests: 13 passed in 2.07s
generated-readiness diagnostic pipeline: /tmp/d3p_m6u_pipeline_readiness_details_full_gate/20260529_100458_final_evidence_pipeline
full gate status: all_goal_evidence_ready=false, all_suites_ready=false, readiness_ready=false, markdown_ready=true
Lift/Can readiness checks: all runtime, TD, calibration, and PPO artifacts valid when passed through --generate_readiness
Square readiness checks in pipeline_summary.json: base_policy_checkpoint=missing, normalization=missing, env_meta=ok, td_critic=not_provided, td_calibration=not_provided, ppo_checkpoint=not_provided
Square next_commands in pipeline_summary.json: collect_fixed_transitions, train_td_critic, calibrate_td_threshold, train_ppo_replan, run_final_suite, run_final_pipeline
pipeline_summary.md includes Readiness Artifact Details and Suggested Square sequence
full tests: 108 passed in 9.88s
```

Interpretation: the generated pipeline summary can now be used as the single handoff artifact for the remaining Square close-out: it shows which artifacts are missing and carries the exact generated command templates needed to produce them and rerun the final gate.

## M6v Final Close-Out Command With Known Suites

M6v turns the final pipeline summary into a more directly runnable close-out handoff. The generated `pipeline_summary.json` now includes a top-level `final_closeout_command`; `pipeline_summary.md` prints the same command in its own section. Unlike the readiness-only `run_final_pipeline` command, this command is built by the final evidence pipeline itself, so it preserves every suite path and artifact argument already supplied to the current run. In the current Lift/Can/Square diagnostic, the command carries the real Lift/Can final suite paths and leaves placeholders only for the still-missing Square final suite directory and Square PPO checkpoint.

Implementation change:

```text
scripts/run_final_evidence_pipeline.py: builds final_closeout_command from current envs, seeds, known --suite args, known TD/PPO/runtime artifacts, generated Square placeholders, and --require_complete.
tests/test_final_evidence_pipeline.py: asserts the generated command includes the known Lift suite path, excludes the Lift placeholder, keeps the Square suite placeholder, includes Square TD/PPO placeholders, and is printed in pipeline_summary.md.
scripts/audit_final_evidence.py: DEFAULT_MARKDOWN_TOKENS includes M6v Final Close-Out Command With Known Suites.
tests/test_final_evidence_audit.py: verifies markdown that includes M6u but omits M6v fails only on the M6v token.
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_final_evidence_pipeline.py tests/test_final_evidence_audit.py
conda run -n d3p python scripts/run_final_evidence_pipeline.py \
  --envs robomimic_lift,robomimic_can,robomimic_square \
  --min_episodes 100 \
  --suite robomimic_lift=/tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite \
  --suite robomimic_lift=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --generate_readiness \
  --td_artifact robomimic_lift:/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json \
  --td_artifact robomimic_can:/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_threshold_calibration.json \
  --ppo_checkpoint robomimic_lift:lift_bc_p90_reg02=/tmp/d3p_m4s_lift_bc_p90_reg02_ppo/20260529_030050_robomimic_lift_ppo_replan_only_seed0/checkpoints/best.pt \
  --ppo_checkpoint robomimic_can:can_bc_p90_reg02=/tmp/d3p_m5e_can_bc_p90_reg02_ppo/20260529_035456_robomimic_can_ppo_replan_only_seed0/checkpoints/best.pt \
  --output_dir /tmp/d3p_m6v_closeout_command_known_suites_full_gate
conda run -n d3p python -m pytest -q tests
```

Verification output:

```text
focused pipeline/audit tests: 13 passed in 2.07s
generated-readiness diagnostic pipeline: /tmp/d3p_m6v_closeout_command_known_suites_full_gate/20260529_101453_final_evidence_pipeline
full gate status: all_goal_evidence_ready=false, all_suites_ready=false, readiness_ready=false, markdown_ready=true
final_closeout_command suite args: 5 total; real Lift base suite, real Lift denoise-only suite, real Can base suite, real Can denoise-only suite, Square final-suite placeholder
final_closeout_command placeholder audit: no Lift suite placeholder, no Can suite placeholder; Square TD placeholder and Square PPO placeholder remain
pipeline_summary.md includes Final Close-Out Command and the same command string
full tests: 108 passed in 9.89s
```

Interpretation: after Square artifacts and the Square final suite are produced, the final close-out no longer requires reconstructing the already validated Lift/Can suite arguments by hand. The remaining manual replacements are the true missing Square outputs.

## M6w Final Close-Out Placeholder Manifest

M6w makes the generated final close-out command auditable as structured data. `pipeline_summary.json` now includes `final_closeout_ready` and `final_closeout_placeholders`; `pipeline_summary.md` prints a Close-Out Placeholders table next to the command. Each placeholder entry records the token, kind, environment, CLI argument, and replacement hint. In incomplete Square runs this exposes the exact remaining substitutions, while complete single-env runs produce an empty placeholder list and `final_closeout_ready=true`.

Implementation change:

```text
scripts/run_final_evidence_pipeline.py: builds final_closeout_placeholders for missing final suite args plus Square TD/PPO placeholders, adds final_closeout_ready, and prints a Markdown placeholder table.
tests/test_final_evidence_pipeline.py: asserts missing Square close-out emits Square suite, TD fixed-run, and PPO checkpoint placeholders; complete generated-readiness Lift runs have no placeholders.
scripts/audit_final_evidence.py: DEFAULT_MARKDOWN_TOKENS includes M6w Final Close-Out Placeholder Manifest.
tests/test_final_evidence_audit.py: verifies markdown that includes M6v but omits M6w fails only on the M6w token.
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_final_evidence_pipeline.py tests/test_final_evidence_audit.py
conda run -n d3p python scripts/run_final_evidence_pipeline.py \
  --envs robomimic_lift,robomimic_can,robomimic_square \
  --min_episodes 100 \
  --suite robomimic_lift=/tmp/d3p_m5h_lift_full_horizon_suite/20260529_043342_robomimic_lift_final_eval_suite \
  --suite robomimic_lift=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m5i_can_full_horizon_suite/20260529_050346_robomimic_can_final_eval_suite \
  --suite robomimic_can=/tmp/d3p_m6d_denoise_only_full_suite/20260529_074649_robomimic_lift_robomimic_can_final_eval_suite \
  --generate_readiness \
  --td_artifact robomimic_lift:/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m3d_data/20260529_000844_robomimic_lift_fixed_chunk_seed0/td_threshold_calibration.json \
  --td_artifact robomimic_can:/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_m5d_can_td_data/20260529_034513_robomimic_can_fixed_chunk_seed0/td_threshold_calibration.json \
  --ppo_checkpoint robomimic_lift:lift_bc_p90_reg02=/tmp/d3p_m4s_lift_bc_p90_reg02_ppo/20260529_030050_robomimic_lift_ppo_replan_only_seed0/checkpoints/best.pt \
  --ppo_checkpoint robomimic_can:can_bc_p90_reg02=/tmp/d3p_m5e_can_bc_p90_reg02_ppo/20260529_035456_robomimic_can_ppo_replan_only_seed0/checkpoints/best.pt \
  --output_dir /tmp/d3p_m6w_closeout_placeholder_manifest_full_gate
conda run -n d3p python -m pytest -q tests
```

Verification output:

```text
focused pipeline/audit tests: 13 passed in 2.09s
generated-readiness diagnostic pipeline: /tmp/d3p_m6w_closeout_placeholder_manifest_full_gate/20260529_102210_final_evidence_pipeline
full gate status: all_goal_evidence_ready=false, all_suites_ready=false, readiness_ready=false, markdown_ready=true
final_closeout_ready: false
final_closeout_placeholders: <square_final_eval_suite_dir> (final_suite, --suite), <fixed_run_dir> (square_td_fixed_run_dir, --td_artifact), <square_ppo_checkpoint.pt> (square_ppo_checkpoint, --ppo_checkpoint)
pipeline_summary.md includes Close-Out Placeholders table with all three tokens
full tests: 108 passed in 9.89s
```

Interpretation: the final handoff now separates the runnable command from the small set of unresolved substitutions, making the remaining Square close-out state easier to inspect from JSON or Markdown.

## M6x Final Close-Out Command Resolver

M6x adds a small resolver for the M6w close-out manifest. `scripts/resolve_final_closeout.py` reads a generated `pipeline_summary.json`, requires explicit `--replace TOKEN=VALUE` arguments for every listed placeholder, substitutes the generated final close-out command, verifies whether any angle-bracket placeholders remain, and writes `resolved_closeout.json` plus an executable `resolved_closeout_command.sh`. This keeps the last Square handoff reproducible: once the Square final suite, fixed-run directory, and PPO checkpoint exist, the command can be resolved without manually editing the pipeline summary.

Implementation change:

```text
scripts/resolve_final_closeout.py: resolves final_closeout_command from final_closeout_placeholders and writes JSON + executable shell output.
tests/test_resolve_final_closeout.py: covers successful resolution, missing replacement rejection, and unresolved unknown placeholder reporting.
scripts/audit_final_evidence.py: DEFAULT_MARKDOWN_TOKENS includes M6x Final Close-Out Command Resolver.
tests/test_final_evidence_audit.py: verifies markdown that includes M6w but omits M6x fails only on the M6x token.
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_resolve_final_closeout.py tests/test_final_evidence_audit.py
conda run -n d3p python scripts/resolve_final_closeout.py \
  --pipeline_summary /tmp/d3p_m6w_closeout_placeholder_manifest_full_gate/20260529_102210_final_evidence_pipeline/pipeline_summary.json \
  --replace '<square_final_eval_suite_dir>=20260529_square_final_eval_suite' \
  --replace '<fixed_run_dir>=20260529_square_fixed_seed0' \
  --replace '<square_ppo_checkpoint.pt>=/tmp/d3p_square_final/ppo_train/20260529_square_ppo/checkpoints/best.pt' \
  --output_dir /tmp/d3p_m6x_resolved_closeout_demo
conda run -n d3p python -m pytest -q tests
```

Verification output:

```text
focused resolver/audit tests: 9 passed in 0.91s
resolver demo output: /tmp/d3p_m6x_resolved_closeout_demo
resolver demo source summary: /tmp/d3p_m6w_closeout_placeholder_manifest_full_gate/20260529_102210_final_evidence_pipeline/pipeline_summary.json
resolver demo ready: true
resolver demo unresolved_tokens: []
resolver demo outputs: resolved_closeout.json and resolved_closeout_command.sh created; resolved command contains no angle-bracket placeholders
full tests: 111 passed in 9.85s
```

Interpretation: the final close-out state now has a machine-readable path from "placeholder manifest" to "resolved command". This does not remove the Square artifact blocker, but it makes the moment those artifacts exist directly verifiable and repeatable.

## M6y Final Close-Out Path Validation

M6y adds a validation mode to the close-out resolver. `scripts/resolve_final_closeout.py --validate_paths` now parses the resolved command with `shlex` and checks the concrete paths referenced by `--suite`, `--td_artifact`, `--ppo_checkpoint`, `--runtime_artifact`, `--readiness_json`, and `--env_meta_path`. The output includes `validate_paths`, `paths_ready`, and a `path_checks` list with kind, argument, path, existence, file/dir status, size, validity, and reason. This makes the final handoff stricter: after replacing the Square placeholders, the resolved command can be proven to point at existing non-empty files and suite directories before running the expensive final gate.

Implementation change:

```text
scripts/resolve_final_closeout.py: adds --validate_paths, path_checks, paths_ready, and path-aware ready computation.
tests/test_resolve_final_closeout.py: covers path-valid resolved commands and missing artifact detection.
scripts/audit_final_evidence.py: DEFAULT_MARKDOWN_TOKENS includes M6y Final Close-Out Path Validation.
tests/test_final_evidence_audit.py: verifies markdown that includes M6x but omits M6y fails only on the M6y token.
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_resolve_final_closeout.py tests/test_final_evidence_audit.py
conda run -n d3p python scripts/resolve_final_closeout.py \
  --pipeline_summary /tmp/d3p_m6w_closeout_placeholder_manifest_full_gate/20260529_102210_final_evidence_pipeline/pipeline_summary.json \
  --replace '<square_final_eval_suite_dir>=20260529_square_final_eval_suite' \
  --replace '<fixed_run_dir>=20260529_square_fixed_seed0' \
  --replace '<square_ppo_checkpoint.pt>=/tmp/d3p_square_final/ppo_train/20260529_square_ppo/checkpoints/best.pt' \
  --output_dir /tmp/d3p_m6y_resolved_closeout_validation_demo \
  --validate_paths
conda run -n d3p python -m pytest -q tests
```

Verification output:

```text
focused resolver/audit tests: 11 passed in 0.93s
validation demo output: /tmp/d3p_m6y_resolved_closeout_validation_demo
validation demo expected exit: 1 because missing Square paths are detected
validation demo ready: false
validation demo paths_ready: false
validation demo invalid path checks: 6
missing paths: Square final suite directory, Square td_critic.pt, Square td_threshold_calibration.json, Square PPO checkpoint, Square base policy checkpoint, Square normalization.npz
full tests: 113 passed in 9.85s
```

Interpretation: the resolver now distinguishes string-complete commands from artifact-ready commands. The current Square demo replacement values are expected to fail path validation until real Square outputs exist, but the same tool will return `paths_ready=true` when the final suite and artifacts are present.

## M6z Square Close-Out Replacement Discovery

M6z adds a small discovery layer for the final Square close-out handoff. `scripts/discover_square_closeout_replacements.py` scans a `square_work_root` for a completed Square final suite, a fixed-chunk TD data run containing transitions/critic/calibration outputs, and a Square PPO `checkpoints/best.pt`. It emits replacement values for the M6w/M6x/M6y placeholders, a `replacement_args` list, per-target validity checks, and a ready-to-run `scripts/resolve_final_closeout.py --validate_paths` command when a pipeline summary is provided. This removes another manual step after Square artifacts are produced: the workflow can discover the correct replacement strings instead of hand-copying directory names.

Implementation change:

```text
scripts/discover_square_closeout_replacements.py: discovers <square_final_eval_suite_dir>, <fixed_run_dir>, and <square_ppo_checkpoint.pt> replacements from square_work_root and can print a resolver command.
tests/test_discover_square_closeout_replacements.py: covers complete discovery, invalid/missing outputs, and generated resolver command contents.
scripts/audit_final_evidence.py: DEFAULT_MARKDOWN_TOKENS includes M6z Square Close-Out Replacement Discovery.
tests/test_final_evidence_audit.py: verifies markdown that includes M6y but omits M6z fails only on the M6z token.
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_discover_square_closeout_replacements.py tests/test_final_evidence_audit.py
conda run -n d3p python scripts/discover_square_closeout_replacements.py \
  --square_work_root /tmp/d3p_square_final \
  --pipeline_summary /tmp/d3p_m6w_closeout_placeholder_manifest_full_gate/20260529_102210_final_evidence_pipeline/pipeline_summary.json \
  --output_json /tmp/d3p_m6z_square_discovery/discovery.json
conda run -n d3p python -m pytest -q tests
```

Verification output:

```text
focused discovery/audit tests: 9 passed in 0.91s
discovery diagnostic output: /tmp/d3p_m6z_square_discovery/discovery.json
discovery diagnostic expected exit: 1 because Square outputs are not present under /tmp/d3p_square_final
discovery ready: false
discovery replacement_args: []
discovery checks: <square_final_eval_suite_dir>=no_valid_candidate, <fixed_run_dir>=no_valid_candidate, <square_ppo_checkpoint.pt>=no_valid_candidate
discovery resolver_command: conda run -n d3p python scripts/resolve_final_closeout.py --pipeline_summary /tmp/d3p_m6w_closeout_placeholder_manifest_full_gate/20260529_102210_final_evidence_pipeline/pipeline_summary.json --validate_paths
full tests: 116 passed in 9.90s
```

Interpretation: once Square runs exist under `/tmp/d3p_square_final`, the close-out path becomes discovery -> resolver validation -> final pipeline. The current local state is expected to report `ready=false` until those Square outputs are generated.

## M7a Square Close-Out Status Artifact

M7a consolidates the remaining Square close-out state into one status artifact. `scripts/build_square_closeout_status.py` reads a `pipeline_summary.json`, reuses the Square replacement discovery checks, optionally merges manual placeholder replacements, runs the final close-out resolver with path validation when replacements are complete, and writes both `square_closeout_status.json` and `square_closeout_status.md`. The status report includes top-level gates, structured blockers, discovery checks, resolver path checks when available, the generated final close-out command, and Square next commands from readiness.

Implementation change:

```text
scripts/build_square_closeout_status.py: builds Square close-out JSON/Markdown status from pipeline summary, square_work_root discovery, optional manual replacements, and resolver path validation.
tests/test_square_closeout_status.py: covers missing-output blocker reporting, ready discovered outputs with resolver path validation, and JSON/Markdown output creation.
scripts/audit_final_evidence.py: DEFAULT_MARKDOWN_TOKENS now requires M7a Square Close-Out Status Artifact.
tests/test_final_evidence_audit.py: verifies markdown that includes M6z but omits M7a fails only on the M7a token.
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_square_closeout_status.py tests/test_final_evidence_audit.py
conda run -n d3p python scripts/build_square_closeout_status.py \
  --pipeline_summary /tmp/d3p_m6w_closeout_placeholder_manifest_full_gate/20260529_102210_final_evidence_pipeline/pipeline_summary.json \
  --square_work_root /tmp/d3p_square_final \
  --output_dir /tmp/d3p_m7a_square_closeout_status
```

Verification output:

```text
focused status/audit tests: 9 passed in 0.92s
status diagnostic output: /tmp/d3p_m7a_square_closeout_status/square_closeout_status.json
status diagnostic expected exit: 1 because Square close-out artifacts are still missing
ready_for_final_pipeline: false
summary_readiness_ready: false
summary_suite_ready: false
discovery_ready: false
resolver_ready: null
missing replacements: <square_final_eval_suite_dir>, <fixed_run_dir>, <square_ppo_checkpoint.pt>
blocker summary: 5 readiness_missing, 12 missing_row, 3 discovery, 3 resolver_missing_replacement
```

Interpretation: the Square close-out handoff is now inspectable from one generated status file instead of requiring separate reads of readiness, discovery, resolver, and pipeline summary outputs. Current local status remains blocked on real Square runtime/checkpoint artifacts and final Square suite outputs, not on missing orchestration code.

## M7b Square Runtime Artifact And Network Config

M7b removes the local Square runtime blocker and fixes the runner-side model config mismatch exposed by the downloaded checkpoint. The Square normalization and checkpoint were downloaded using the project-local Google Drive mappings in `script/download_url.py`:

```text
normalization: data/robomimic/square/normalization.npz, size 1498 bytes
checkpoint: log/robomimic-pretrain/square/square_pre_diffusion_mlp_ta4_td20/2024-07-10_01-46-16/checkpoint/state_8000.pt, size 18,454,123 bytes
```

The first Square fixed rollout attempt failed after loading the checkpoint because `scripts/run_baseline.py` built the same `time_dim=16`, `mlp_dims=[512,512,512]` DiffusionMLP used by Lift/Can, while the Square config and checkpoint require `time_dim=32`, `mlp_dims=[1024,1024,1024]`, and `cond_mlp_dims=[512,64]`. M7b adds task-specific network kwargs for `robomimic_square` in both the baseline runner and PPO trainer so the downloaded checkpoint loads consistently across fixed, TD-error, denoise-only, and PPO paths.

Implementation change:

```text
scripts/run_baseline.py: adds diffusion_mlp_kwargs(task) and Square-specific DiffusionMLP actor config.
scripts/train_replan_ppo.py: uses the same Square-specific DiffusionMLP actor config for PPO training/evaluation rollouts.
tests/test_robomimic_network_config.py: covers Square network kwargs and confirms Lift/Can keep the previous defaults.
scripts/audit_final_evidence.py: DEFAULT_MARKDOWN_TOKENS now requires M7b Square Runtime Artifact And Network Config.
tests/test_final_evidence_audit.py: verifies markdown that includes M7a but omits M7b fails only on the M7b token.
```

Verification and generated Square artifacts:

```bash
conda run -n d3p gdown --id 1FFMqWVv0145OJjbA_iglkWywmdK22Za- -O data/robomimic/square/normalization.npz
conda run -n d3p gdown --id 1lP9mNe2AxMigfOywcaHOOR7FxQ-KR_Ee -O log/robomimic-pretrain/square/square_pre_diffusion_mlp_ta4_td20/2024-07-10_01-46-16/checkpoint/state_8000.pt
conda run -n d3p python -m pytest -q tests/test_robomimic_network_config.py
conda run -n d3p python scripts/check_final_readiness.py --output_dir /tmp/d3p_m7b_square_runtime_readiness
conda run -n d3p python scripts/run_baseline.py --env robomimic_square --method fixed_chunk --episodes 3 --max_episode_steps 400 --save_transitions true --base_policy_checkpoint /home/koukanni/Documents/D3P/log/robomimic-pretrain/square/square_pre_diffusion_mlp_ta4_td20/2024-07-10_01-46-16/checkpoint/state_8000.pt --normalization_path /home/koukanni/Documents/D3P/data/robomimic/square/normalization.npz --env_meta_path /home/koukanni/Documents/D3P/cfg/robomimic/env_meta/square.json --output_root /tmp/d3p_square_final/td_data
conda run -n d3p python scripts/train_td_critic.py --dataset /tmp/d3p_square_final/td_data/20260529_110630_robomimic_square_fixed_chunk_seed0/transitions.npz --output_dir /tmp/d3p_square_final/td_data/20260529_110630_robomimic_square_fixed_chunk_seed0/td_critic --epochs 10 --batch_size 64 --hidden_dims 64 --device cpu
conda run -n d3p python scripts/calibrate_td_threshold.py --dataset /tmp/d3p_square_final/td_data/20260529_110630_robomimic_square_fixed_chunk_seed0/transitions.npz --critic_checkpoint /tmp/d3p_square_final/td_data/20260529_110630_robomimic_square_fixed_chunk_seed0/td_critic/td_critic.pt --output_json /tmp/d3p_square_final/td_data/20260529_110630_robomimic_square_fixed_chunk_seed0/td_threshold_calibration.json --percentiles 70,80,90,95 --z_thresholds 0.5,1.0,1.5,2.0 --device cpu
conda run -n d3p python scripts/train_replan_ppo.py --env robomimic_square --seed 0 --total_env_steps 720 --rollout_steps 80 --max_episode_steps 400 --eval_episodes 3 --hidden_dims 64 --minibatch_size 40 --update_epochs 2 --lambda_C 0.003 --lambda_D 0.03 --td_critic_checkpoint /tmp/d3p_square_final/td_data/20260529_110630_robomimic_square_fixed_chunk_seed0/td_critic/td_critic.pt --uncertainty_replan_bonus 0.3 --bc_teacher td_error --bc_warmstart_steps 720 --bc_epochs 10 --bc_minibatch_size 80 --bc_learning_rate 0.001 --bc_td_calibration /tmp/d3p_square_final/td_data/20260529_110630_robomimic_square_fixed_chunk_seed0/td_threshold_calibration.json --bc_td_threshold_percentile 90 --bc_regularizer_coef 0.2 --base_policy_checkpoint /home/koukanni/Documents/D3P/log/robomimic-pretrain/square/square_pre_diffusion_mlp_ta4_td20/2024-07-10_01-46-16/checkpoint/state_8000.pt --normalization_path /home/koukanni/Documents/D3P/data/robomimic/square/normalization.npz --env_meta_path /home/koukanni/Documents/D3P/cfg/robomimic/env_meta/square.json --output_root /tmp/d3p_square_final/ppo_train
conda run -n d3p python scripts/run_final_eval_suite.py --envs robomimic_square --seeds 0 --episodes 3 --max_episode_steps 400 --baseline_methods fixed_chunk,td_error_replan,denoise_only --td_artifact robomimic_square:/tmp/d3p_square_final/td_data/20260529_110630_robomimic_square_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_square_final/td_data/20260529_110630_robomimic_square_fixed_chunk_seed0/td_threshold_calibration.json --td_threshold_percentile 95 --runtime_artifact robomimic_square:/home/koukanni/Documents/D3P/log/robomimic-pretrain/square/square_pre_diffusion_mlp_ta4_td20/2024-07-10_01-46-16/checkpoint/state_8000.pt=/home/koukanni/Documents/D3P/data/robomimic/square/normalization.npz --env_meta_path /home/koukanni/Documents/D3P/cfg/robomimic/env_meta/square.json --timeline_episodes 1 --output_root /tmp/d3p_square_final/final_suite_debug --ppo_checkpoint robomimic_square:square_bc_p90_reg02=/tmp/d3p_square_final/ppo_train/20260529_110822_robomimic_square_ppo_replan_only_seed0/checkpoints/best.pt
```

Verification output:

```text
network config tests: 2 passed in 0.91s
runtime readiness after downloads: Square base_policy_checkpoint=true, normalization=true, env_meta=true; missing td_critic, td_calibration, ppo_checkpoint
Square fixed TD data run: /tmp/d3p_square_final/td_data/20260529_110630_robomimic_square_fixed_chunk_seed0
Square fixed TD data metrics: episodes=3, success_rate=0.3333, mean_episode_return=89.0, mean_nfe_per_action=5.0, replan_rate=0.25
TD critic: num_transitions=1200, obs_dim=23, final_loss=0.25422540307044983
TD calibration p95 threshold: 1.4947950065135953
Square PPO run: /tmp/d3p_square_final/ppo_train/20260529_110822_robomimic_square_ppo_replan_only_seed0
Square PPO eval: episodes=3, success_rate=0.3333, mean_episode_return=85.3333, mean_nfe_per_action=7.2667, replan_rate=0.3633
updated readiness with TD/PPO artifacts: Square final_eval_ready=true, missing=[]
Square debug final suite: /tmp/d3p_square_final/final_suite_debug/20260529_110937_robomimic_square_final_eval_suite, num_jobs=4, num_completed=4
Square debug suite rows: fixed_chunk success_rate=0.3333 return=66.0; td_error_p95 success_rate=0.3333 return=84.6667; denoise_only success_rate=0.0 return=0.0; square_bc_p90_reg02 success_rate=0.0 return=0.0
M7b closeout status after artifacts: fixed_run_dir and square_ppo_checkpoint replacements discovered; <square_final_eval_suite_dir> still missing because the debug suite intentionally lives under final_suite_debug
```

Interpretation: Square is no longer blocked by missing local runtime artifacts or a checkpoint/model-shape mismatch. The remaining required Square work is the formal 3-seed x 100-episode final suite under `/tmp/d3p_square_final/final_suite`, followed by final aggregate audit/report generation.

## M7c Resumable Final Suite Execution

M7c makes long final evidence suites resumable and chunkable. `scripts/run_final_eval_suite.py` now accepts `--suite_dir` to pin the output directory, `--resume` to keep completed rows from an existing `suite_results.csv`, and `--max_jobs` to run only the next N pending jobs. This is specifically needed for the formal Square suite because the required matrix is 12 long jobs (`3 seeds x 4 labels`) and the previous runner only wrote a completed summary after all jobs finished sequentially.

Implementation change:

```text
scripts/run_final_eval_suite.py: adds --suite_dir, --resume, --max_jobs, completed-row detection, pending-job filtering, and partial-suite accounting fields.
tests/test_final_eval_suite.py: covers a fake two-job suite completed over two invocations with --resume --max_jobs 1.
scripts/audit_final_evidence.py: DEFAULT_MARKDOWN_TOKENS now requires M7c Resumable Final Suite Execution.
tests/test_final_evidence_audit.py: verifies markdown that includes M7b but omits M7c fails only on the M7c token.
```

Verification:

```bash
conda run -n d3p python -m pytest -q tests/test_final_eval_suite.py
conda run -n d3p python scripts/run_final_eval_suite.py \
  --envs robomimic_square \
  --seeds 0,1,2 \
  --episodes 100 \
  --max_episode_steps 400 \
  --baseline_methods fixed_chunk,td_error_replan,denoise_only \
  --td_artifact robomimic_square:/tmp/d3p_square_final/td_data/20260529_110630_robomimic_square_fixed_chunk_seed0/td_critic/td_critic.pt=/tmp/d3p_square_final/td_data/20260529_110630_robomimic_square_fixed_chunk_seed0/td_threshold_calibration.json \
  --td_threshold_percentile 95 \
  --runtime_artifact robomimic_square:/home/koukanni/Documents/D3P/log/robomimic-pretrain/square/square_pre_diffusion_mlp_ta4_td20/2024-07-10_01-46-16/checkpoint/state_8000.pt=/home/koukanni/Documents/D3P/data/robomimic/square/normalization.npz \
  --env_meta_path /home/koukanni/Documents/D3P/cfg/robomimic/env_meta/square.json \
  --timeline_episodes 1 \
  --suite_dir /tmp/d3p_m7c_square_formal_suite_dry_run \
  --ppo_checkpoint robomimic_square:square_bc_p90_reg02=/tmp/d3p_square_final/ppo_train/20260529_110822_robomimic_square_ppo_replan_only_seed0/checkpoints/best.pt \
  --dry_run
```

Verification output:

```text
focused final-suite tests: 9 passed in 4.86s
formal Square dry-run suite: /tmp/d3p_m7c_square_formal_suite_dry_run
formal Square dry-run jobs: 12
formal Square dry-run rows: fixed_chunk/td_error_p95/denoise_only/square_bc_p90_reg02 for seeds 0,1,2
formal Square dry-run num_completed: 0
formal Square dry-run num_pending: 12
formal Square dry-run suite_complete: false
```

Interpretation: the formal Square suite can now be launched safely as repeated chunks, for example with `--suite_dir /tmp/d3p_square_final/final_suite/20260529_robomimic_square_final_eval_suite --max_jobs 1` for the first job and then `--resume --max_jobs 1` until all 12 jobs are complete. This keeps partial evidence usable and prevents completed long rollouts from being discarded if a later job fails.

## M7d Square Formal 3-Seed Final Suite

M7d completes the formal Square final suite that M7c made resumable. The suite was accumulated under `/tmp/d3p_square_final/final_suite/20260529_formal_robomimic_square_final_eval_suite` with `--suite_dir`, `--resume`, and `--max_jobs 1` chunks until all 12 planned jobs were complete: seeds `0,1,2` x labels `fixed_chunk`, `td_error_p95`, `denoise_only`, and `square_bc_p90_reg02`, with `episodes=100` and `max_episode_steps=400`.

Formal Square aggregate results from the completed suite:

```text
fixed_chunk: success_rate_mean=0.3233, mean_episode_return_mean=53.6767, mean_nfe_per_action_mean=5.0000, replan_rate_mean=0.2500
td_error_p95: success_rate_mean=0.3367, mean_episode_return_mean=54.9600, mean_nfe_per_action_mean=6.2600, replan_rate_mean=0.3130, learned_replan_rate_mean=0.0868
denoise_only: success_rate_mean=0.1167, mean_episode_return_mean=22.6367, mean_nfe_per_action_mean=1.2733, replan_rate_mean=0.2500
square_bc_p90_reg02: success_rate_mean=0.3367, mean_episode_return_mean=52.4133, mean_high_level_return_mean=83.1187, mean_nfe_per_action_mean=7.3820, replan_rate_mean=0.3691, learned_replan_rate_mean=0.1604
```

Verification output:

```text
Square formal final suite: /tmp/d3p_square_final/final_suite/20260529_formal_robomimic_square_final_eval_suite
suite_summary dry_run: false
suite_summary num_jobs: 12
suite_summary num_completed: 12
suite_summary num_pending: 0
suite_summary suite_complete: true
suite_summary analysis_summary.num_rows: 12
suite_summary analysis_summary.num_aggregates: 4
```

Interpretation: Square now has the same final-suite shape as Lift and Can: three seeds, 100 episodes per seed, fixed/TD/denoise/PPO labels, per-step and per-episode source metrics, aggregate CSV/Markdown tables, Pareto plots, and one-episode timeline SVGs. TD-error and PPO tie on Square success in this run (`0.3367`), TD-error has the best raw return (`54.96`), PPO has the best high-level return (`83.1187`), fixed_chunk remains the lower-compute full-quality baseline (`5.0` NFE/action), and denoise_only is the lowest-compute negative-performance ablation (`1.2733` NFE/action).

## M7e Final Lift/Can/Square Evidence Pipeline

M7e runs the final aggregate evidence pipeline after Square completion. The resolved close-out command from `/tmp/d3p_m7d_resolved_closeout/resolved_closeout_command.sh` generated `/tmp/d3p_square_final/final_pipeline/20260529_122305_final_evidence_pipeline` with readiness generation, final evidence audit, final method matrix, final environment summary, and final Markdown/JSON report.

Top-level pipeline gates:

```text
all_goal_evidence_ready: true
all_suites_ready: true
readiness_ready: true
markdown_ready: true
require_complete: true
final_closeout_ready: true
final_closeout_placeholders: []
robomimic_lift: suite_ready=true, missing_rows=[], readiness_missing=[], failed_checks=[]
robomimic_can: suite_ready=true, missing_rows=[], readiness_missing=[], failed_checks=[]
robomimic_square: suite_ready=true, missing_rows=[], readiness_missing=[], failed_checks=[]
```

Final environment winners from `final_env_summary.csv`:

```text
lift: best_success_label=fixed_chunk, best_success_rate_mean=0.7900, best_return_label=td_error_p95, best_return_mean=46.0667, best_high_level_return_label=lift_bc_p90_reg02, best_high_level_return_mean=60.9855, lowest_compute_label=denoise_only, lowest_compute_nfe_per_action_mean=1.7284
can: best_success_label=can_bc_p90_reg02, best_success_rate_mean=0.7167, best_return_label=can_bc_p90_reg02, best_return_mean=88.9700, best_high_level_return_label=can_bc_p90_reg02, best_high_level_return_mean=163.8692, lowest_compute_label=denoise_only, lowest_compute_nfe_per_action_mean=1.7220
square: best_success_label=square_bc_p90_reg02, best_success_rate_mean=0.3367, best_return_label=td_error_p95, best_return_mean=54.9600, best_high_level_return_label=square_bc_p90_reg02, best_high_level_return_mean=83.1187, lowest_compute_label=denoise_only, lowest_compute_nfe_per_action_mean=1.2733
```

Final report artifacts:

```text
pipeline_summary: /tmp/d3p_square_final/final_pipeline/20260529_122305_final_evidence_pipeline/pipeline_summary.json
final_report_json: /tmp/d3p_square_final/final_pipeline/20260529_122305_final_evidence_pipeline/report/final_report.json
final_results_matrix_csv: /tmp/d3p_square_final/final_pipeline/20260529_122305_final_evidence_pipeline/report/final_results_matrix.csv
final_env_summary_csv: /tmp/d3p_square_final/final_pipeline/20260529_122305_final_evidence_pipeline/report/final_env_summary.csv
final_report_markdown: /tmp/d3p_square_final/final_pipeline/20260529_122305_final_evidence_pipeline/report/final_report.md
```

Interpretation: the Agents.md evidence objective now has a complete machine-readable data bundle for Lift, Can, and Square. The final matrix supports method-level comparison against fixed_chunk within each environment, and the final env summary provides compact winners for success, raw return, high-level return, NFE/action, and wall-time/action. Square evaluation is complete.

## Current Interpretation

- The fixed baseline replans only when the buffer is empty, as expected. With `H=4`, debug rollouts produce `replan_rate=0.25`, and final fixed_chunk rows keep `mean_nfe_per_action=5.0`.
- Lift full-horizon evidence favors fixed_chunk on success (`0.79`) and TD-error p95 on raw return (`46.0667`). Lift PPO has the best high-level return but does not beat fixed_chunk on raw success.
- Can full-horizon evidence is the strongest PPO result: `can_bc_p90_reg02` has the best success (`0.7167`), raw return (`88.97`), and high-level return (`163.8692`), but it costs about `12.338` NFE/action and much higher discard/replan rates.
- Square final evidence is complete. TD-error p95 and `square_bc_p90_reg02` tie on success (`0.3367`), TD-error has the best raw return (`54.96`), and PPO has the best high-level return (`83.1187`). The Square PPO advantage over fixed_chunk on success is small (`+0.0133`) and comes with higher compute (`7.382` vs `5.0` NFE/action).
- Denoise-only is consistently the lowest-compute ablation across Lift, Can, and Square, but it is a negative task-performance result. It should be reported as an efficiency ablation, not as a competitive final policy.
- The final data bundle is ready for analysis: `final_results_matrix.csv` is the method-level table, `final_env_summary.csv` is the environment-level winner table, and the generated final report records the same complete Lift/Can/Square evidence with readiness and audit gates passing.

## Open Work

No required evidence collection remains for the current Agents.md objective. Useful follow-up work is analysis/reporting, not infrastructure closure: preserve the `/tmp/d3p_*` artifact directories needed by the final report, decide how to present the task-specific tradeoffs, and rerun suites only if changing checkpoints, seeds, horizons, or replanning thresholds.

## Known Environment Notes

- Use `conda run -n d3p ...` for all verification.
- Full historical pytest collection outside `tests/` can hit unrelated legacy imports; target `tests` or specific new tests for this milestone.
- Robomimic smoke emits Gym/robosuite warnings and a `mujoco_py` rebuild message; these occurred on passing runs and did not block execution.
