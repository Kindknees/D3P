# Adaptive Commitment Experiment Summary

> Local Robomimic experiments. Main method: `adaptive_commitment_with_repair`.
>
> Main conclusion: strongest evidence is **anti-reset / continuity preservation** at similar NFE/action. Return improvement is positive in many runs, but fixed-baseline return dominance is not yet confirmed.

---

## 0. Claim Dashboard

| Claim | Current status | Best wording |
|---|---|---|
| Reduce destructive random reset | Strong | Reduces or eliminates non-empty-buffer random reset/discard. |
| Improve action continuity | Strong | Action jump / buffer jump are much lower than phase-blind random reset. |
| Keep compute similar | Supported | NFE/action usually stays near baseline. |
| Improve return over random reset | Partial | Strongest on Square; Can/Lift are more tuning-dependent. |
| Improve return over fixed chunk | Trend only | Many positive deltas, but confirmatory CI can cross zero. |
| Learned benefit predictor | Proof of concept | Works as branch-table controller; not final learned policy. |
| Criticality-aware compute | Task-calibrated | Square positive; not universal across tasks. |

---

## 1. Experiment Index

| Block | Output root | Scope | Main takeaway |
|---|---|---:|---|
| Pilot verified | `lift_policy_disagreement_verified`, `can_policy_disagreement_verified`, `square_policy_disagreement_verified_band_rngsafe` | 3 tasks × 5 seeds × 1 ep | Positive return/success vs fixed. |
| Matrix A full | `matrix_A_policy_disagreement_full_v1` | 105 runs | Positive signals vs fixed and random reset; anti-reset effect strong. |
| Matrix B stress | `matrix_B_execution_disturbance_v1` | 180 runs | Adaptive variants beat random reset in Can/Square disturbance cases. |
| Can repair gate | `matrix_B_can_repair_acceptance_v1` | 60 runs | Action-jump gate fixes harmful Can repairs under stress. |
| Matrix C Square | `matrix_C_square_policy_disagreement_v1` | 25 runs | Residual repair 0.50 and adaptive+repair beat random reset. |
| Matrix D Pareto | `matrix_D_square_pareto_v1` | 135 runs | Square best frontier at reset NFE 10, repair NFE 2. |
| Matrix E benefit | `matrix_E_benefit_predictor_v2` | 60 runs | Branch-table predictor beats TD-only random reset on return/success. |
| H5 compute | `matrix_H5_criticality_compute_task_tuned_v1` | 45 runs | Square gets higher return with about half NFE/action. |
| Confirmatory small | `confirmatory_repair_v1_small` | paired Can/Square | Return over fixed not confirmed; continuity/discard reduction supported. |

---

## 2. Pilot Verified Result vs Fixed Chunk

| Task | Fixed success | Fixed return | Fixed NFE/action | Adaptive+repair success | Adaptive+repair return | Adaptive+repair NFE/action | Δ success | Δ return | NFE ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Lift | 0.8 | 32.8 | 5.000 | 0.8 | 35.0 | 5.140 | +0.0 | +2.2 | 1.028 |
| Can | 0.6 | 75.4 | 5.000 | 0.8 | 115.4 | 5.167 | +0.2 | +40.0 | 1.033 |
| Square | 0.6 | 85.6 | 5.000 | 0.6 | 103.8 | 5.120 | +0.0 | +18.2 | 1.024 |

---

## 3. Full Matrix A: Main Method vs Fixed

| Task | Fixed success | Fixed return | Adaptive+repair success | Adaptive+repair return | Δ success | Δ return | NFE ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lift | 0.8 | 32.8 | 0.8 | 33.4 | +0.0 | +0.6 | 1.028 |
| Can | 0.6 | 75.4 | 0.6 | 80.6 | +0.0 | +5.2 | 1.033 |
| Square | 0.6 | 85.6 | 0.6 | 103.8 | +0.0 | +18.2 | 1.024 |

| Baseline | Positive signals |
|---|---:|
| Fixed chunk | 9 |
| TD-anywhere random reset | 7 |

---

## 4. Full Matrix A: Main Method vs Phase-Blind Random Reset

| Task | Random-reset return | Adaptive+repair return | Δ return | Random resets | Discarded residual actions | Action jump |
|---|---:|---:|---:|---:|---:|---:|
| Lift | 34.4 | 33.4 | -1.0 | 0.2 → 0.0 | 0.6 → 0.0 | 0.0413 → 0.0187 |
| Can | 76.4 | 80.6 | +4.2 | 1.0 → 0.0 | 3.0 → 0.0 | 0.4078 → 0.1394 |
| Square | 42.6 | 103.8 | +61.2 | 0.8 → 0.0 | 2.4 → 0.0 | 0.1878 → 0.0170 |

---

## 5. Matrix C: Square Repair / Reset Comparison

| Method | Success | Mean return | NFE/action | Action jump | Buffer jump |
|---|---:|---:|---:|---:|---:|
| `td_replan_anywhere_random_reset` | 0.2 | 42.6 | 5.140 | 0.1878 | 0.1785 |
| `td_repair_residual_025` | 0.0 | 0.0 | 5.120 | 0.0928 | 0.1657 |
| `td_repair_residual_050` | 0.4 | 75.0 | 5.120 | 0.0808 | 0.1661 |
| `td_repair_residual_075` | 0.2 | 52.4 | 5.120 | 0.0908 | 0.1478 |
| `adaptive_commitment_with_repair` | 0.6 | 103.8 | 5.120 | 0.0170 | 0.0453 |

| Method | Δ success vs random reset | Δ return vs random reset | NFE ratio |
|---|---:|---:|---:|
| `adaptive_commitment_with_repair` | +0.4 | +61.2 | 0.996 |
| `td_repair_residual_050` | +0.2 | +32.4 | 0.996 |
| `td_repair_residual_075` | +0.0 | +9.8 | 0.996 |

---

## 6. Matrix B: Execution-Disturbance Stress Test

| Sigma | Task | TD-anywhere return / success | Adaptive heuristic return / success | Adaptive+repair return / success |
|---:|---|---:|---:|---:|
| 0.02 | Square | 24.2 / 0.2 | 52.4 / 0.2 | 54.2 / 0.2 |
| 0.05 | Can | 54.4 / 0.4 | 75.2 / 0.6 | 53.2 / 0.4 |
| 0.05 | Square | 23.2 / 0.2 | 50.0 / 0.2 | 52.4 / 0.2 |
| 0.10 | Can | 54.6 / 0.4 | 112.2 / 0.8 | 53.2 / 0.4 |
| 0.10 | Square | 45.6 / 0.2 | 76.0 / 0.4 | 73.8 / 0.4 |

| Sigma | Baseline | Positive signals |
|---:|---|---:|
| 0.02 | Fixed chunk | 4 |
| 0.02 | TD-anywhere random reset | 3 |
| 0.05 | Fixed chunk | 1 |
| 0.05 | TD-anywhere random reset | 8 |
| 0.10 | Fixed chunk | 0 |
| 0.10 | TD-anywhere random reset | 5 |

---

## 7. Can Repair-Acceptance Gate

| Sigma | Method | Success | Mean return | NFE/action | Mean repairs |
|---:|---|---:|---:|---:|---:|
| 0.02 | `adaptive_commitment_with_repair` | 0.6 | 74.4 | 5.167 | 0.6 |
| 0.05 | `adaptive_commitment_with_repair` | 0.6 | 74.4 | 5.167 | 0.6 |
| 0.10 | `adaptive_commitment_with_repair` | 0.8 | 111.4 | 5.167 | 0.6 |

| Sigma | Δ success vs random reset | Δ return vs random reset | NFE ratio |
|---:|---:|---:|---:|
| 0.05 | +0.2 | +20.0 | 0.994 |
| 0.10 | +0.4 | +56.8 | 0.994 |

---

## 8. Matrix D: Square Return-Cost Pareto

| Reset NFE | Repair NFE | Method | Success | Mean return | NFE/action |
|---:|---:|---|---:|---:|---:|
| 10 | 2 | `fixed_chunk` | 0.8 | 150.8 | 2.500 |
| 10 | 5 | `fixed_chunk` | 0.8 | 150.8 | 2.500 |
| 10 | 10 | `fixed_chunk` | 0.8 | 150.8 | 2.500 |
| 10 | 2 | `adaptive_commitment_with_repair` | 0.8 | 152.6 | 2.555 |

| Reset NFE | Repair NFE | Method | Success | Mean return | NFE/action |
|---:|---:|---|---:|---:|---:|
| 10 | 2 | `td_replan_anywhere_random_reset` | 0.4 | 68.2 | 2.575 |
| 20 | 10 | `fixed_chunk` | 0.6 | 85.6 | 5.000 |
| 20 | 10 | `adaptive_commitment_with_repair` | 0.6 | 103.8 | 5.120 |
| 20 | 10 | `td_replan_anywhere_random_reset` | 0.2 | 42.6 | 5.140 |

---

## 9. Matrix E: Benefit-Predictor Controller

| Task | Method | Success | Mean return | NFE/action | Random resets | Discard non-empty buffer |
|---|---|---:|---:|---:|---:|---:|
| Can | `td_replan_anywhere_random_reset` | 0.6 | 76.4 | 5.200 | 1.0 | 3.0 |
| Can | `benefit_predictor_controller` | 0.8 | 78.2 | 5.200 | 1.0 | 2.0 |
| Lift | `td_replan_anywhere_random_reset` | 0.8 | 34.4 | 5.147 | 0.2 | 0.6 |
| Lift | `benefit_predictor_controller` | 1.0 | 40.2 | 5.227 | 1.4 | 4.0 |
| Square | `td_replan_anywhere_random_reset` | 0.2 | 42.6 | 5.140 | 0.8 | 2.4 |
| Square | `benefit_predictor_controller` | 0.4 | 75.6 | 5.125 | 0.0 | 0.0 |

| Task | Δ success vs TD-anywhere | Δ return vs TD-anywhere | NFE ratio |
|---|---:|---:|---:|
| Can | +0.2 | +1.8 | 1.000 |
| Lift | +0.2 | +5.8 | 1.016 |
| Square | +0.2 | +33.0 | 0.997 |

---

## 10. H5: Criticality-Aware Compute Allocation

| Task | Method | Success | Mean return | NFE/action |
|---|---|---:|---:|---:|
| Can | `fixed_chunk` | 0.6 | 75.4 | 5.000 |
| Can | `adaptive_commitment_with_repair` | 0.6 | 80.6 | 5.167 |
| Can | `criticality_aware_commitment_with_repair` | 0.6 | 80.6 | 5.167 |
| Lift | `fixed_chunk` | 0.8 | 32.8 | 5.000 |
| Lift | `adaptive_commitment_with_repair` | 0.8 | 33.4 | 5.140 |
| Lift | `criticality_aware_commitment_with_repair` | 0.8 | 33.4 | 5.140 |
| Square | `fixed_chunk` | 0.6 | 85.6 | 5.000 |
| Square | `adaptive_commitment_with_repair` | 0.6 | 103.8 | 5.120 |
| Square | `criticality_aware_commitment_with_repair` | 0.6 | 114.6 | 2.625 |

| Task | Δ success vs fixed | Δ return vs fixed | NFE ratio vs fixed |
|---|---:|---:|---:|
| Can | +0.0 | +5.2 | 1.033 |
| Lift | +0.0 | +0.6 | 1.028 |
| Square | +0.0 | +29.0 | 0.525 |

| Task | Δ success vs adaptive+repair | Δ return vs adaptive+repair | NFE ratio vs adaptive+repair |
|---|---:|---:|---:|
| Square | +0.0 | +10.8 | 0.513 |

---

## 11. Confirmatory / Held-Out Status

| Run | Output root | Result status |
|---|---|---|
| Confirmatory repair v1 small | `confirmatory_repair_v1_small` | Does not confirm return improvement over fixed; supports sparse repairs and continuity/discard reduction vs random reset. |
| Original pilot setup rerun | `confirmatory_repair_pilot_setup_seed0_ep50` | Original Can no-gate setup does not recover fixed-baseline return gain; random-reset trend positive but CI crosses zero. |
| Can task-specific tuning | `can_repair_tuning_heldout_t00575_seed50_ep50` | +3.96 return, +0.08 success, 1.033× NFE vs fixed; CI crosses zero. |
| Lift task-specific tuning | `lift_repair_tuning_heldout_t010_n075_a075_max040_seed100_ep50` | Selected gated setup does not hold on seed 100; use only as sparse-repair / anti-reset evidence. |
| Square task-specific tuning | `square_repair_tuning_nfe10_rnfe2_t003_n050_a050_min005_max015_seeds5_9_ep10` | +6.24 return, +0.06 success, 1.022× NFE vs fixed; CI crosses zero. |

---

## 12. Hypothesis Status

| Hypothesis | Status | Evidence |
|---|---|---|
| H1: adaptive commitment avoids arbitrary mid-chunk interruption | Partial | Eliminates non-empty random resets and reduces action jump/discard; no fixed-return dominance. |
| H2: buffer-preserving repair reduces mode switching | Supported on proxies | Action jump / buffer jump strongly reduced; geometric mode labels not implemented. |
| H3: three-way controller beats binary controller | Partial | Beats phase-blind random reset on continuity/discard; Square return trend strongest. |
| H4: benefit prediction is cleaner than TD-error-only | Proof of concept | Branch-table predictor beats TD-only random reset on return/success; false-reset reduction task-dependent. |
| H5: criticality-aware compute improves efficiency | Task-calibrated | Square improves return and halves NFE/action; Can/Lift need full NFE. |

---

## 13. Final Claim Wording

| Type | Claim |
|---|---|
| Main | Adaptive commitment with buffer-preserving repair is a sparse anti-reset mechanism for chunked diffusion policies. |
| Strong | It reduces phase-blind random resets, non-empty-buffer discard, and action discontinuity at similar NFE/action. |
| Moderate | It shows positive return/success trends, especially on Can/Square and against TD-anywhere random reset. |
| Limitation | Return dominance over fixed chunk is task-dependent and not yet confirmatorily verified. |

---

## 14. Do Not Claim Yet

| Do not claim | Reason |
|---|---|
| Universal return improvement over fixed chunk | Confirmatory CI can cross zero. |
| Fully learned benefit predictor | Current controller is table-based proof of concept. |
| Globally reliable false-reset reduction for H4 | Lift increases random resets/discards in Matrix E. |
| Universal criticality-aware compute | H5 is task-calibrated; Square positive, Can/Lift need full NFE. |
| Direct mode-switch measurement | Current evidence uses action/buffer jumps as proxies. |
| Paper-scale evaluation | Many blocks are 5 seeds × 1 episode; confirmatory runs are larger but still limited. |
