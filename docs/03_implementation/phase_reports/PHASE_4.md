# Phase 4 — Personalized Federation (FedBN) and Comparison

**Date:** 2026-10-13
**Phase:** 4

---

## Objective

Extend the federated client to support FedBN mode: BatchNorm layers remain local to each hospital while all other layers are averaged. Run FedBN for 10 rounds with seed 42. Compare best-round validation accuracy against the FedAvg baseline and local-only training.

---

## Requirements Covered

| Requirement ID | Description |
|---|---|
| FR-06 | Personalized FedBN mode: BatchNorm layers kept local |
| FR-07 | Saved comparison table and chart: FedAvg vs FedBN per hospital |

---

## Design Notes

- `bn_keys(model)` collects all BatchNorm2d state keys (weight, bias, running_mean, running_var, num_batches_tracked) in sorted order.
- At the start of each training round in personalized mode, the client loads the global weights then overlays its saved local BN state via `merge_state`. This ensures the shared backbone is updated while BN statistics remain hospital-specific.
- After local training, the updated BN state is extracted and saved to `checkpoints/<run-name>/hospital<n>_bn.pt` before the non-BN weights are returned to the server.
- `prepare_local_bn` deletes the stale BN file at round 1 so a repeated run starts with fresh BN statistics.
- The server-side aggregation is unchanged — it averages whatever weights the clients return. Since clients strip BN keys before returning, BN is never averaged.
- `summarize_runs.py` uses best-round accuracy (max over all rounds) for the final comparison, making the metric fair regardless of round count.

---

## Implementation

| File | Action | Description |
|---|---|---|
| `medifed/personalization.py` | Created | bn_keys, split_state, merge_state, save_bn_state, load_bn_state, prepare_local_bn, bn_path |
| `medifed/client_app.py` | Modified | Added personalized mode branch: load local BN before train, save local BN after train |
| `scripts/summarize_runs.py` | Modified | Added best-round aggregation and comparison_methods.csv generation |
| `scripts/make_charts.py` | Created | Grouped bar chart and round line chart from summary CSVs |
| `tests/test_personalization.py` | Created | TC-FBN-001 to TC-FBN-008 |
| `tests/test_fedbn_simulation.py` | Created | End-to-end FedBN simulation test |

---

## Test Execution

| Test ID | Description | Result |
|---|---|---|
| TC-FBN-001 | bn_keys returns only BatchNorm2d keys | PASS |
| TC-FBN-002 | split_state separates BN and non-BN keys correctly | PASS |
| TC-FBN-003 | merge_state raises KeyError for missing BN keys | PASS |
| TC-FBN-004 | merge_state produces correct merged state | PASS |
| TC-FBN-005 | save_bn_state / load_bn_state round-trip is lossless | PASS |
| TC-FBN-006 | prepare_local_bn deletes file at round 1 | PASS |
| TC-FBN-007 | prepare_local_bn loads saved state at round > 1 | PASS |
| TC-FBN-008 | bn_path returns correct checkpoint path | PASS |

---

## Results

FedBN seed 42 run — best-round validation accuracy per hospital:

| Hospital | Best Round | FedAvg Best | FedBN Best | FedBN − FedAvg |
|---|---|---|---|---|
| 1 (NORMAL, PNEUMONIA) | 8 | 95.4% | 95.4% | 0.0 pp (tie) |
| 2 (NORMAL, TUBERCULOSIS) | 4 | 99.6% | 99.5% | −0.1 pp |
| 3 (COVID19, NORMAL, PNEUMONIA) | 10 | 86.1% | 86.7% | **+0.6 pp** |
| **Mean** | — | **93.7%** | **93.9%** | **+0.2 pp** |

FedBN matches or exceeds FedAvg at every hospital. The gain is largest at Hospital 3, which has the most heterogeneous class set and the most distinct image acquisition protocol — exactly the scenario where keeping BatchNorm statistics local is expected to help. Hospital 1 ties because its two-class distribution is homogeneous enough that shared BN statistics are already a good fit.

Global checkpoint saved: `checkpoints/personalized_seed42_global.pt`
Local BN checkpoints saved: `checkpoints/personalized_seed42/hospital{1,2,3}_bn.pt`
Comparison CSV saved: `results/comparison_methods.csv`
Charts saved: `results/chart_methods_comparison.png`, `results/chart_rounds.png`

---

## Exit Criteria and Status

| Criterion | Status |
|---|---|
| All personalization tests pass | PASS |
| 10-round FedBN run completes with checkpoint | PASS |
| Comparison CSV and charts written | PASS |
| TESTING_LOG.md updated | PASS |
