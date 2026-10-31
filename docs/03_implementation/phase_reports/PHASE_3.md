# Phase 3 — Federated Baseline (FedAvg)

**Date:** 2026-10-12
**Phase:** 3

---

## Objective

Deploy the Flower federated runtime with three hospital SuperNodes and one SuperLink aggregation server. Run FedAvg for 10 rounds with seed 42. Save the global checkpoint and per-round metrics. Evaluate the global model on each hospital's validation set and compare against local-only training.

---

## Requirements Covered

| Requirement ID | Description |
|---|---|
| FR-05 | Federated FedAvg baseline: all layers averaged each round |
| NFR-01 | Raw images never leave a hospital client process |
| NFR-03 | Reproducible run with seed 42 |

---

## Design Notes

- The Flower `ClientApp` receives the global `state_dict` as an `ArrayRecord`, loads it into a non-pretrained ResNet18, trains for `FL_LOCAL_EPOCHS=1` with a fresh Adam optimizer, and returns the updated CPU weights.
- The `ServerApp` uses `FedAvg` with `fraction_train=1.0`, `fraction_evaluate=1.0`, `min_train_nodes=3`, `min_evaluate_nodes=3` so every hospital participates in every round.
- Each client writes one train row and one val row per round to `results/runs/<run-name>/hospital<n>_rounds.csv`. Files are per-hospital to avoid concurrent write conflicts.
- `reset_round_csv_for_first_round` removes a stale CSV at round 1 so repeated runs start clean.
- `MEDIFED_ROOT` is set by the launcher so Flower's copied app can locate manifests and checkpoints.

---

## Implementation

| File | Action | Description |
|---|---|---|
| `medifed/client_app.py` | Created | Flower ClientApp: train and evaluate handlers |
| `medifed/server_app.py` | Created | Flower ServerApp: FedAvg strategy, checkpoint save |
| `medifed/federated.py` | Created | Pure helpers: partition_to_hospital, local_update, append_round_row, state_dict_to_cpu |
| `scripts/start_federation.ps1` | Created | Opens SuperLink + 3 SuperNode windows |
| `scripts/stop_federation.ps1` | Created | Kills Flower processes and checks ports |
| `scripts/run_experiments.ps1` | Created | Writes run_config.toml, calls flwr run, verifies completion |
| `scripts/summarize_runs.py` | Created | Reads run CSVs, writes summary_rounds.csv, summary_final.csv, comparison_methods.csv |
| `scripts/evaluate_federated.py` | Created | Loads global checkpoint, evaluates per hospital, writes final_metrics.csv |
| `tests/test_federated.py` | Created | 12 unit tests: TC-FED-001 to TC-FED-012 |

---

## Test Execution

| Test ID | Description | Result |
|---|---|---|
| TC-FED-001 | resolve_project_root returns env path when set | PASS |
| TC-FED-002 | resolve_project_root returns fallback when unset | PASS |
| TC-FED-003 | partition_to_hospital maps 0,1,2 to 1,2,3; raises ValueError for 3 and -1 | PASS |
| TC-FED-004 | local_update keeps state_dict keys/shapes; train_loss is finite | PASS |
| TC-FED-005 | append_round_row writes header once and appends rows | PASS |
| TC-FED-006 | client_app and server_app import and expose app | PASS |
| TC-FED-007 | state_dict_to_cpu returns tensors on cpu | PASS |
| TC-FED-008 | MEDIFED_DEVICE=cpu makes get_device return cpu | PASS |
| TC-FED-009 | Invalid MEDIFED_DEVICE raises ValueError | PASS |
| TC-FED-010 | validate_mode accepts baseline and rejects invalid mode | PASS |
| TC-FED-011 | PowerShell scripts parse without syntax errors | PASS |
| TC-FED-012 | Round 1 removes existing rounds CSV; later rounds preserve it | PASS |

---

## Results

Baseline seed 42 run — best-round validation accuracy per hospital:

| Hospital | Best Round | Val Accuracy | Majority Baseline |
|---|---|---|---|
| 1 (NORMAL, PNEUMONIA) | 8 | 95.4% | 50.3% |
| 2 (NORMAL, TUBERCULOSIS) | 9 | 99.6% | 50.0% |
| 3 (COVID19, NORMAL, PNEUMONIA) | 4 | 86.1% | 33.3% |
| **Mean** | — | **93.7%** | — |

All three hospitals beat the majority baseline by a large margin. The global FedAvg model generalises across hospital distributions without any raw image leaving its hospital process.

Global checkpoint saved: `checkpoints/baseline_seed42_global.pt`
Round metrics saved: `results/runs/baseline_seed42/hospital{1,2,3}_rounds.csv`
Final metrics saved: `results/baseline_seed42_final_metrics.csv`

---

## Exit Criteria and Status

| Criterion | Status |
|---|---|
| All federated tests pass | PASS |
| 10-round baseline run completes with checkpoint | PASS |
| Per-hospital final metrics CSV written | PASS |
| TESTING_LOG.md updated | PASS |
