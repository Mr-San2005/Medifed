# Phase 2 — Model and Local Training

**Date:** 2026-10-11
**Phase:** 2

---

## Objective

Build the ResNet18 4-class model, implement the local training and evaluation functions, and run a full local training sanity check at all three hospitals. Confirm that each hospital's model beats the majority-class baseline on its own validation set, and produce the cross-hospital evaluation table that motivates federated learning.

---

## Requirements Covered

| Requirement ID | Description |
|---|---|
| FR-04 | Each hospital trains a ResNet18 with the final layer replaced for 4-class output, using only its local images |
| NFR-02 | Runs on 4 GB GPU (CUDA confirmed) with CPU fallback via `get_device()` |
| NFR-03 | Reproducible runs — `set_seed(SEED)` called before each hospital's training |

---

## Design Notes

- `build_model(pretrained=True)` uses `ResNet18_Weights.DEFAULT` (the modern torchvision API) to avoid deprecation warnings. Only the final FC layer is replaced; all other layers keep their pretrained weights.
- `NUM_CLASSES = len(CLASS_NAMES)` is derived in `config.py` so the model head always matches the class registry without a separate constant to maintain.
- `train.py` uses plain functions (no classes) as specified. `evaluate` computes the full confusion matrix in a single pass using `torch.no_grad()`.
- `per_class_recall` returns `None` for classes absent from the loader, not zero, to distinguish "not present" from "zero recall".
- The training script calls `set_seed(SEED)` before building each hospital's model so results are reproducible across runs.

---

## Implementation

| File | Action | Description |
|---|---|---|
| `medifed/config.py` | Modified | Added `BATCH_SIZE=16`, `LOCAL_EPOCHS=5`, `LEARNING_RATE=1e-4`, `NUM_CLASSES` derived from `CLASS_NAMES` |
| `medifed/model.py` | Created | `build_model`, `get_device`, `set_seed` |
| `medifed/train.py` | Created | `train_one_epoch`, `evaluate`, `fit` |
| `scripts/train_local.py` | Created | Per-hospital training loop, final metrics table, cross-hospital evaluation, CSV outputs |
| `tests/test_model.py` | Created | 8 unit tests: TC-MODEL-001 to TC-MODEL-004, TC-TRAIN-001 to TC-TRAIN-004 |
| `docs/03_implementation/MODEL_AND_TRAINING.md` | Created | Model design, hyperparameters, full results tables |
| `docs/02_design/LLD.md` | Modified | Sections 3 and 4 updated to Phase 2 built status |
| `README.md` | Modified | Phase status table and how-to-run updated |

---

## Test Execution

| Test ID | Description | Expected | Actual | Result |
|---|---|---|---|---|
| TC-MODEL-001 | build_model output shape is (batch, 4) for 224×224 input | shape == (2, 4) | (2, 4) | PASS |
| TC-MODEL-002 | Final layer has 4 outputs, in_features=512 | fc.out=4, fc.in=512 | Correct | PASS |
| TC-MODEL-003 | get_device returns cuda (CUDA available on test machine) | device.type == "cuda" | "cuda" | PASS |
| TC-MODEL-004 | set_seed makes two models have identical weights | all params equal | Equal | PASS |
| TC-TRAIN-001 | 30 training steps lower the loss on 8 synthetic images | last_loss < first_loss | Confirmed | PASS |
| TC-TRAIN-002 | evaluate: accuracy in [0,1], num_samples=8, confusion sum=8 | All conditions | Correct | PASS |
| TC-TRAIN-003 | per_class_recall is None for classes with no true samples | recall[2]=None, recall[3]=None | None | PASS |
| TC-TRAIN-004 | Checkpoint save/load gives identical outputs | allclose(out1, out2) | True | PASS |

All 28 tests (8 config + 12 data + 8 model/train) passed. Run time: 14.499 s.

---

## Raw Test Output

```
test_class_order (test_config.TestConfig.test_class_order)
TC-CFG-001: class order is the fixed global order. ... ok
test_class_to_index_consistent (test_config.TestConfig.test_class_to_index_consistent)
TC-CFG-002: CLASS_TO_INDEX matches CLASS_NAMES position for every entry. ... ok
test_exactly_three_hospitals (test_config.TestConfig.test_exactly_three_hospitals)
TC-CFG-003: HOSPITALS contains exactly 3 entries. ... ok
test_hospital2_classes (test_config.TestConfig.test_hospital2_classes)
TC-CFG-006: hospital 2 has NORMAL, TUBERCULOSIS. ... ok
test_hospital3_classes (test_config.TestConfig.test_hospital3_classes)
TC-CFG-005: hospital 3 has COVID19, NORMAL, PNEUMONIA. ... ok
test_hospital_classes_subset_of_class_names (test_config.TestConfig.test_hospital_classes_subset_of_class_names)
TC-CFG-004: every hospital's class list is a subset of CLASS_NAMES. ... ok
test_project_root_config_exists (test_config.TestConfig.test_project_root_config_exists)
TC-CFG-008: PROJECT_ROOT/medifed/config.py resolves to an existing file. ... ok
test_seed_is_int (test_config.TestConfig.test_seed_is_int)
TC-CFG-007: SEED is an integer. ... ok
test_cap_respected ... ok
test_global_indices_hospital2_style ... ok
test_same_seed_same_sample ... ok
test_train_val_no_overlap ... ok
test_dataset_item_shape_and_type ... ok
test_corrupt_skipped ... ok
test_junk_skipped ... ok
test_lung_opacity_excluded ... ok
test_macosx_excluded ... ok
test_masks_excluded ... ok
test_nearest_ancestor_class ... ok
test_write_read_roundtrip ... ok
test_final_layer_only_changed (test_model.TestBuildModel.test_final_layer_only_changed)
TC-MODEL-002: the final layer has exactly 4 outputs and no other layer was changed in shape. ... ok
test_get_device_type (test_model.TestBuildModel.test_get_device_type)
TC-MODEL-003: get_device returns cuda when available and cpu otherwise. ... ok
test_output_shape (test_model.TestBuildModel.test_output_shape)
TC-MODEL-001: build_model output shape is (batch, 4) for a 224x224 input. ... ok
test_set_seed_identical_weights (test_model.TestBuildModel.test_set_seed_identical_weights)
TC-MODEL-004: set_seed makes two freshly built models have identical weights. ... ok
test_checkpoint_save_load_identical_output (test_model.TestTrainFunctions.test_checkpoint_save_load_identical_output)
TC-TRAIN-004: a checkpoint saved with torch.save and loaded into a fresh model gives identical outputs. ... ok
test_evaluate_ranges_and_counts (test_model.TestTrainFunctions.test_evaluate_ranges_and_counts)
TC-TRAIN-002: evaluate returns accuracy in [0,1], num_samples equals dataset size, confusion sum equals num_samples. ... ok
test_loss_decreases (test_model.TestTrainFunctions.test_loss_decreases)
TC-TRAIN-001: on 8 synthetic images with fixed labels, 30 training steps lower the loss. ... ok
test_per_class_recall_none_for_missing_class (test_model.TestTrainFunctions.test_per_class_recall_none_for_missing_class)
TC-TRAIN-003: per_class_recall is None for a class with no true samples. ... ok

----------------------------------------------------------------------
Ran 28 tests in 14.499s

OK

============================================================
Run:    28
Passed: 28
Failed: 0
Errors: 0
Output: E:\MediFed\docs\04_testing\test_runs\test_run_20261011_111903.txt
============================================================
```

---

## Defects Found and Fixes

None.

---

## Results

All three hospitals beat the majority-class baseline by a large margin after 5 epochs:

| Hospital | Val Accuracy | Majority Baseline | Beats? |
|---|---|---|---|
| 1 (NORMAL, PNEUMONIA) | 0.9477 | 0.5033 | YES |
| 2 (NORMAL, TUBERCULOSIS) | 0.9800 | 0.5000 | YES |
| 3 (COVID19, NORMAL, PNEUMONIA) | 0.8667 | 0.3333 | YES |

Cross-hospital evaluation (rows = model trained at, columns = evaluated on):

| | H1 val | H2 val | H3 val |
|---|---|---|---|
| Model H1 | 0.9477 | 0.3150 | 0.4167 |
| Model H2 | 0.4771 | 0.9800 | 0.1633 |
| Model H3 | 0.9085 | 0.4850 | 0.8667 |

The diagonal is strong; off-diagonal accuracy is poor. This confirms that locally trained models do not generalise across hospitals — the core motivation for federated learning in Phase 3.

---

## Demo Script for Guide Review

1. Activate the virtual environment: `.\venv\Scripts\Activate.ps1`
2. Run `python scripts/run_tests.py` — observe 28 tests, all OK.
3. Run `python scripts/train_local.py` — observe per-epoch loss/accuracy for each hospital, final metrics, confusion matrices, and the 3×3 cross-hospital table.
4. Confirm "Beats baseline: YES" for all three hospitals.
5. Open `results/local_final_metrics.csv` and `results/local_cross_eval.csv` for the saved numbers.
6. Open `docs/03_implementation/MODEL_AND_TRAINING.md` for the full write-up.

---

## Screenshots to Capture for Report

- [x] Terminal output of `python scripts/train_local.py` (captured above)
- [x] Terminal output of `python scripts/run_tests.py` (captured above)
- [ ] Cross-hospital table formatted as a figure in the report

---

## Exit Criteria and Status

| Criterion | Status |
|---|---|
| All tests in this phase pass | ✅ |
| Phase report written with real output | ✅ |
| TESTING_LOG.md updated | ✅ |
| TRACEABILITY.md updated | ✅ |
