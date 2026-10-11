# Phase 1 — Data Pipeline

**Date:** 2026-10-10
**Phase:** 1

---

## Objective

Build the complete data pipeline for all three hospitals: walk raw directories, apply all filters (junk, extension, excluded folders, corrupt files), cap each class to 500 images with a fixed seed, split into train/val, write manifests, and expose the data through a PyTorch Dataset and DataLoader. Verify the pipeline end-to-end with a batch load and a saved sample grid.

---

## Requirements Covered

| Requirement ID | Description |
|---|---|
| FR-01 | Load images with junk/corrupt/mask filtering |
| FR-02 | One fixed global class mapping for all hospitals |
| FR-03 | Reproducible per-class cap with fixed seed |
| NFR-03 | Reproducible runs — SEED=42 used for cap and split |

---

## Design Notes

- `find_images` walks the directory tree and assigns each image's class from the nearest ancestor folder whose lowercase name is in `FOLDER_TO_CLASS`. This handles both flat layouts (hospital 1/2) and nested layouts (hospital 3: `COVID19/images/x.png`).
- Paths in the manifest are stored relative to the hospital's own raw directory (not the global `RAW_DIR`). This keeps the manifest portable and avoids path reconstruction errors.
- `build_manifest` uses a single `random.Random(SEED)` instance for both the cap sample and the shuffle before splitting, so both operations are reproducible from the same seed.
- No horizontal flip augmentation: chest X-rays have a fixed left-right orientation (heart position is diagnostically significant).
- `make_loader` passes the hospital-specific raw dir to `ChestXrayDataset` so relative paths in the manifest resolve correctly.

---

## Implementation

| File | Action | Description |
|---|---|---|
| `medifed/config.py` | Modified | Added `HOSPITAL_RAW_DIRS`, `FOLDER_TO_CLASS`, `VAL_FRACTION`, `IMAGE_EXTENSIONS`; added `__MACOSX` to `EXCLUDE_FOLDER_NAMES` |
| `medifed/data.py` | Created | `find_images`, `build_manifest`, `write_manifest`, `read_manifest`, `get_transforms`, `ChestXrayDataset`, `make_loader` |
| `scripts/prepare_data.py` | Created | Runs pipeline for hospitals 1–3, prints counts, saves manifests and `results/data_counts.csv` |
| `scripts/verify_data.py` | Created | Reads manifests, checks file existence and index validity, loads one batch per hospital, saves `results/sample_grid.png` |
| `tests/test_data.py` | Created | 12 unit tests covering all filter rules, cap, seed, global indices, split, manifest I/O, dataset item shape |
| `docs/03_implementation/DATA_PIPELINE.md` | Created | Guide-review document with data flow diagram, filter rules, class mapping, results |
| `docs/02_design/LLD.md` | Modified | Updated sections 1 and 2 with Phase 1 built status and real function signatures |
| `README.md` | Modified | Updated phase status table and how-to-run section |

**Defect found and fixed during this phase:** `verify_data.py` and `make_loader` initially used the global `RAW_DIR` to reconstruct absolute paths from manifest entries, but manifest paths are relative to the hospital-specific raw dir. Fixed by passing `HOSPITAL_RAW_DIRS[hid]` in both places. See DEF-001 in DEFECT_LOG.md.

---

## Test Execution

| Test ID | Description | Expected | Actual | Result |
|---|---|---|---|---|
| TC-DATA-001 | `._` junk file not returned, counted in skipped['junk'] | junk >= 1 | junk = 1 | PASS |
| TC-DATA-002 | Corrupt image not returned, counted in skipped['unreadable'] | unreadable >= 1 | unreadable = 1 | PASS |
| TC-DATA-003 | masks/ folder images excluded and counted | excluded >= 1, masked.png absent | Correct | PASS |
| TC-DATA-004 | Lung_Opacity/ folder images excluded | lo.png absent | Absent | PASS |
| TC-DATA-005 | __MACOSX/ folder images excluded | mac.png absent | Absent | PASS |
| TC-DATA-006 | Class from nearest ancestor (COVID19/images/x.png → COVID19) | nested.png in COVID19 | Correct | PASS |
| TC-DATA-007 | Per-class cap respected | no class > cap | All ≤ cap | PASS |
| TC-DATA-008 | Same seed gives identical sample twice | rows1 == rows2 | Equal | PASS |
| TC-DATA-009 | Hospital-2-style: NORMAL=1, TUBERCULOSIS=3 (global indices) | indices match | Correct | PASS |
| TC-DATA-010 | Train/val no overlap, both non-empty | intersection = {} | Empty | PASS |
| TC-DATA-011 | Manifest write/read round-trip lossless, forward slashes | loaded == original | Lossless | PASS |
| TC-DATA-012 | Dataset item: float tensor 3×224×224, integer label | shape=(3,224,224), dtype=float32 | Correct | PASS |

All 20 tests (8 config + 12 data) passed. Run time: 3.426 s.

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
test_cap_respected (test_data.TestBuildManifest.test_cap_respected)
TC-DATA-007: per-class cap is respected; no class exceeds the cap. ... ok
test_global_indices_hospital2_style (test_data.TestBuildManifest.test_global_indices_hospital2_style)
TC-DATA-009: hospital-2-style folder gives NORMAL=1 and TUBERCULOSIS=3 (global indices). ... ok
test_same_seed_same_sample (test_data.TestBuildManifest.test_same_seed_same_sample)
TC-DATA-008: the same seed produces an identical sample on two calls. ... ok
test_train_val_no_overlap (test_data.TestBuildManifest.test_train_val_no_overlap)
TC-DATA-010: train and val splits are stratified per class and have no path overlap. ... ok
test_dataset_item_shape_and_type (test_data.TestDataset.test_dataset_item_shape_and_type)
TC-DATA-012: dataset item is a float tensor of shape 3x224x224 with an integer label. ... ok
test_corrupt_skipped (test_data.TestFindImages.test_corrupt_skipped)
TC-DATA-002: corrupt image is not returned and is counted in skipped['unreadable']. ... ok
test_junk_skipped (test_data.TestFindImages.test_junk_skipped)
TC-DATA-001: '._' junk file is not returned and is counted in skipped['junk']. ... ok
test_lung_opacity_excluded (test_data.TestFindImages.test_lung_opacity_excluded)
TC-DATA-004: images inside 'Lung_Opacity' folder are excluded. ... ok
test_macosx_excluded (test_data.TestFindImages.test_macosx_excluded)
TC-DATA-005: images inside '__MACOSX' folder are excluded. ... ok
test_masks_excluded (test_data.TestFindImages.test_masks_excluded)
TC-DATA-003: images inside a 'masks' folder are excluded and counted. ... ok
test_nearest_ancestor_class (test_data.TestFindImages.test_nearest_ancestor_class)
TC-DATA-006: class comes from the nearest matching ancestor (COVID19/images/x.png -> COVID19). ... ok
test_write_read_roundtrip (test_data.TestManifestIO.test_write_read_roundtrip)
TC-DATA-011: manifest write/read round-trip is lossless; relative paths use forward slashes. ... ok

----------------------------------------------------------------------
Ran 20 tests in 3.426s

OK

============================================================
Run:    20
Passed: 20
Failed: 0
Errors: 0
Output: E:\MediFed\docs\04_testing\test_runs\test_run_20261010_215959.txt
============================================================
```

---

## Defects Found and Fixes

| Defect ID | Description | Fix Applied |
|---|---|---|
| DEF-001 | `verify_data.py` and `make_loader` used global `RAW_DIR` to reconstruct absolute paths; manifest paths are relative to the hospital-specific raw dir, so all 766 files appeared missing | Changed both to use `HOSPITAL_RAW_DIRS[hid]` |

See `docs/04_testing/DEFECT_LOG.md` for full details.

---

## Results

- Hospital 1: NORMAL 381 found/kept (305 train, 76 val), PNEUMONIA 385 found/kept (308 train, 77 val). No files skipped.
- Hospital 2: NORMAL 600 found → 500 kept (400 train, 100 val), TUBERCULOSIS 600 found → 500 kept (400 train, 100 val). No files skipped.
- Hospital 3: COVID19 600→500 (400/100), NORMAL 600→500 (400/100), PNEUMONIA 600→500 (400/100). No files skipped.
- All manifests verified: every file exists, every class index is valid.
- Batch shape confirmed: (4, 3, 224, 224) for all three hospitals.
- Sample grid saved: `results/sample_grid.png` (3 rows × 4 tiles).
- Summary CSV saved: `results/data_counts.csv`.

---

## Demo Script for Guide Review

1. Activate the virtual environment: `.\venv\Scripts\Activate.ps1`
2. Run `python scripts/prepare_data.py` — observe per-hospital skip counts, found/kept counts, train/val split, manifest paths.
3. Run `python scripts/verify_data.py` — observe per-hospital tables, batch tensor shapes, label sets, and "ALL CHECKS PASSED".
4. Open `results/sample_grid.png` — three rows of X-ray thumbnails, one row per hospital, each tile labelled with its class name.
5. Run `python scripts/run_tests.py` — observe 20 tests, all OK.
6. Open `docs/03_implementation/DATA_PIPELINE.md` for the full pipeline description and results table.

---

## Screenshots to Capture for Report

- [x] Terminal output of `python scripts/prepare_data.py` (captured above)
- [x] Terminal output of `python scripts/verify_data.py` (captured above)
- [x] Terminal output of `python scripts/run_tests.py` (captured above)
- [ ] `results/sample_grid.png` — include in report as Figure: Sample X-rays per hospital

---

## Exit Criteria and Status

| Criterion | Status |
|---|---|
| All tests in this phase pass | ✅ |
| Phase report written with real output | ✅ |
| TESTING_LOG.md updated | ✅ |
| TRACEABILITY.md updated | ✅ |
