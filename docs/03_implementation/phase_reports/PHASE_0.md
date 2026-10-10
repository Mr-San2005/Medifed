# Phase 0 — Project Foundation, Documentation Structure and Test Framework

**Date:** 2026-10-10
**Phase:** 0

---

## Objective

Establish the project skeleton: the `medifed` Python package with its configuration constants, the `unittest` test runner, and the full documentation structure covering requirements, design, testing and reporting. Every subsequent phase builds on this foundation.

---

## Requirements Covered

| Requirement ID | Description |
|---|---|
| FR-02 | One fixed global class mapping defined in `medifed/config.py` |
| NFR-03 | Reproducible runs — `SEED = 42` defined and tested |
| NFR-04 | Every requirement covered by a documented test — traceability matrix created |

---

## Design Notes

- `medifed/config.py` is the single source of truth for all paths, seeds, class names and hospital metadata. No other module may define its own class-to-index mapping.
- Paths are built with `pathlib` anchored to `__file__` so the project is relocatable.
- `scripts/run_tests.py` discovers tests via `unittest.TestLoader.discover`, captures output to a timestamped file, and exits non-zero on any failure or error.
- Documentation follows the software development life cycle: `docs/01_requirements/`, `docs/02_design/`, `docs/03_implementation/`, `docs/04_testing/`, `docs/06_report/`.

---

## Implementation

| File | Action | Description |
|---|---|---|
| `medifed/__init__.py` | Created | Package marker with one-line docstring |
| `medifed/config.py` | Created | All project constants: paths, seed, image size, class names, hospital registry |
| `tests/__init__.py` | Created | Test package marker |
| `tests/test_config.py` | Created | 8 unit tests covering config correctness |
| `scripts/run_tests.py` | Created | Test runner: discovers, runs, saves timestamped output, exits non-zero on failure |
| `results/.gitkeep` | Created | Keeps the results/ directory in git |
| `docs/04_testing/test_runs/.gitkeep` | Created | Keeps the test_runs/ directory in git |
| `README.md` | Created | Project overview, approach, hospitals, folder structure, phase status, how to run |
| `docs/01_requirements/SRS.md` | Created | Full SRS: FR-01 to FR-10, NFR-01 to NFR-05 |
| `docs/02_design/HLD.md` | Created | High-level design with Mermaid diagrams |
| `docs/02_design/LLD.md` | Created | Low-level design skeleton, one section per planned module |
| `docs/02_design/UI_DESIGN_GUIDE.md` | Created | Design tokens, layout rules, accessibility requirements |
| `docs/03_implementation/phase_reports/TEMPLATE.md` | Created | Phase report template |
| `docs/04_testing/TEST_PLAN.md` | Created | Test objectives, levels, ID scheme, entry/exit criteria |
| `docs/04_testing/TESTING_LOG.md` | Created | Log table (populated below) |
| `docs/04_testing/TRACEABILITY.md` | Created | Requirement-to-test matrix |
| `docs/04_testing/DEFECT_LOG.md` | Created | Defect table (empty — no defects this phase) |
| `docs/06_report/REPORT_OUTLINE.md` | Created | Final report chapter structure with source mapping |

---

## Test Execution

| Test ID | Description | Expected | Actual | Result |
|---|---|---|---|---|
| TC-CFG-001 | Class order is the fixed global order | `["COVID19","NORMAL","PNEUMONIA","TUBERCULOSIS"]` | Matches | PASS |
| TC-CFG-002 | CLASS_TO_INDEX matches CLASS_NAMES position for every entry | Index consistent, length equal | Consistent, length 4 | PASS |
| TC-CFG-003 | HOSPITALS contains exactly 3 entries | `len == 3` | 3 | PASS |
| TC-CFG-004 | Every hospital's class list is a subset of CLASS_NAMES | All subsets | All subsets | PASS |
| TC-CFG-005 | Hospital 3 has COVID19, NORMAL, PNEUMONIA | Sorted list matches | Matches | PASS |
| TC-CFG-006 | Hospital 2 has NORMAL, TUBERCULOSIS | Sorted list matches | Matches | PASS |
| TC-CFG-007 | SEED is an integer | `isinstance(SEED, int)` | True | PASS |
| TC-CFG-008 | PROJECT_ROOT/medifed/config.py resolves to an existing file | File exists | Exists | PASS |

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

----------------------------------------------------------------------
Ran 8 tests in 0.000s

OK

============================================================
Run:    8
Passed: 8
Failed: 0
Errors: 0
Output: E:\MediFed\docs\04_testing\test_runs\test_run_20261010_214617.txt
============================================================
```

---

## Defects Found and Fixes

None.

---

## Results

- 8 unit tests defined, 8 passed, 0 failed, 0 errors.
- Run time: 0.000 s.
- Full output saved to `docs/04_testing/test_runs/test_run_20261010_214617.txt`.

---

## Demo Script for Guide Review

1. Activate the virtual environment: `.\venv\Scripts\Activate.ps1`
2. Run the test suite: `python scripts/run_tests.py`
3. Observe: 8 tests, all OK, output file path printed.
4. Open `docs/04_testing/test_runs/test_run_20261010_214617.txt` to see the full saved output.
5. Open `medifed/config.py` to review the class registry and hospital metadata.

---

## Screenshots to Capture for Report

- [x] Terminal output of `python scripts/run_tests.py` (captured above)

---

## Exit Criteria and Status

| Criterion | Status |
|---|---|
| All tests in this phase pass | ✅ |
| Phase report written with real output | ✅ |
| TESTING_LOG.md updated | ✅ |
| TRACEABILITY.md updated | ✅ |
