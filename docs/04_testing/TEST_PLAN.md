# Test Plan — MediFed

## 1. Objectives

- Verify that every functional and non-functional requirement in the SRS is exercised by at least one test.
- Catch regressions between phases by running the full suite after every change.
- Produce a permanent, timestamped record of every test run.

## 2. Scope

All Python modules in `medifed/`, all scripts in `scripts/`, and the web application in `app/`. Data files and model checkpoints are not under test directly; they are exercised through the data pipeline and training tests.

## 3. Test Levels

| Level | Description | Tools |
|---|---|---|
| Unit | Individual functions and classes in isolation | `unittest` |
| Integration | Interactions between modules (e.g. data pipeline feeding the model) | `unittest` |
| System | End-to-end federated training run with all three hospital clients | Manual + `unittest` |
| Acceptance | Guide review: demo script executed, results match expected outcomes | Manual |

## 4. Environment

- Windows 11, PowerShell.
- Python 3.13 in `.\venv`.
- All tests run with `python scripts/run_tests.py` from the project root.
- GPU (CUDA) used when available; tests must also pass on CPU.

## 5. Test-Case ID Scheme

```
TC-<MODULE>-NNN
```

| Module code | Module |
|---|---|
| CFG | `medifed/config.py` |
| DATA | `medifed/data.py` |
| MODEL | `medifed/model.py` |
| TRAIN | `medifed/trainer.py` |
| FED | Federated client/server |
| PERS | `medifed/personalization.py` |
| EVAL | `medifed/evaluation.py` |
| GCAM | `medifed/gradcam.py` |
| APP | Web application |

## 6. Entry Criteria

- The virtual environment is activated and all required packages are installed.
- `data/raw/hospital1`, `hospital2`, `hospital3` exist with at least one image per class.
- `python scripts/run_tests.py` can be executed without import errors.

## 7. Exit Criteria

- All tests in the suite pass (zero failures, zero errors).
- Test output is saved to `docs/04_testing/test_runs/` with a timestamp.
- `docs/04_testing/TESTING_LOG.md` is updated with results from the run.
- `docs/04_testing/TRACEABILITY.md` shows every requirement covered.

## 8. Pass / Fail Rules

- A test **passes** if it completes without raising an assertion error or exception.
- A test **fails** if it raises `AssertionError`.
- A test **errors** if it raises any other exception (treated as a defect in the test or the code under test).
- A phase is not complete until all tests introduced in that phase pass.

## 9. How to Run

```powershell
.\venv\Scripts\Activate.ps1
python scripts/run_tests.py
```

The script exits with code 0 on success and 1 on any failure or error.

## 10. How Results Are Recorded

- Full output is saved automatically to `docs/04_testing/test_runs/test_run_YYYYMMDD_HHMMSS.txt`.
- A summary row is added to `docs/04_testing/TESTING_LOG.md` after each phase.
- Defects are logged in `docs/04_testing/DEFECT_LOG.md`.
- Requirement coverage is tracked in `docs/04_testing/TRACEABILITY.md`.
