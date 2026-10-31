# Phase 6 — Local Dashboard and Diagnosis Interface

**Date:** 2026-10-13  
**Phase:** 6

## Objective

Expose saved MediFed results in a local dashboard and provide an upload-to-prediction workflow backed by saved model checkpoints.

## Requirements Covered

| Requirement ID | Description |
|---|---|
| FR-09 | Diagnosis page with ranked predictions and Grad-CAM |
| FR-10 | Dashboard with hospital results and federated round curves |
| NFR-05 | Clinical, uncluttered local interface |

## Design Notes

The app binds to `127.0.0.1`, uses installed FastAPI/Starlette/Uvicorn packages, and serves plain HTML, CSS, and JavaScript. Results are read from CSV files. Uploaded bytes are bounded to 10 MiB and processed in memory.

## Implementation

| File | Action | Description |
|---|---|---|
| `app/server.py` | Created | Local routes, error handling, headers, and bounded image body |
| `app/services.py` | Created | CSV aggregation and inference service calls |
| `app/static/` | Created | Dashboard, charts, and diagnosis interface |
| `scripts/run_app.ps1` | Created | PowerShell launcher |
| `tests/test_app.py` | Created | Service and loopback HTTP tests |

## Test Execution

| Test ID | Description | Expected | Actual | Result |
|---|---|---|---|---|
| TC-APP-001–010 | Results aggregation and HTTP behavior | All checks pass | 10 passed | PASS |

## Demo Script for Guide Review

1. Run `.\scripts\run_app.ps1` from `E:\MediFed` in PowerShell.
2. Open `http://127.0.0.1:8000` and show the results dashboard.
3. Open `/diagnose`, select an available checkpoint variant, choose an image, and run the prediction.
4. Adjust the Grad-CAM overlay opacity and explain the non-clinical limitations.

## Exit Criteria and Status

| Criterion | Status |
|---|---|
| API and service tests pass | PASS |
| Dashboard and diagnosis routes serve HTML | PASS |
| Browser visual review on the demonstration machine | NOT RUN |
