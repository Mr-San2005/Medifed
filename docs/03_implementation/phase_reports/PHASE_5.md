# Phase 5 — Explainability and Inference

**Date:** 2026-10-13  
**Phase:** 5 (partial; multi-disease evaluation remains open)

## Objective

Implement a reusable Grad-CAM path and checkpoint-backed predictions for the local diagnosis demonstration.

## Requirements Covered

| Requirement ID | Description |
|---|---|
| FR-08 | Grad-CAM heatmap for model predictions |

## Design Notes

Grad-CAM uses the final ResNet18 `layer4` block and removes its forward hook on every exit. Checkpoint variants are loaded from the configured checkpoint folder; FedBN variants merge the requested hospital's BatchNorm state with the saved personalized global state.

## Implementation

| File | Action | Description |
|---|---|---|
| `medifed/gradcam.py` | Created | Grad-CAM, colormap, and image overlay |
| `medifed/inference.py` | Created | Variant discovery, model loading, and image prediction |
| `scripts/demo_gradcam.py` | Created | Four-class image and prediction table |
| `tests/test_gradcam.py` | Created | Grad-CAM and overlay behavior tests |
| `tests/test_inference.py` | Created | Prediction, missing checkpoint, and FedBN state tests |

## Test Execution

| Test ID | Description | Expected | Actual | Result |
|---|---|---|---|---|
| TC-CAM-001–006 | Grad-CAM and overlay behavior | All checks pass | 6 passed | PASS |
| TC-INF-001–005 | Variant and image prediction behavior | All checks pass | 5 passed | PASS |

## Results

`python scripts/demo_gradcam.py --variant fedbn_hospital3` produced `results/gradcam_demo_fedbn_hospital3.png` from four real validation images. Hospital 3 examples for COVID19 and NORMAL were predicted correctly. The sampled Hospital 1 PNEUMONIA and Hospital 2 TUBERCULOSIS images were predicted as COVID19 by the Hospital 3 model.

## Demo Script for Guide Review

1. Run `python scripts/demo_gradcam.py --variant fedbn_hospital3`.
2. Review the printed true class, source hospital, prediction, and top-three scores.
3. Open `results/gradcam_demo_fedbn_hospital3.png` and explain that the heatmap is model-score influence, not clinical localization.

## Exit Criteria and Status

| Criterion | Status |
|---|---|
| Grad-CAM and inference tests pass | PASS |
| Real demonstration image generated | PASS |
| Pooled multi-disease and FedBN cross-hospital evaluation complete | NOT COMPLETE |