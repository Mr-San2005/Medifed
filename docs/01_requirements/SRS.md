# Software Requirements Specification — MediFed

## 1. Purpose

This document specifies the functional and non-functional requirements for MediFed, a federated learning system for chest X-ray diagnosis across three simulated hospitals. It is the authoritative reference for design, implementation and testing decisions.

## 2. Scope

MediFed trains a shared ResNet18 model without centralizing patient images, compares FedAvg (baseline) against FedBN (personalized) federated strategies, and exposes results through a local web application with a Dashboard and a Diagnosis Page.

## 3. Users

- Project team members running training and evaluation locally.
- Project guide reviewing results at each phase review.
- Demonstration audience at the final presentation.

## 4. Constraints

- Windows 11, PowerShell, Python 3.13 in `.\venv`.
- GPU: NVIDIA RTX 3050 Laptop 4 GB; CPU fallback required.
- No patient images may leave a hospital client process.
- No external network calls at inference time (offline deployment).
- Allowed libraries: PyTorch, torchvision, Pillow, NumPy, Flower (flwr), standard library, and any additional package approved and pinned before use.

## 5. Assumptions

- The three Kaggle datasets have been downloaded and placed under `data/raw/hospital1`, `hospital2`, `hospital3` with class folders directly inside each hospital folder.
- Image counts vary by dataset version; the code never hardcodes counts.
- The project guide has approved the design described in PROJECT_BRIEF.md; it is not to be changed.

## 6. Out of Scope

- Real hospital network deployment or DICOM integration.
- Diseases or modalities beyond the three datasets listed.
- Automated hyperparameter search.
- Mobile or cloud deployment.

---

## 7. Functional Requirements

| ID | Requirement |
|---|---|
| FR-01 | The data pipeline loads images from all three hospital folders, skipping files whose name starts with `._`, skipping folders named `masks` (any case), `Lung_Opacity` or `__MACOSX`, and skipping any file that PIL cannot open. |
| FR-02 | One fixed global class mapping `CLASS_NAMES = ["COVID19","NORMAL","PNEUMONIA","TUBERCULOSIS"]` is defined in `medifed/config.py` and used identically by all hospitals. No hospital builds its own mapping. |
| FR-03 | The data pipeline applies a reproducible per-class cap (`MAX_IMAGES_PER_CLASS = 500`) using a fixed random seed (`SEED = 42`). |
| FR-04 | Each hospital trains a ResNet18 with the final layer replaced for 4-class output, using only its local images. |
| FR-05 | Federated training runs in FedAvg (baseline) mode: all model layers are averaged across hospitals each round. |
| FR-06 | Federated training runs in FedBN (personalized) mode: BatchNorm layers remain local to each hospital; all other layers are averaged. |
| FR-07 | After both modes are run, a comparison table and chart (accuracy per hospital, FedAvg vs FedBN) are saved to `results/`. |
| FR-08 | A Grad-CAM heatmap is generated for the top prediction of any input X-ray, implemented with PyTorch hooks (no extra library). |
| FR-09 | The Diagnosis Page accepts an uploaded X-ray, displays ranked class predictions with confidence scores, and overlays the Grad-CAM heatmap. |
| FR-10 | The Dashboard displays per-hospital training results (loss and accuracy curves, final metrics) for both federated modes. |

## 8. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-01 | Raw patient images never leave a hospital client process; only model weights are transmitted to the aggregation server. |
| NFR-02 | The system runs on a 4 GB GPU with automatic CPU fallback when CUDA is unavailable. Batch sizes are kept at 16–32 at 224 × 224. |
| NFR-03 | All random operations use `SEED = 42` so that data splits, caps and weight initialization are reproducible across runs. |
| NFR-04 | Every requirement in this document is covered by at least one documented test case in `docs/04_testing/TRACEABILITY.md`. |
| NFR-05 | The web application uses a clean, clinical visual design as specified in `docs/02_design/UI_DESIGN_GUIDE.md`; the interface is uncluttered and data-first. |
