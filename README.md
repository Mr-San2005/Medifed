# MediFed — Decentralized Medical Diagnostics via Federated Learning

## Problem

Hospitals hold chest X-ray images that are clinically valuable for training diagnostic models, but patient-privacy regulations and institutional policy prevent those images from being shared or centralized. A model trained on a single hospital's data is limited by that hospital's patient population and equipment.

## Approach

MediFed trains one shared ResNet18 model across three simulated hospitals using **federated learning**: each hospital trains locally on its own images and shares only model weights with a central aggregation server. No raw image ever leaves a hospital.

Two modes are implemented and compared:

| Mode | Description |
|---|---|
| FedAvg (baseline) | All layers averaged across hospitals each round |
| FedBN (personalized) | BatchNorm layers kept local; all other layers averaged |

The headline result is a saved before/after comparison of FedAvg versus FedBN accuracy per hospital.

## The Three Hospitals

| ID | Location | Dataset | Classes |
|---|---|---|---|
| Hospital 1 | Guangzhou, China | paultimothymooney/chest-xray-pneumonia | NORMAL, PNEUMONIA |
| Hospital 2 | Qatar Univ + Dhaka | tawsifurrahman/tuberculosis-tb-chest-xray-dataset | NORMAL, TUBERCULOSIS |
| Hospital 3 | Qatar Univ et al. | tawsifurrahman/covid19-radiography-database | COVID19, NORMAL, PNEUMONIA |

## Model

- Architecture: ResNet18 (pretrained on ImageNet), final fully-connected layer replaced for 4-class output.
- Input: 224 × 224 RGB chest X-ray.
- Output: probability over `[COVID19, NORMAL, PNEUMONIA, TUBERCULOSIS]`.
- Explainability: Grad-CAM heatmap on the top prediction, implemented with PyTorch hooks.

## Folder Structure

```
MediFed/
├── medifed/          # Python package (config, data, model, federated, app modules)
├── scripts/          # Runnable scripts (run_tests.py, prepare_data.py, …)
├── tests/            # unittest test suite
├── docs/
│   ├── 01_requirements/   # SRS.md
│   ├── 02_design/         # HLD.md, LLD.md, UI_DESIGN_GUIDE.md
│   ├── 03_implementation/ # phase_reports/
│   ├── 04_testing/        # TEST_PLAN.md, TESTING_LOG.md, TRACEABILITY.md, DEFECT_LOG.md, test_runs/
│   └── 06_report/         # REPORT_OUTLINE.md
├── results/          # Committed outputs (CSV, JSON, PNG)
├── data/             # Raw and processed images (git-ignored)
├── checkpoints/      # Model weights (git-ignored)
└── app/              # Web application (later phase)
```

## Phase Status

| Phase | Description | Status |
|---|---|---|
| 0 | Skeleton, config, test runner, documentation structure | Done |
| 1 | Data pipeline (3 hospitals, filters, cap, verified counts) | Done |
| 2 | ResNet18 4-class model and local training sanity check | Done |
| 3 | Flower FedAvg baseline with 3 hospital clients | Implemented; federation run pending |
| 4 | FedBN personalized mode and baseline vs personalized comparison | Not started |
| 5 | Grad-CAM and multi-disease demo | Not started |
| 6 | Web app (Dashboard and Diagnosis Page) | Not started |
| 7 | Final testing and report pack | Not started |

## How to Run

Activate the virtual environment first:

```powershell
.\venv\Scripts\Activate.ps1
```

Run the full test suite:

```powershell
python scripts/run_tests.py
```

Build manifests for all three hospitals (filters, cap, train/val split):

```powershell
python scripts/prepare_data.py
```

Verify manifests, load a batch from each hospital, save sample grid:

```powershell
python scripts/verify_data.py
```

Train local ResNet18 models at all three hospitals (saves checkpoints, prints metrics):

```powershell
python scripts/train_local.py
```

Start the Flower deployment windows (add `-Cpu` to force CPU on the SuperNodes):

```powershell
.\scripts\start_federation.ps1
```

After all three SuperNodes are ready, run the baseline seeds:

```powershell
.\scripts\run_experiments.ps1 -Mode baseline -Seeds 42,43,44
```

Stop the Flower processes and check their ports:

```powershell
.\scripts\stop_federation.ps1
```

Test output is saved automatically to `docs/04_testing/test_runs/test_run_YYYYMMDD_HHMMSS.txt`.
