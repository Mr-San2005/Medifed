# MediFed — Decentralized Medical Diagnostics via Federated Learning

7th-semester B.E. major project · Information Science and Engineering · Team of 4

---

## What this project does

Three simulated hospitals train one shared chest X-ray classifier **without sharing any patient images**.
Each hospital trains locally and sends only model weights to a central aggregation server (federated learning).

Two strategies are compared:

| Strategy | What gets averaged |
|---|---|
| FedAvg (baseline) | All model layers |
| FedBN (personalized) | All layers except BatchNorm, which stays local per hospital |

A web dashboard shows the results and a diagnosis page lets you upload an X-ray and get a ranked prediction with a Grad-CAM heatmap.

---

## Headline results (seed 42, best-round accuracy)

| Hospital | Local | FedAvg | FedBN | FedBN gain |
|---|---:|---:|---:|---:|
| H1 — Guangzhou (NORMAL / PNEUMONIA) | 96.7% | 94.8% | 95.4% | +0.7 pp |
| H2 — Qatar/Dhaka (NORMAL / TB) | 99.6% | 99.2% | 100.0% | +0.8 pp |
| H3 — Qatar (COVID19 / NORMAL / PNEUMONIA) | 87.5% | 87.5% | 89.2% | +1.7 pp |
| **Mean** | **94.6%** | **93.8%** | **94.9%** | **+1.1 pp** |

FedBN beats FedAvg at every hospital. Full CSVs and charts are in `results/`.

---

## Repository layout

```
MediFed/
├── medifed/          # Python package — config, data, model, federated, app modules
├── app/              # FastAPI web app (dashboard + diagnosis page)
├── scripts/          # Runnable scripts and PowerShell helpers
├── tests/            # unittest test suite (81 tests)
├── docs/
│   ├── 01_requirements/   # SRS.md
│   ├── 02_design/         # HLD.md, LLD.md, UI_DESIGN_GUIDE.md
│   ├── 03_implementation/ # phase_reports/, REVIEW2_PREPARATION.md, …
│   ├── 04_testing/        # TEST_PLAN.md, TESTING_LOG.md, TRACEABILITY.md, DEFECT_LOG.md
│   └── 06_report/         # REPORT_OUTLINE.md  ← start here for the report
├── results/          # Committed CSVs, PNGs — all experiment outputs
├── data/             # Raw images — git-ignored, download separately
├── checkpoints/      # Model weights — git-ignored, train locally
├── pyproject.toml    # Flower app config
└── requirements.txt  # Pinned Python dependencies
```

---

## Prerequisites

- Windows 11, PowerShell
- Python 3.13
- NVIDIA GPU recommended (CPU fallback works but is slow for training)
- Kaggle account (to download the three datasets)

---

## Setup (do this once after cloning)

### 1. Create and activate the virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
pip install -e .
```

> `requirements.txt` pulls PyTorch with CUDA 12.6 support. If you are on CPU only, replace the
> `--extra-index-url` line and the `torch`/`torchvision` lines with the CPU wheels from
> https://pytorch.org/get-started/locally/

### 3. Install additional runtime dependencies

```powershell
pip install fastapi uvicorn pillow matplotlib
```

### 4. Download the three datasets from Kaggle

Place them under `data/raw/` exactly as shown:

```
data/raw/
├── hospital1/
│   ├── NORMAL/        ← from paultimothymooney/chest-xray-pneumonia
│   └── PNEUMONIA/
├── hospital2/
│   ├── NORMAL/        ← from tawsifurrahman/tuberculosis-tb-chest-xray-dataset
│   └── TUBERCULOSIS/
└── hospital3/
    ├── COVID19/       ← from tawsifurrahman/covid19-radiography-database
    ├── NORMAL/
    └── PNEUMONIA/
```

Class folders must sit **directly** under each hospital folder (no extra nesting).

---

## Running the project

All commands assume the venv is active and you are in the `MediFed/` root.

### Verify the data pipeline

```powershell
python scripts/prepare_data.py   # builds manifests in data/processed/
python scripts/verify_data.py    # prints counts, saves results/sample_grid.png
```

### Train local models (no federation)

```powershell
python scripts/train_local.py
```

Saves `checkpoints/local_hospital{1,2,3}.pt` and prints per-hospital accuracy.

### Run federated training

Open **four** PowerShell windows, all with the venv active:

**Window 1 — SuperLink**
```powershell
flower-superlink --insecure --disable-runtime-dependency-installation
```

**Windows 2, 3, 4 — one SuperNode per hospital**
```powershell
# Window 2
flower-supernode --insecure --superlink 127.0.0.1:9092 --node-config "partition-id=0 num-partitions=3" --port 9094
# Window 3
flower-supernode --insecure --superlink 127.0.0.1:9092 --node-config "partition-id=1 num-partitions=3" --port 9095
# Window 4
flower-supernode --insecure --superlink 127.0.0.1:9092 --node-config "partition-id=2 num-partitions=3" --port 9096
```

Wait until all three SuperNodes print "idle". Then in a fifth window:

**FedAvg baseline (10 rounds)**
```powershell
flwr run . local-deployment --stream --run-config "mode='baseline' run-name='baseline_seed42' seed=42"
```

**FedBN personalized (10 rounds)**
```powershell
flwr run . local-deployment --stream --run-config "mode='personalized' run-name='personalized_seed42' seed=42"
```

Or use the helper script that does both automatically:
```powershell
.\scripts\run_full_experiment.ps1
```

### Generate result CSVs and charts

```powershell
python scripts/summarize_runs.py
python scripts/cross_hospital_eval.py
python scripts/make_charts.py
```

### Run the web app

```powershell
.\scripts\run_app.ps1
```

Open `http://127.0.0.1:8000`. The dashboard shows all results; the diagnosis page lets you upload an X-ray.

### Run the test suite

```powershell
python scripts/run_tests.py
```

Output is saved to `docs/04_testing/test_runs/test_run_YYYYMMDD_HHMMSS.txt`.

---

## Flower config file (one-time setup)

Flower's deployment runtime needs a config file at `C:\Users\<you>\.flwr\config.toml`:

```toml
[superlink.local-deployment]
address = "127.0.0.1:9093"
insecure = true
```

---

## For teammates writing the report

The report structure is mapped out in [`docs/06_report/REPORT_OUTLINE.md`](docs/06_report/REPORT_OUTLINE.md).
Each chapter points to the exact source file to draw from.

| What you need | Where to find it |
|---|---|
| Problem statement, scope, objectives | `docs/01_requirements/SRS.md` |
| System architecture, component diagram | `docs/02_design/HLD.md` |
| Detailed design, class/sequence diagrams | `docs/02_design/LLD.md` |
| UI design decisions | `docs/02_design/UI_DESIGN_GUIDE.md` |
| Phase-by-phase implementation notes | `docs/03_implementation/phase_reports/` |
| Data pipeline details | `docs/03_implementation/DATA_PIPELINE.md` |
| Model and training details | `docs/03_implementation/MODEL_AND_TRAINING.md` |
| Grad-CAM explainability | `docs/03_implementation/EXPLAINABILITY.md` |
| Web app design | `docs/03_implementation/WEB_APP.md` |
| Test plan and test cases | `docs/04_testing/TEST_PLAN.md` |
| Test execution log | `docs/04_testing/TESTING_LOG.md` |
| Requirements traceability matrix | `docs/04_testing/TRACEABILITY.md` |
| Defect log | `docs/04_testing/DEFECT_LOG.md` |
| Headline results table | `results/comparison_methods.csv` |
| Round-by-round accuracy curves | `results/summary_rounds.csv` |
| Cross-hospital generalization matrix | `results/crosshospital_fedbn.csv` |
| Method comparison chart (PNG) | `results/chart_methods_comparison.png` |
| Grad-CAM example images | `results/gradcam_demo_*.png` |
| Viva Q&A and rubric alignment | `docs/03_implementation/REVIEW2_PREPARATION.md` |

---

## Key design decisions (for the report)

- **Why FedBN over FedAvg?** Each hospital has a different scanner and patient population. BatchNorm statistics encode the local pixel intensity distribution. Keeping them local (FedBN) lets the shared backbone generalize while each site's normalization adapts — this is the personalization benefit.
- **Why ResNet18?** Pretrained on ImageNet, small enough to fit on a 4 GB GPU, strong enough for transfer learning on chest X-rays.
- **Why best-round accuracy?** Federated training resets the Adam optimizer each round, causing round-to-round noise. Reporting the best checkpoint is standard ML practice and makes the FedAvg/FedBN comparison fair.
- **Why Grad-CAM without a library?** Implemented with PyTorch hooks directly on `model.layer4[-1]` to avoid extra dependencies and to demonstrate understanding of the technique.

---

## Datasets (cite these in the report)

1. Kermany, D. et al. — Chest X-Ray Images (Pneumonia). Kaggle: `paultimothymooney/chest-xray-pneumonia`
2. Rahman, T. et al. — Tuberculosis (TB) Chest X-ray Database. Kaggle: `tawsifurrahman/tuberculosis-tb-chest-xray-dataset`
3. Rahman, T. et al. — COVID-19 Radiography Database. Kaggle: `tawsifurrahman/covid19-radiography-database`

## Key references (cite these in the report)

- McMahan, B. et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *(FedAvg)*
- Li, X. et al. (2021). FedBN: Federated Learning on Non-IID Features via Local Batch Normalization. *(FedBN)*
- He, K. et al. (2016). Deep Residual Learning for Image Recognition. *(ResNet)*
- Selvaraju, R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *(Grad-CAM)*
- Beutel, D. et al. (2022). Flower: A Friendly Federated Learning Framework. *(Flower)*
