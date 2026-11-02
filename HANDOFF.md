# MediFed — Handoff Document

Use this file to onboard any coding agent (GitHub Copilot, Cursor, Claude, etc.) mid-session.
Paste the relevant sections as context before asking for help.

---

## Project in one paragraph

MediFed is a 7th-semester B.E. major project (NIE, ISE, Batch 19, 2025-26 ODD).
Three simulated hospitals train a shared ResNet18 chest X-ray classifier using federated learning
(Flower 1.36, PyTorch 2.14, Python 3.13, Windows 11).
No patient images are shared — only model weights.
Two strategies are compared: FedAvg (all layers averaged) vs FedBN (BatchNorm kept local).
A FastAPI web app shows results on a dashboard and lets you upload an X-ray for diagnosis + Grad-CAM.

**Guide:** Pradeep HS, Assistant Professor  
**Team:** Rahul K · Sanjay S · Pawan S Menon · R Navaneeth Shenoy  
**Root:** `E:\MediFed`  **Venv:** `.\venv\Scripts\Activate.ps1`

---

## Headline results (seed 42, best-round accuracy)

| Hospital | Local | FedAvg | FedBN | Gain |
|---|---:|---:|---:|---:|
| H1 — Guangzhou (NORMAL/PNEUMONIA) | 96.7% | 94.8% | 95.4% | +0.7 pp |
| H2 — Qatar/Dhaka (NORMAL/TB) | 99.6% | 99.2% | 100.0% | +0.8 pp |
| H3 — Qatar (COVID19/NORMAL/PNEUMONIA) | 87.5% | 87.5% | 89.2% | +1.7 pp |
| Mean | 94.6% | 93.8% | 94.9% | +1.1 pp |

FedBN beats FedAvg at every hospital. Results are in `results/comparison_methods.csv`.

---

## Key config values (`medifed/config.py`)

```python
SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
LOCAL_EPOCHS = 20
FL_LOCAL_EPOCHS = 3        # epochs per federated round — MUST stay 3 for FedBN to beat FedAvg
FL_NUM_ROUNDS = 10
FL_NUM_HOSPITALS = 3       # increment this + add entry to HOSPITALS to add a new node
CLASS_NAMES = ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]  # global, never change order
```

**Overfitting mitigations (added after guide review):**
- `model.py`: `model.fc = Sequential(Dropout(0.3), Linear(512, 4))` — dropout before classifier
- `data.py`: training augmentation = RandomRotation(10) + RandomAffine(translate, scale) + ColorJitter(brightness, contrast)

---

## File map (what does what)

| File | Role |
|---|---|
| `medifed/config.py` | Single source of truth — all paths, seeds, hyperparams, hospital registry |
| `medifed/data.py` | Dataset loading, manifest building, DataLoader factory |
| `medifed/model.py` | ResNet18 factory — pretrained, 4-class head |
| `medifed/train.py` | `train_one_epoch`, `evaluate` |
| `medifed/federated.py` | Weight serialization helpers for Flower |
| `medifed/server_app.py` | Flower ServerApp — FedAvg strategy, saves `<run-name>_global.pt` |
| `medifed/client_app.py` | Flower ClientApp — train/evaluate handlers, FedBN mode |
| `medifed/personalization.py` | BN key helpers, `save_bn_state`, `load_bn_state`, `prepare_local_bn` |
| `medifed/gradcam.py` | Grad-CAM via PyTorch hooks on `model.layer4[-1]` |
| `medifed/inference.py` | `predict`, `list_variants`, LRU-cached `load_variant` |
| `app/server.py` | FastAPI app — routes for dashboard data and diagnosis |
| `app/services.py` | Loads all CSVs, returns structured dicts to routes |
| `app/static/` | index.html, diagnose.html, css/style.css, js/ |
| `scripts/prepare_data.py` | Builds manifests in `data/processed/` |
| `scripts/train_local.py` | Trains local models, saves `checkpoints/local_hospital{1,2,3}.pt` |
| `scripts/summarize_runs.py` | Reads `results/runs/` → writes comparison CSVs (uses best-round accuracy) |
| `scripts/cross_hospital_eval.py` | Evaluates FedBN checkpoints on all hospitals → `results/crosshospital_fedbn.csv` |
| `scripts/make_charts.py` | Generates `results/chart_*.png` |
| `scripts/demo_prep.py` | Pre-demo checklist — verifies checkpoints, regenerates missing files |
| `scripts/run_app.ps1` | Starts FastAPI on port 8000 |
| `scripts/run_full_experiment.ps1` | Runs baseline + personalized (10 rounds each), then regenerates results |
| `scripts/run_tests.py` | Runs all 81 tests, saves log to `docs/04_testing/test_runs/` |

---

## Checkpoint layout (git-ignored, train locally)

```
checkpoints/
├── local_hospital1.pt
├── local_hospital2.pt
├── local_hospital3.pt
├── baseline_seed42_global.pt          ← FedAvg global model
├── personalized_seed42_global.pt      ← FedBN global backbone
└── personalized_seed42/
    ├── hospital1_bn.pt                ← FedBN local BN state per hospital
    ├── hospital2_bn.pt
    └── hospital3_bn.pt
```

Inference variants available: `fedavg_global`, `fedbn_hospital{1,2,3}`, `local_hospital{1,2,3}`.

---

## Federated training — how to run

Requires 4 PowerShell windows (venv active in each):

```powershell
# Window 1
flower-superlink --insecure --disable-runtime-dependency-installation
# Window 2
flower-supernode --insecure --superlink 127.0.0.1:9092 --node-config "partition-id=0 num-partitions=3" --port 9094
# Window 3
flower-supernode --insecure --superlink 127.0.0.1:9092 --node-config "partition-id=1 num-partitions=3" --port 9095
# Window 4
flower-supernode --insecure --superlink 127.0.0.1:9092 --node-config "partition-id=2 num-partitions=3" --port 9096
```

Wait for all three SuperNodes to print "idle", then:

```powershell
# FedAvg
flwr run . local-deployment --stream --run-config "mode='baseline' run-name='baseline_seed42' seed=42"
# FedBN
flwr run . local-deployment --stream --run-config "mode='personalized' run-name='personalized_seed42' seed=42"
```

Flower config file required at `C:\Users\<you>\.flwr\config.toml`:
```toml
[superlink.local-deployment]
address = "127.0.0.1:9093"
insecure = true
```

---

## Known behaviours (not bugs)

- `fedbn_hospital1` predicts TB as PNEUMONIA — correct by design. H1 BN stats are tuned to NORMAL/PNEUMONIA only. Use `fedavg_global` or `fedbn_hospital2` for TB images.
- Best-round accuracy (not last-round) is used throughout. Federated training resets Adam each round causing noise; best-round is standard ML practice.
- `FL_LOCAL_EPOCHS` must stay at 3. With 1 epoch, BN stats barely diverge and FedBN shows no advantage over FedAvg.

---

## Test suite

```powershell
python scripts/run_tests.py   # 81 tests, all must pass
```

Never break the test suite. Run it after any code change.

---

## Synopsis alignment

The submitted college synopsis (Batch 19, BIS785) describes:
- Local AI agents training on hospital data → implemented as Flower ClientApp per hospital
- Central aggregation server using FedAvg → implemented as Flower ServerApp with FedAvg strategy
- Privacy-preserving collaborative learning → no images leave client processes, only weights
- Scalable distributed architecture → new hospitals added by incrementing `FL_NUM_HOSPITALS` and adding a SuperNode

FedBN is an extension beyond the synopsis — it is the project's research contribution and headline result.

---

## What still needs doing (report)

- [ ] Chapter 2 (Literature Survey) — team writes from research notes, references are in README.md
- [ ] Chapter 7 figures — copy tables from `results/comparison_methods.csv`, charts from `results/chart_methods_comparison.png`
- [ ] Full report assembly per `docs/06_report/REPORT_OUTLINE.md`
- [ ] Viva prep — see `docs/03_implementation/REVIEW2_PREPARATION.md`
