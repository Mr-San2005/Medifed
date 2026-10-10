# Low-Level Design — MediFed

Each section covers one planned module. Status is updated at the end of each phase.

---

## 1. `medifed/config.py` — Configuration Constants

**Status: Phase 0 — Built**

**Purpose:** Single source of truth for all paths, seeds, class names and hospital metadata. No logic beyond path construction.

**Contents:**
- `PROJECT_ROOT`, `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `CHECKPOINT_DIR`, `RESULTS_DIR` — `pathlib.Path` objects anchored to `__file__`.
- `SEED`, `IMAGE_SIZE`, `MAX_IMAGES_PER_CLASS` — scalar constants.
- `CLASS_NAMES`, `CLASS_TO_INDEX` — global class registry.
- `EXCLUDE_FOLDER_NAMES` — folders never treated as class folders.
- `HOSPITALS` — dict keyed by hospital ID with display name, Kaggle slug and local class list.

---

## 2. `medifed/data.py` — Data Pipeline

**Status: Phase 1 — Planned**

**Purpose:** Load, filter, cap and transform images for a given hospital into a PyTorch Dataset.

**Planned functions:**

| Function | Inputs | Output |
|---|---|---|
| `list_image_paths(hospital_dir, classes)` | Path to hospital folder, list of class names | `dict[class_name -> [Path]]` of valid image paths after all filters |
| `apply_cap(paths_by_class, cap, seed)` | Dict of paths, integer cap, int seed | Same dict with each list capped to `cap` items |
| `build_dataset(hospital_id, split, transform)` | Hospital ID (1/2/3), split name, torchvision transform | `torch.utils.data.Dataset` yielding `(tensor, global_index)` |
| `get_dataloader(dataset, batch_size, shuffle)` | Dataset, batch size, shuffle flag | `DataLoader` with `num_workers=0` |

**Filtering rules (from FR-01):**
- Skip filenames starting with `._`.
- Skip folders in `EXCLUDE_FOLDER_NAMES` and `__MACOSX`.
- Skip files PIL cannot open.

---

## 3. `medifed/model.py` — Model Definition

**Status: Phase 2 — Planned**

**Purpose:** Construct and return the ResNet18 model with a 4-class output head.

**Planned functions:**

| Function | Inputs | Output |
|---|---|---|
| `build_model(num_classes, pretrained)` | Number of output classes (4), bool | ResNet18 with replaced `fc` layer |
| `get_device()` | — | `torch.device` (CUDA if available, else CPU) |

---

## 4. `medifed/trainer.py` — Local Training Loop

**Status: Phase 2 — Planned**

**Purpose:** Run one or more local epochs of SGD on a hospital's dataset.

**Planned functions:**

| Function | Inputs | Output |
|---|---|---|
| `train_one_epoch(model, loader, optimizer, criterion, device)` | Model, DataLoader, optimizer, loss fn, device | `(avg_loss, accuracy)` |
| `evaluate(model, loader, criterion, device)` | Model, DataLoader, loss fn, device | `(avg_loss, accuracy)` |

---

## 5. `medifed/client_app.py` — Flower Client

**Status: Phase 3 — Planned**

**Purpose:** Flower ClientApp that receives global weights, runs local training, and returns updated weights and metrics.

**Planned handlers:**
- `@app.train()` — load weights from message, run `train_one_epoch`, return updated weights + `num_examples`.
- `@app.evaluate()` — load weights, run `evaluate`, return loss + `num_examples`.

**FedBN note:** In personalized mode, BatchNorm layer weights are excluded from the received global weights and restored from a local checkpoint file before training.

---

## 6. `medifed/server_app.py` — Flower Server

**Status: Phase 3 — Planned**

**Purpose:** Flower ServerApp that runs FedAvg aggregation for the configured number of rounds.

**Planned handlers:**
- `@app.main()` — initialize model, run `strategy.start(...)`, save final weights to `checkpoints/`.

---

## 7. `medifed/personalization.py` — FedBN Logic

**Status: Phase 4 — Planned**

**Purpose:** Utilities for separating BatchNorm parameters from the rest of the model state dict, and for saving/loading per-hospital BatchNorm checkpoints between rounds.

**Planned functions:**

| Function | Inputs | Output |
|---|---|---|
| `split_state_dict(state_dict)` | Full model state dict | `(shared_dict, bn_dict)` |
| `merge_state_dict(shared_dict, bn_dict)` | Two partial dicts | Full state dict |
| `save_bn_state(bn_dict, hospital_id)` | BN state dict, hospital ID | Writes to `checkpoints/bn_hospital{id}.pt` |
| `load_bn_state(hospital_id)` | Hospital ID | BN state dict or `None` if not found |

---

## 8. `medifed/evaluation.py` — Evaluation and Results

**Status: Phase 4 — Planned**

**Purpose:** Compute per-hospital and aggregate metrics; save comparison table and chart.

**Planned functions:**

| Function | Inputs | Output |
|---|---|---|
| `compute_metrics(model, loader, device)` | Model, DataLoader, device | `dict` with accuracy, per-class precision/recall |
| `save_comparison(fedavg_results, fedbn_results)` | Two result dicts | CSV + PNG saved to `results/` |

---

## 9. `medifed/gradcam.py` — Grad-CAM Explainability

**Status: Phase 5 — Planned**

**Purpose:** Generate a Grad-CAM heatmap for the top prediction of a single image using PyTorch hooks.

**Planned functions:**

| Function | Inputs | Output |
|---|---|---|
| `compute_gradcam(model, image_tensor, target_class)` | Model, `(1,3,H,W)` tensor, class index | `(H,W)` numpy heatmap in `[0,1]` |
| `overlay_heatmap(image_pil, heatmap)` | PIL image, numpy heatmap | PIL image with coloured overlay |

---

## 10. `app/` — Web Application

**Status: Phase 6 — Planned**

**Purpose:** Local Flask web app with two pages: Dashboard and Diagnosis Page.

**Planned modules:**
- `app/app.py` — Flask routes.
- `app/templates/dashboard.html` — per-hospital training curves and final metrics.
- `app/templates/diagnosis.html` — upload form, ranked predictions, heatmap overlay.
- `app/static/` — CSS (design tokens from `UI_DESIGN_GUIDE.md`), minimal JS.
