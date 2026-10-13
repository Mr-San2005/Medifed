# Low-Level Design — MediFed

Each section covers one planned module. Status is updated at the end of each phase.

---

## 1. `medifed/config.py` — Configuration Constants

**Status: Phase 1 — Built**

**Purpose:** Single source of truth for all paths, seeds, class names and hospital metadata. No logic beyond path construction.

**Contents:**
- `PROJECT_ROOT`, `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `CHECKPOINT_DIR`, `RESULTS_DIR` — `pathlib.Path` objects anchored to `__file__`.
- `SEED`, `IMAGE_SIZE`, `MAX_IMAGES_PER_CLASS`, `VAL_FRACTION`, `IMAGE_EXTENSIONS` — scalar constants.
- `CLASS_NAMES`, `CLASS_TO_INDEX` — global class registry.
- `EXCLUDE_FOLDER_NAMES` — folders never treated as class folders (`masks`, `Lung_Opacity`, `__MACOSX`).
- `HOSPITAL_RAW_DIRS` — maps hospital ID (1/2/3) to its raw data path.
- `FOLDER_TO_CLASS` — maps lowercase folder name to global class name, per hospital.
- `HOSPITALS` — dict keyed by hospital ID with display name, Kaggle slug and local class list.

---

## 2. `medifed/data.py` — Data Pipeline

**Status: Phase 1 — Built**

**Purpose:** Load, filter, cap and transform images for a given hospital into a PyTorch Dataset and DataLoader.

**Functions:**

| Function | Inputs | Output |
|---|---|---|
| `find_images(raw_dir, folder_to_class)` | Hospital raw dir `Path`, `dict[str->str]` folder-to-class map | `(images_by_class: dict[str->[Path]], skipped: dict[str->int])` |
| `build_manifest(images_by_class, raw_dir, cap, val_fraction, seed)` | Dict of paths, raw dir, cap int, fraction float, seed int | `list[dict]` rows with `relative_path`, `class_name`, `class_index`, `split` |
| `write_manifest(rows, path)` | List of row dicts, output `Path` | Writes CSV to `path` |
| `read_manifest(path)` | CSV `Path` | `list[dict]` with `class_index` as `int` |
| `get_transforms(train)` | `bool` | `transforms.Compose` (Resize, ToTensor, Normalize; +RandomRotation(5) if train) |
| `ChestXrayDataset.__init__(rows, raw_dir, transform)` | Manifest rows, raw dir, transform | Dataset instance |
| `ChestXrayDataset.__getitem__(idx)` | Index | `(float tensor 3×224×224, int class_index)` |
| `make_loader(hospital_id, split, batch_size, shuffle)` | Hospital ID, split name, batch size, shuffle bool | `DataLoader` with `num_workers=0` |

**Filtering rules (FR-01):**
- Skip filenames starting with `._`.
- Skip files with extensions not in `IMAGE_EXTENSIONS`.
- Skip files inside folders in `EXCLUDE_FOLDER_NAMES` (case-insensitive).
- Skip files PIL cannot open.

**Class assignment:** nearest ancestor folder whose lowercase name is in `folder_to_class`.

---

## 3. `medifed/model.py` — Model Definition

**Status: Phase 2 — Built**

**Purpose:** Construct the ResNet18 model with a 4-class output head, provide device selection and seed utilities.

**Functions:**

| Function | Inputs | Output |
|---|---|---|
| `build_model(pretrained=True)` | `bool` | ResNet18 with `fc` replaced by `nn.Linear(512, NUM_CLASSES)` |
| `get_device()` | — | `torch.device` (CUDA if available, else CPU) |
| `set_seed(seed)` | `int` | Seeds Python `random`, NumPy and PyTorch (CPU + CUDA) |

---

## 4. `medifed/train.py` — Local Training Loop

**Status: Phase 2 — Built**

**Purpose:** Plain functions for one training epoch, full evaluation with metrics, and a multi-epoch fit loop.

**Functions:**

| Function | Inputs | Output |
|---|---|---|
| `train_one_epoch(model, loader, optimizer, loss_fn, device)` | Model, DataLoader, Adam optimizer, CrossEntropyLoss, device | `(mean_loss: float, accuracy: float)` |
| `evaluate(model, loader, device)` | Model, DataLoader, device | `dict` with `loss`, `accuracy`, `num_samples`, `confusion_matrix` (NUM_CLASSES×NUM_CLASSES list), `per_class_recall` (list, None for absent classes) |
| `fit(model, train_loader, val_loader, epochs, lr, device)` | Model, two DataLoaders, int, float, device | `list[dict]` one per epoch: `epoch`, `train_loss`, `train_acc`, `val_loss`, `val_acc` |

---

## 5. `medifed/client_app.py` — Flower Client

**Status: built in Phase 3 Part 1, federation run pending**

**Purpose:** Flower 1.36 `ClientApp` for one hospital. Receives the global `state_dict`, trains locally, returns CPU weights and scalar metrics. Images never leave the process (NFR-01).

Hospital identity comes from `context.node_config["partition-id"]` via `partition_to_hospital`. The run name, seed and mode come from `context.run_config["run-name"]`, `context.run_config["seed"]` and `context.run_config["mode"]`. The current round comes from `msg.content["config"]["server-round"]` (injected by `FedAvg.configure_train` / `configure_evaluate` in flwr 1.36.0). Only baseline mode is enabled; requesting personalized mode raises `ValueError("personalized mode is added in Phase 4")`.

**Handlers:**

| Function | Inputs | Output |
|---|---|---|
| `train(msg, context)` | Flower `Message`, `Context` | `Message` reply with `ArrayRecord` of the complete CPU `state_dict` and `MetricRecord(train_loss, train_acc, num-examples)` |
| `evaluate(msg, context)` | Flower `Message`, `Context` | `Message` reply with `MetricRecord(eval_loss, eval_acc, num-examples)` |

Each round the client: `set_seed(seed + 1000 * hospital_id + server_round)`, `build_model(pretrained=False)`, load received weights, run `local_update` for `FL_LOCAL_EPOCHS` with a fresh Adam optimizer, and write one train row and (in `evaluate`) one val row to `results/runs/<run-name>/hospital<n>_rounds.csv` (columns: `round, hospital, split, loss, accuracy, num_samples`). At round 1, `reset_round_csv_for_first_round` removes an earlier file so a repeated run starts with fresh rows. One file per hospital prevents processes from sharing a writer. If fallback manifests are missing and `MEDIFED_ROOT` is unset, the client directs the operator to `scripts/start_federation.ps1`.

**FedBN note:** In personalized mode (Phase 4), BatchNorm layer weights are excluded from the received global weights and restored from a local checkpoint file before training.

---

## 5b. `medifed/federated.py` — Federated helpers

**Status: built in Phase 3 Part 1, federation run pending**

**Purpose:** Plain helpers with no Flower imports, used by the client app and by unit tests.

**Functions:**

| Function | Inputs | Output |
|---|---|---|
| `partition_to_hospital(partition_id)` | Flower partition id `int` | Hospital id `1/2/3`; `ValueError` for any other value |
| `validate_mode(mode, allowed)` | Mode string, allowed mode tuple | Mode string; `ValueError` listing accepted values when invalid |
| `reset_round_csv_for_first_round(csv_path, server_round)` | CSV `Path`, round number | Removes an existing CSV only for round 1 |
| `local_update(model, loader, epochs, lr, device)` | Model, DataLoader, int, float, device | `dict` with `train_loss`, `train_acc` after `epochs` of `train_one_epoch` with a fresh Adam optimizer |
| `state_dict_to_cpu(model)` | Model | `state_dict` with every tensor on CPU |
| `append_round_row(csv_path, row_dict)` | `Path`, row `dict` | Creates parent folders, writes the header only when the file is new, appends one row |

---

## 6. `medifed/server_app.py` — Flower Server

**Status: built in Phase 3 Part 1, federation run pending**

**Purpose:** Flower 1.36 `ServerApp` that runs plain FedAvg over the complete `state_dict` (weights and BatchNorm statistics), weighted by `num-examples`.

**Handlers:**

| Function | Inputs | Output |
|---|---|---|
| `main(grid, context)` | Flower `Grid`, `Context` | `None`. Reads `num-server-rounds`, `run-name`, `seed` and `mode` from `context.run_config`. It validates baseline mode, calls `set_seed(seed)`, builds `build_model(pretrained=True)` and sends it as `ArrayRecord(model.state_dict())`. |

`FedAvg` is constructed with `fraction_train=1.0`, `fraction_evaluate=1.0`, `min_train_nodes=3`, `min_evaluate_nodes=3`, `min_available_nodes=3` (parameter names from `FedAvg.__init__` in flwr 1.36.0). `train_config` is `ConfigRecord({"lr": LEARNING_RATE})`. After `strategy.start`, the final weights `result.arrays.to_torch_state_dict()` are saved to `checkpoints/<run-name>_global.pt` and the process prints `FEDERATED RUN FINISHED: <run-name>`.

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
