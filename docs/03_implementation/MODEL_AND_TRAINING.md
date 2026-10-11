# Model and Local Training — MediFed

## Model Choice and Transfer Learning

The model is a ResNet18 pretrained on ImageNet. Transfer learning is used because each hospital's local dataset is small (at most 500 images per class after the cap), which is insufficient to train a deep convolutional network from scratch without overfitting. The pretrained backbone provides a strong feature extractor for visual patterns; only the final fully-connected layer is replaced and the entire network is fine-tuned.

ResNet18 is chosen over larger variants (ResNet50, VGG) because it fits comfortably within the 4 GB GPU memory budget at batch size 16 and 224 × 224 resolution, and its residual connections make it stable to fine-tune.

## 4-Class Output Head and Global Class Mapping

The original ResNet18 FC layer (1000 ImageNet classes) is replaced by a single `nn.Linear(512, 4)` layer. The four output indices correspond to the global class order defined in `medifed/config.py`:

| Index | Class |
|---|---|
| 0 | COVID19 |
| 1 | NORMAL |
| 2 | PNEUMONIA |
| 3 | TUBERCULOSIS |

Every hospital uses this same 4-class head. A hospital that does not have a particular class (e.g. Hospital 1 has no COVID19 or TUBERCULOSIS) will never see those labels in training, but the head still has four outputs so that federated weight averaging across hospitals is always valid — all models have identical architecture.

## Hyperparameters

| Parameter | Value | Reason |
|---|---|---|
| `BATCH_SIZE` | 16 | Keeps peak GPU memory under 4 GB at 224 × 224 |
| `LOCAL_EPOCHS` | 5 | Enough to confirm the model learns from each hospital's data; full convergence is not the goal at this phase |
| `LEARNING_RATE` | 1e-4 | Conservative rate for fine-tuning a pretrained backbone; avoids destroying pretrained features |
| `SEED` | 42 | Fixed for reproducibility of weight initialisation and data shuffling |
| Optimiser | Adam | Adaptive learning rates per parameter; well-suited to fine-tuning |
| Loss | CrossEntropyLoss | Standard for multi-class classification |

## Training and Evaluation Procedure

1. `set_seed(SEED)` is called before building each model to ensure reproducibility.
2. The pretrained ResNet18 is moved to the available device (CUDA or CPU).
3. `fit()` runs `LOCAL_EPOCHS` epochs. Each epoch calls `train_one_epoch` (forward + backward + Adam step) then `evaluate` on the validation loader.
4. After training, `evaluate` is called once more to compute the full confusion matrix and per-class recall.
5. The majority-class baseline is computed from the validation set label distribution.
6. The model state dict is saved to `checkpoints/local_hospital<n>.pt`.

## Metrics

- **Accuracy**: fraction of correctly classified validation samples.
- **Per-class recall**: TP / (TP + FN) for each class. Reported as `N/A` for classes absent from a hospital's validation set.
- **Confusion matrix**: NUM_CLASSES × NUM_CLASSES table; rows = true class, columns = predicted class.
- **Majority-class baseline**: accuracy of always predicting the most frequent class in the validation set. The trained model must beat this to demonstrate it has learned something.

---

## Results

### Hospital 1 — Guangzhou, China (NORMAL, PNEUMONIA)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.4501 | 0.8515 | 0.2625 | 0.9477 |
| 2 | 0.1385 | 0.9494 | 0.1962 | 0.9346 |
| 3 | 0.0782 | 0.9755 | 0.1680 | 0.9150 |
| 4 | 0.0567 | 0.9837 | 0.1816 | 0.9477 |
| 5 | 0.0694 | 0.9788 | 0.2195 | 0.9477 |

Final val accuracy: **0.9477** | Majority baseline: 0.5033 | **Beats baseline: YES**

Per-class recall: NORMAL=0.9605, PNEUMONIA=0.9351, COVID19=N/A, TUBERCULOSIS=N/A

Confusion matrix:

|  | COVID19 | NORMAL | PNEUMONIA | TUBERCULOSIS |
|---|---|---|---|---|
| COVID19 | 0 | 0 | 0 | 0 |
| NORMAL | 0 | 73 | 3 | 0 |
| PNEUMONIA | 0 | 5 | 72 | 0 |
| TUBERCULOSIS | 0 | 0 | 0 | 0 |

### Hospital 2 — Qatar Univ + Dhaka (NORMAL, TUBERCULOSIS)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.2292 | 0.9363 | 0.1172 | 0.9750 |
| 2 | 0.0259 | 0.9925 | 0.0652 | 0.9750 |
| 3 | 0.0222 | 0.9938 | 0.0836 | 0.9750 |
| 4 | 0.0083 | 0.9988 | 0.1038 | 0.9750 |
| 5 | 0.0117 | 0.9988 | 0.0866 | 0.9800 |

Final val accuracy: **0.9800** | Majority baseline: 0.5000 | **Beats baseline: YES**

Per-class recall: NORMAL=1.0000, TUBERCULOSIS=0.9600, COVID19=N/A, PNEUMONIA=N/A

Confusion matrix:

|  | COVID19 | NORMAL | PNEUMONIA | TUBERCULOSIS |
|---|---|---|---|---|
| COVID19 | 0 | 0 | 0 | 0 |
| NORMAL | 0 | 100 | 0 | 0 |
| PNEUMONIA | 0 | 0 | 0 | 0 |
| TUBERCULOSIS | 0 | 4 | 0 | 96 |

### Hospital 3 — Qatar Univ et al. (COVID19, NORMAL, PNEUMONIA)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.6361 | 0.7192 | 0.4414 | 0.8167 |
| 2 | 0.3897 | 0.8275 | 0.3234 | 0.8833 |
| 3 | 0.2753 | 0.8900 | 0.3693 | 0.8500 |
| 4 | 0.2300 | 0.9067 | 0.3205 | 0.8600 |
| 5 | 0.2074 | 0.9167 | 0.3297 | 0.8667 |

Final val accuracy: **0.8667** | Majority baseline: 0.3333 | **Beats baseline: YES**

Per-class recall: COVID19=0.9100, NORMAL=0.7400, PNEUMONIA=0.9500, TUBERCULOSIS=N/A

Confusion matrix:

|  | COVID19 | NORMAL | PNEUMONIA | TUBERCULOSIS |
|---|---|---|---|---|
| COVID19 | 91 | 6 | 3 | 0 |
| NORMAL | 18 | 74 | 8 | 0 |
| PNEUMONIA | 1 | 4 | 95 | 0 |
| TUBERCULOSIS | 0 | 0 | 0 | 0 |

### Cross-Hospital Evaluation (rows = model trained at, columns = evaluated on)

| | H1 val | H2 val | H3 val |
|---|---|---|---|
| Model H1 | 0.9477 | 0.3150 | 0.4167 |
| Model H2 | 0.4771 | 0.9800 | 0.1633 |
| Model H3 | 0.9085 | 0.4850 | 0.8667 |

**Interpretation:** The diagonal (each model on its own hospital) is strong (0.87–0.98). Off-diagonal accuracy drops sharply — a model trained at one hospital generalises poorly to another. This is the core motivation for federated learning: Hospital 1's model scores only 0.32 on Hospital 2's data, and Hospital 2's model scores only 0.16 on Hospital 3's data. Federated averaging will allow all hospitals to benefit from each other's data distribution without sharing images.
