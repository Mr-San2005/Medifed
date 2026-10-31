# Grad-CAM and Image Inference

## Purpose

MediFed provides a local demonstration path from an uploaded chest X-ray to a four-class model output and Grad-CAM overlay. The interface is a research demonstration, not a clinical diagnostic product.

## Inference variants

`medifed/inference.py` exposes checkpoint-backed variants:

| Variant | Checkpoint state |
|---|---|
| `fedavg_global` | `baseline_seed42_global.pt` |
| `fedbn_hospital1` to `fedbn_hospital3` | `personalized_seed42_global.pt` merged with the selected hospital's BatchNorm checkpoint |
| `local_hospital1` to `local_hospital3` | Hospital-specific local checkpoint |

Only variants whose files exist are offered by the application. The model uses the same resize, tensor conversion, and ImageNet normalization as validation.

## Grad-CAM

`medifed/gradcam.py` registers a forward hook on the last ResNet18 block in `layer4`. It backpropagates the selected class score, globally averages the activation gradients to obtain channel weights, forms a ReLU-weighted activation map, upsamples it to the model input size, and scales it to `[0, 1]`. The hook is removed in a `finally` block. A constant map becomes zeros rather than producing NaN values.

The overlay uses a blue-to-cyan-to-yellow-to-red activation scale. Warm colours indicate regions that contributed more strongly to the selected model score; they do not establish clinical localization, pathology, or causal reasoning.

## Observed demonstration

Command run on 2026-10-13:

```powershell
python scripts/demo_gradcam.py --variant fedbn_hospital3
```

The real output was:

| True class | Source hospital | Predicted class | Top probability |
|---|---:|---|---:|
| COVID19 | 3 | COVID19 | 97.8% |
| NORMAL | 3 | NORMAL | 95.6% |
| PNEUMONIA | 1 | COVID19 | 79.7% |
| TUBERCULOSIS | 2 | COVID19 | 90.1% |

The PNG is `results/gradcam_demo_fedbn_hospital3.png`. The Hospital 3 model was trained on COVID19, NORMAL, and PNEUMONIA data; it has no tuberculosis training examples. The cross-hospital examples are intentionally labelled with their source hospital and show why a model's global four-class output does not imply equivalent class support at every hospital.

## Limitations

- The saved comparison represents seed 42 only. The displayed standard deviation is zero because only one seed is represented; it is not an estimate of run-to-run variation.
- The validation split is one deterministic split and the data are capped per class.
- The Grad-CAM map explains score sensitivity within this model; it is not a clinical segmentation or validated explanation.
- Confidence scores have not been calibrated or clinically validated.
- The full multi-disease pooled and FedBN cross-hospital evaluation remains incomplete.