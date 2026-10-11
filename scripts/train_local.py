"""Train a local ResNet18 model at each hospital and evaluate cross-hospital generalisation."""

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from medifed.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CLASS_NAMES,
    HOSPITALS,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    NUM_CLASSES,
    RESULTS_DIR,
    SEED,
)
from medifed.data import make_loader
from medifed.model import build_model, get_device, set_seed
from medifed.train import evaluate, fit


def majority_baseline(loader) -> float:
    """Return the accuracy of always predicting the most frequent class in loader."""
    counts = [0] * NUM_CLASSES
    for _, labels in loader:
        for lbl in labels.tolist():
            counts[lbl] += 1
    total = sum(counts)
    return max(counts) / total if total > 0 else 0.0


def print_confusion(matrix: list, class_names: list) -> None:
    """Print a confusion matrix with class labels."""
    col_w = 12
    header = " " * col_w + "".join(f"{c:>{col_w}}" for c in class_names)
    print(header)
    for i, row in enumerate(matrix):
        row_str = f"{class_names[i]:>{col_w}}" + "".join(f"{v:>{col_w}}" for v in row)
        print(row_str)


def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}\n")

    history_rows: list[dict] = []
    final_rows:   list[dict] = []
    models: dict[int, torch.nn.Module] = {}

    # ── Per-hospital training ──────────────────────────────────────────────
    for hid in (1, 2, 3):
        print(f"{'='*60}")
        print(f"Hospital {hid}: {HOSPITALS[hid]['name']}")
        print(f"{'='*60}")

        set_seed(SEED)
        model = build_model(pretrained=True).to(device)

        train_loader = make_loader(hid, "train", BATCH_SIZE, shuffle=True)
        val_loader   = make_loader(hid, "val",   BATCH_SIZE, shuffle=False)

        print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

        history = fit(model, train_loader, val_loader, LOCAL_EPOCHS, LEARNING_RATE, device)

        print(f"\n  {'Epoch':>6}  {'Train Loss':>11}  {'Train Acc':>10}  {'Val Loss':>9}  {'Val Acc':>8}")
        print(f"  {'-'*52}")
        for row in history:
            print(f"  {row['epoch']:>6}  {row['train_loss']:>11.4f}  {row['train_acc']:>10.4f}"
                  f"  {row['val_loss']:>9.4f}  {row['val_acc']:>8.4f}")
            history_rows.append({"hospital": hid, **row})

        # Final validation metrics.
        val_metrics = evaluate(model, val_loader, device)
        baseline    = majority_baseline(val_loader)
        beats       = val_metrics["accuracy"] > baseline

        print(f"\n  Final val accuracy : {val_metrics['accuracy']:.4f}")
        print(f"  Majority baseline  : {baseline:.4f}")
        print(f"  Beats baseline     : {'YES' if beats else 'NO'}")

        print(f"\n  Per-class recall:")
        for i, cls in enumerate(CLASS_NAMES):
            r = val_metrics["per_class_recall"][i]
            print(f"    {cls:<16} {f'{r:.4f}' if r is not None else 'N/A (no samples)'}")

        print(f"\n  Confusion matrix (rows=true, cols=predicted):")
        print_confusion(val_metrics["confusion_matrix"], CLASS_NAMES)

        # Save checkpoint.
        ckpt_path = CHECKPOINT_DIR / f"local_hospital{hid}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"\n  Checkpoint saved: {ckpt_path}")

        models[hid] = model

        # Build final metrics row.
        recall = val_metrics["per_class_recall"]
        final_rows.append({
            "hospital":         hid,
            "val_accuracy":     val_metrics["accuracy"],
            "majority_baseline": baseline,
            "recall_COVID19":    recall[0] if recall[0] is not None else "",
            "recall_NORMAL":     recall[1] if recall[1] is not None else "",
            "recall_PNEUMONIA":  recall[2] if recall[2] is not None else "",
            "recall_TUBERCULOSIS": recall[3] if recall[3] is not None else "",
        })

    # ── Cross-hospital evaluation ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Cross-hospital evaluation (rows=model trained at, cols=evaluated on)")
    print(f"{'='*60}")

    cross_rows: list[dict] = []
    col_w = 14

    # Header
    header = " " * col_w + "".join(f"{'H'+str(j)+' val':>{col_w}}" for j in (1, 2, 3))
    print(header)

    for i in (1, 2, 3):
        row_str = f"{'Model H'+str(i):>{col_w}}"
        for j in (1, 2, 3):
            val_loader_j = make_loader(j, "val", BATCH_SIZE, shuffle=False)
            metrics = evaluate(models[i], val_loader_j, device)
            acc = metrics["accuracy"]
            row_str += f"{acc:>{col_w}.4f}"
            cross_rows.append({"model_hospital": i, "eval_hospital": j, "accuracy": acc})
        print(row_str)

    # ── Save CSVs ──────────────────────────────────────────────────────────
    history_path = RESULTS_DIR / "local_training.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["hospital", "epoch", "train_loss",
                                               "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(history_rows)

    final_path = RESULTS_DIR / "local_final_metrics.csv"
    with open(final_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["hospital", "val_accuracy", "majority_baseline",
                                               "recall_COVID19", "recall_NORMAL",
                                               "recall_PNEUMONIA", "recall_TUBERCULOSIS"])
        writer.writeheader()
        writer.writerows(final_rows)

    cross_path = RESULTS_DIR / "local_cross_eval.csv"
    with open(cross_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model_hospital", "eval_hospital", "accuracy"])
        writer.writeheader()
        writer.writerows(cross_rows)

    print(f"\nSaved: {history_path}")
    print(f"Saved: {final_path}")
    print(f"Saved: {cross_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
