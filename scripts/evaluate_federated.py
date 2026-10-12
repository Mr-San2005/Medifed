"""Evaluate a saved FedAvg global checkpoint on each hospital validation set."""

import argparse
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
    DEFAULT_RUN_NAME,
    HOSPITALS,
    NUM_CLASSES,
    RESULTS_DIR,
)
from medifed.data import make_loader
from medifed.model import build_model, get_device
from medifed.train import evaluate


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


def _recall_cell(value) -> str:
    return f"{value:.4f}" if value is not None else "N/A (no samples)"


def _recall_csv(value):
    return value if value is not None else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a MediFed federated global checkpoint.")
    parser.add_argument(
        "run_name",
        nargs="?",
        default=DEFAULT_RUN_NAME,
        help=f"Run name (default: {DEFAULT_RUN_NAME})",
    )
    args = parser.parse_args()
    run_name = args.run_name

    ckpt_path = CHECKPOINT_DIR / f"{run_name}_global.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Run the federated baseline first, then retry this script.")
        sys.exit(1)

    device = get_device()
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}\n")

    model = build_model(pretrained=False)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)

    local_path = RESULTS_DIR / "local_final_metrics.csv"
    local_acc = {}
    with open(local_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            local_acc[int(row["hospital"])] = float(row["val_accuracy"])

    final_rows = []
    comparison_rows = []

    for hid in (1, 2, 3):
        print(f"{'=' * 60}")
        print(f"Hospital {hid}: {HOSPITALS[hid]['name']}")
        print(f"{'=' * 60}")

        val_loader = make_loader(hid, "val", BATCH_SIZE, shuffle=False)
        metrics = evaluate(model, val_loader, device)
        baseline = majority_baseline(val_loader)

        print(f"  Val accuracy       : {metrics['accuracy']:.4f}")
        print(f"  Majority baseline  : {baseline:.4f}")
        print("  Per-class recall:")
        for i, cls in enumerate(CLASS_NAMES):
            print(f"    {cls:<16} {_recall_cell(metrics['per_class_recall'][i])}")
        print("  Confusion matrix (rows=true, cols=predicted):")
        print_confusion(metrics["confusion_matrix"], CLASS_NAMES)
        print()

        recall = metrics["per_class_recall"]
        final_rows.append({
            "hospital": hid,
            "val_accuracy": metrics["accuracy"],
            "majority_baseline": baseline,
            "recall_COVID19": _recall_csv(recall[0]),
            "recall_NORMAL": _recall_csv(recall[1]),
            "recall_PNEUMONIA": _recall_csv(recall[2]),
            "recall_TUBERCULOSIS": _recall_csv(recall[3]),
        })
        fed_acc = metrics["accuracy"]
        loc_acc = local_acc[hid]
        comparison_rows.append({
            "hospital": hid,
            "local_accuracy": loc_acc,
            "fedavg_accuracy": fed_acc,
            "difference": fed_acc - loc_acc,
        })

    print(f"{'=' * 60}")
    print("Local-only vs FedAvg global (validation accuracy)")
    print(f"{'=' * 60}")
    print(f"{'hospital':>10}  {'local_accuracy':>16}  {'fedavg_accuracy':>16}  {'difference':>12}")
    for row in comparison_rows:
        print(
            f"{row['hospital']:>10}  {row['local_accuracy']:>16.4f}  "
            f"{row['fedavg_accuracy']:>16.4f}  {row['difference']:>12.4f}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    final_path = RESULTS_DIR / f"{run_name}_final_metrics.csv"
    with open(final_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "hospital",
                "val_accuracy",
                "majority_baseline",
                "recall_COVID19",
                "recall_NORMAL",
                "recall_PNEUMONIA",
                "recall_TUBERCULOSIS",
            ],
        )
        writer.writeheader()
        writer.writerows(final_rows)

    comparison_path = RESULTS_DIR / "comparison_local_vs_fedavg.csv"
    with open(comparison_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["hospital", "local_accuracy", "fedavg_accuracy", "difference"],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"\nSaved: {final_path}")
    print(f"Saved: {comparison_path}")


if __name__ == "__main__":
    main()
