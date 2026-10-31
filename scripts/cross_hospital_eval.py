"""Evaluate each FedBN hospital checkpoint on all three hospital validation sets.

Writes results/crosshospital_fedbn.csv with columns:
    model_hospital, eval_hospital, accuracy
"""

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from medifed.config import BATCH_SIZE, CHECKPOINT_DIR, HOSPITALS, RESULTS_DIR
from medifed.data import make_loader
from medifed.model import build_model, get_device
from medifed.personalization import bn_keys, load_bn_state, merge_state
from medifed.train import evaluate


def load_fedbn_model(model_hospital: int, device: torch.device) -> torch.nn.Module:
    global_path = CHECKPOINT_DIR / "personalized_seed42_global.pt"
    bn_path = CHECKPOINT_DIR / "personalized_seed42" / f"hospital{model_hospital}_bn.pt"
    if not global_path.exists():
        raise FileNotFoundError(f"Missing: {global_path}")
    if not bn_path.exists():
        raise FileNotFoundError(f"Missing: {bn_path}")
    model = build_model(pretrained=False)
    state = torch.load(global_path, map_location="cpu", weights_only=True)
    local_bn = load_bn_state(bn_path)
    state = merge_state(state, local_bn, bn_keys(model))
    model.load_state_dict(state)
    return model.to(device).eval()


def main() -> None:
    device = get_device()
    print(f"Device: {device}\n")
    rows = []
    for model_hospital in sorted(HOSPITALS):
        print(f"Loading FedBN model for Hospital {model_hospital}…")
        try:
            model = load_fedbn_model(model_hospital, device)
        except FileNotFoundError as exc:
            print(f"  SKIP: {exc}")
            continue
        for eval_hospital in sorted(HOSPITALS):
            loader = make_loader(eval_hospital, "val", BATCH_SIZE, shuffle=False)
            metrics = evaluate(model, loader, device)
            acc = metrics["accuracy"]
            rows.append({
                "model_hospital": model_hospital,
                "eval_hospital": eval_hospital,
                "accuracy": acc,
            })
            marker = " <--" if model_hospital == eval_hospital else ""
            print(f"  Eval Hospital {eval_hospital}: {acc:.4f}{marker}")
    print()

    out_path = RESULTS_DIR / "crosshospital_fedbn.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_hospital", "eval_hospital", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_path}")

    print("\nCross-hospital FedBN accuracy matrix:")
    print(f"{'':>20}  {'H1 eval':>10}  {'H2 eval':>10}  {'H3 eval':>10}")
    for mh in sorted(HOSPITALS):
        vals = [next((r["accuracy"] for r in rows if r["model_hospital"] == mh and r["eval_hospital"] == eh), None) for eh in (1, 2, 3)]
        cells = [f"{v:.4f}" if v is not None else "  N/A  " for v in vals]
        print(f"  H{mh} model trained:  {'  '.join(cells)}")


if __name__ == "__main__":
    main()
