"""Compare hospital-3 local performance against the hospital-3 FedBN model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medifed.config import CHECKPOINT_DIR, HOSPITALS
from medifed.model import build_model, get_device
from medifed.personalization import bn_keys, load_bn_state, merge_state
from medifed.train import evaluate


def evaluate_model(model, hospital_id: int, device):
    """Evaluate the given model on hospital_id's validation set."""
    from medifed.data import make_loader

    val_loader = make_loader(hospital_id, "val", batch_size=16, shuffle=False)
    return evaluate(model, val_loader, device)


def main() -> None:
    hospital_id = 3
    device = get_device()

    local_path = CHECKPOINT_DIR / f"local_hospital{hospital_id}.pt"
    global_path = CHECKPOINT_DIR / "personalized_seed42_global.pt"
    bn_path = CHECKPOINT_DIR / "personalized_seed42" / f"hospital{hospital_id}_bn.pt"

    local_model = build_model(pretrained=False).to(device)
    local_state = torch.load(local_path, map_location=device, weights_only=True)
    local_model.load_state_dict(local_state)

    fedbn_model = build_model(pretrained=False).to(device)
    global_state = torch.load(global_path, map_location=device, weights_only=True)
    local_bn = load_bn_state(bn_path)
    fedbn_model.load_state_dict(merge_state(global_state, local_bn, bn_keys(fedbn_model)))

    local_metrics = evaluate_model(local_model, hospital_id, device)
    fedbn_metrics = evaluate_model(fedbn_model, hospital_id, device)

    print(f"Hospital {hospital_id}: {HOSPITALS[hospital_id]['name']}")
    print("Local model metrics")
    print(local_metrics["confusion_matrix"])
    print("Per-class recall:", local_metrics["per_class_recall"])
    print()
    print("FedBN model metrics")
    print(fedbn_metrics["confusion_matrix"])
    print("Per-class recall:", fedbn_metrics["per_class_recall"])
    print()

    same_confusion = local_metrics["confusion_matrix"] == fedbn_metrics["confusion_matrix"]
    print(f"Confusion matrices identical: {same_confusion}")
    print(f"Local accuracy: {local_metrics['accuracy']:.4f}")
    print(f"FedBN accuracy: {fedbn_metrics['accuracy']:.4f}")
    print("FedBN model source: personalized_seed42_global.pt + hospital3_bn.pt")


if __name__ == "__main__":
    main()
