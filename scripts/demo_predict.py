"""Generate deterministic demo predictions for each hospital and save them to CSV."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medifed.config import CHECKPOINT_DIR, CLASS_NAMES, HOSPITALS, HOSPITAL_RAW_DIRS, PROCESSED_DIR
from medifed.data import read_manifest, get_transforms
from medifed.model import build_model, get_device
from medifed.personalization import bn_keys, load_bn_state, merge_state


VALID_METHODS = {"local", "fedavg", "fedbn"}


def select_demo_images(hospital_id: int) -> dict[str, dict]:
    """Return the first validation row for each class present in a hospital."""
    rows = read_manifest(PROCESSED_DIR / f"hospital{hospital_id}_manifest.csv")
    selected: dict[str, dict] = {}
    for row in rows:
        if row["split"] != "val":
            continue
        cls = row["class_name"]
        if cls not in selected:
            selected[cls] = row
    return {cls: selected[cls] for cls in sorted(selected, key=lambda item: CLASS_NAMES.index(item))}


def load_model(method: str, hospital_id: int) -> torch.nn.Module:
    """Load the model for the selected method (local, fedavg, fedbn)."""
    if method not in VALID_METHODS:
        raise ValueError(f"Unknown method: {method!r}. Use one of: {sorted(VALID_METHODS)}")

    model = build_model(pretrained=False)
    if method == "local":
        state = torch.load(CHECKPOINT_DIR / f"local_hospital{hospital_id}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model

    if method == "fedavg":
        state = torch.load(CHECKPOINT_DIR / "baseline_seed42_global.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model

    global_state = torch.load(CHECKPOINT_DIR / "personalized_seed42_global.pt", map_location="cpu", weights_only=True)
    local_bn = load_bn_state(CHECKPOINT_DIR / "personalized_seed42" / f"hospital{hospital_id}_bn.pt")
    model.load_state_dict(merge_state(global_state, local_bn, bn_keys(model)))
    return model


def predict_class(model: torch.nn.Module, image_path: Path, device: torch.device) -> tuple[str, float]:
    """Return the predicted class name and top probability for a single image."""
    transform = get_transforms(train=False)
    image = __import__('PIL').Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
    probs = F.softmax(logits, dim=1).squeeze(0)
    top_index = int(torch.argmax(probs).item())
    return CLASS_NAMES[top_index], float(probs[top_index].item())


def main() -> None:
    device = get_device()
    rows: list[dict] = []
    for hospital_id in (1, 2, 3):
        selected = select_demo_images(hospital_id)
        for true_class, row in selected.items():
            image_path = HOSPITAL_RAW_DIRS[hospital_id] / row["relative_path"]
            for method in ("local", "fedavg", "fedbn"):
                model = load_model(method, hospital_id).to(device)
                predicted_class, top_prob = predict_class(model, image_path, device)
                correct = predicted_class == true_class
                rows.append({
                    "hospital": hospital_id,
                    "true_class": true_class,
                    "method": method,
                    "predicted_class": predicted_class,
                    "top_probability": top_prob,
                    "correct": int(correct),
                })

    output_path = PROJECT_ROOT / "results" / "demo_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["hospital", "true_class", "method", "predicted_class", "top_probability", "correct"])
        writer.writeheader()
        writer.writerows(rows)

    print("hospital\ttrue_class\tmethod\tpredicted\tconfidence\tresult")
    for row in rows:
        print(f"{row['hospital']}\t{row['true_class']}\t{row['method']}\t{row['predicted_class']}\t{row['top_probability']:.4f}\t{'OK' if row['correct'] else 'WRONG'}")

    counts = {method: 0 for method in VALID_METHODS}
    for row in rows:
        if row["correct"]:
            counts[row["method"]] += 1
    print("\nCorrect predictions per method:")
    for method in ("local", "fedavg", "fedbn"):
        print(f"{method}: {counts[method]}")


if __name__ == "__main__":
    main()
