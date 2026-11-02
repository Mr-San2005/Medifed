"""Read MediFed result files and connect the diagnosis page to model inference."""

from __future__ import annotations

import csv
from pathlib import Path

from medifed.config import CHECKPOINT_DIR, HOSPITALS, RESULTS_DIR
from medifed.inference import list_variants, predict


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _number(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def load_results(results_dir: str | Path = RESULTS_DIR) -> dict:
    """Aggregate saved result CSVs into a JSON-ready dashboard response."""
    root = Path(results_dir)
    counts_rows = _read_csv(root / "data_counts.csv")
    comparison_rows = _read_csv(root / "comparison_methods.csv")
    rounds_rows = _read_csv(root / "summary_rounds.csv")
    local_cross_rows = _read_csv(root / "local_cross_eval.csv")
    fedbn_cross_rows = _read_csv(root / "crosshospital_fedbn.csv")

    counts_by_hospital: dict[int, list[dict]] = {hospital_id: [] for hospital_id in HOSPITALS}
    for row in counts_rows:
        hospital_id = int(row["hospital"])
        counts_by_hospital.setdefault(hospital_id, []).append(
            {
                "class": row["class"],
                "found": int(row["found"]),
                "kept": int(row["kept"]),
                "train": int(row["train"]),
                "val": int(row["val"]),
            }
        )

    comparison_by_hospital = {int(row["hospital"]): row for row in comparison_rows}
    hospitals = []
    # Generate colors for any number of hospitals from a fixed palette.
    _palette = ["#0F6B6E", "#3F5E9A", "#A3532B", "#6B3FA0", "#2E7D32", "#C62828"]
    hospital_colors = {hid: _palette[(hid - 1) % len(_palette)] for hid in HOSPITALS}
    for hospital_id, details in HOSPITALS.items():
        metrics = comparison_by_hospital.get(hospital_id, {})
        hospitals.append(
            {
                "id": hospital_id,
                "name": details["name"],
                "color": hospital_colors[hospital_id],
                "classes": details["classes"],
                "counts": counts_by_hospital.get(hospital_id, []),
                "local_accuracy": _number(metrics.get("local_accuracy")),
                "fedavg_accuracy": _number(metrics.get("fedavg_mean")),
                "fedavg_std": _number(metrics.get("fedavg_std")),
                "fedbn_accuracy": _number(metrics.get("fedbn_mean")),
                "fedbn_std": _number(metrics.get("fedbn_std")),
            }
        )

    rounds = [
        {
            "mode": row["mode"],
            "hospital": int(row["hospital"]),
            "round": int(row["round"]),
            "accuracy": float(row["mean_val_acc"]),
            "accuracy_std": float(row["std_val_acc"]),
            "loss": float(row["mean_val_loss"]),
        }
        for row in rounds_rows
    ]
    local_cross = [
        {
            "model_hospital": int(row["model_hospital"]),
            "eval_hospital": int(row["eval_hospital"]),
            "accuracy": float(row["accuracy"]),
        }
        for row in local_cross_rows
    ]
    fedbn_cross = [
        {
            "model_hospital": int(row["model_hospital"]),
            "eval_hospital": int(row["eval_hospital"]),
            "accuracy": float(row["accuracy"]),
        }
        for row in fedbn_cross_rows
    ]

    recall_methods = []
    for method, filename in (
        ("Local", "local_final_metrics.csv"),
        ("FedAvg", "baseline_seed42_final_metrics.csv"),
        ("FedBN", "personalized_seed42_final_metrics.csv"),
    ):
        for row in _read_csv(root / filename):
            hospital_id = int(row["hospital"])
            for class_name in HOSPITALS[hospital_id]["classes"]:
                recall_methods.append(
                    {
                        "method": method,
                        "hospital": hospital_id,
                        "class": class_name,
                        "recall": _number(row.get(f"recall_{class_name}")),
                    }
                )

    return {
        "hospitals": hospitals,
        "rounds": rounds,
        "cross_hospital": {"local": local_cross, "fedbn": fedbn_cross},
        "per_class_recall": recall_methods,
        "empty_states": {
            "data_counts": not bool(counts_rows),
            "method_comparison": not bool(comparison_rows),
            "round_curves": not bool(rounds_rows),
            "cross_hospital_local": not bool(local_cross_rows),
            "cross_hospital_fedbn": not bool(fedbn_cross_rows),
            "per_class_recall": not bool(recall_methods),
        },
    }


def available_variants(checkpoint_dir: str | Path = CHECKPOINT_DIR) -> list[str]:
    """List model options backed by checkpoints currently on disk."""
    return list_variants(checkpoint_dir)


def run_prediction(variant: str, image_bytes: bytes) -> dict:
    """Run an in-memory prediction; the image is never written to disk."""
    return predict(variant, image_bytes)