"""Summarize baseline federated runs and write comparison CSVs."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from statistics import mean, pstdev

from medifed.config import RESULTS_DIR, RUNS_DIR, HOSPITALS

RUN_NAME_RE = re.compile(r"^(?P<mode>[a-zA-Z]+)_seed(?P<seed>\d+)$")


def parse_run_name(run_name: str) -> tuple[str, int]:
    """Return (mode, seed) for a run directory name like baseline_seed42."""
    match = RUN_NAME_RE.match(run_name)
    if not match:
        raise ValueError(f"Unsupported run folder name: {run_name!r}")
    return match.group("mode"), int(match.group("seed"))


def _read_round_rows(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["round"] = int(row["round"])
            row["hospital"] = int(row["hospital"])
            row["loss"] = float(row["loss"])
            row["accuracy"] = float(row["accuracy"])
            row["num_samples"] = int(row["num_samples"])
            if row["split"] == "val":
                rows.append(row)
    return rows


def build_summary(runs_dir: str | Path = RUNS_DIR) -> dict:
    """Build summary data for all run folders in a results/runs directory."""
    runs_dir = Path(runs_dir)
    by_mode: dict[str, dict[str, dict[int, dict[str, list[float]]]]] = {}

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        try:
            mode, _seed = parse_run_name(run_dir.name)
        except ValueError:
            continue
        by_mode.setdefault(mode, {})
        for csv_path in sorted(run_dir.glob("hospital*_rounds.csv")):
            hospital_id = int(csv_path.stem.replace("hospital", "").replace("_rounds", ""))
            rows = _read_round_rows(csv_path)
            by_mode[mode].setdefault(hospital_id, {})
            for row in rows:
                round_id = row["round"]
                entry = by_mode[mode][hospital_id].setdefault(round_id, {"accuracy": [], "loss": []})
                entry["accuracy"].append(row["accuracy"])
                entry["loss"].append(row["loss"])

    summary_rounds: list[dict] = []
    for mode, hospitals in sorted(by_mode.items()):
        for hospital_id, round_map in sorted(hospitals.items()):
            for round_id, values in sorted(round_map.items()):
                accuracy_values = values["accuracy"]
                loss_values = values["loss"]
                summary_rounds.append(
                    {
                        "mode": mode,
                        "hospital": hospital_id,
                        "round": round_id,
                        "mean_val_acc": mean(accuracy_values),
                        "std_val_acc": pstdev(accuracy_values) if len(accuracy_values) > 1 else 0.0,
                        "mean_val_loss": mean(loss_values),
                        "std_val_loss": pstdev(loss_values) if len(loss_values) > 1 else 0.0,
                        "n_seeds": len(accuracy_values),
                    }
                )

    summary_final: list[dict] = []
    per_seed_means: dict[str, list[float]] = {}
    for mode, hospitals in sorted(by_mode.items()):
        for hospital_id in sorted(hospitals):
            seed_values: list[float] = []
            for run_dir in sorted(runs_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                try:
                    run_mode, seed = parse_run_name(run_dir.name)
                except ValueError:
                    continue
                if run_mode != mode:
                    continue
                csv_path = run_dir / f"hospital{hospital_id}_rounds.csv"
                if not csv_path.exists():
                    continue
                rows = _read_round_rows(csv_path)
                if not rows:
                    continue
                last_val = max(rows, key=lambda item: item["round"])
                seed_values.append(last_val["accuracy"])
            if seed_values:
                summary_final.append(
                    {
                        "mode": mode,
                        "hospital": hospital_id,
                        "mean_val_acc": mean(seed_values),
                        "std_val_acc": pstdev(seed_values) if len(seed_values) > 1 else 0.0,
                        "min_val_acc": min(seed_values),
                        "max_val_acc": max(seed_values),
                        "n_seeds": len(seed_values),
                    }
                )
                per_seed_means.setdefault(mode, [])
                per_seed_means[mode].append(mean(seed_values))

    all_hospital_values: list[dict] = []
    for mode in sorted(by_mode):
        if not by_mode[mode]:
            continue
        all_values_by_seed: list[float] = []
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            try:
                run_mode, _seed = parse_run_name(run_dir.name)
            except ValueError:
                continue
            if run_mode != mode:
                continue
            seed_accs: list[float] = []
            for hospital_id in sorted(by_mode[mode]):
                csv_path = run_dir / f"hospital{hospital_id}_rounds.csv"
                if not csv_path.exists():
                    continue
                rows = _read_round_rows(csv_path)
                if not rows:
                    continue
                last_val = max(rows, key=lambda item: item["round"])
                seed_accs.append(last_val["accuracy"])
            if seed_accs:
                all_values_by_seed.append(mean(seed_accs))
        if all_values_by_seed:
            summary_final.append(
                {
                    "mode": mode,
                    "hospital": "ALL",
                    "mean_val_acc": mean(all_values_by_seed),
                    "std_val_acc": pstdev(all_values_by_seed) if len(all_values_by_seed) > 1 else 0.0,
                    "min_val_acc": min(all_values_by_seed),
                    "max_val_acc": max(all_values_by_seed),
                    "n_seeds": len(all_values_by_seed),
                }
            )
            all_hospital_values.append({"mode": mode, "values": all_values_by_seed})

    comparison_styles: list[dict] = []
    comparison_seed_rows: list[dict] = []
    all_run_names = sorted(item.name for item in runs_dir.iterdir() if item.is_dir())
    local_file = RESULTS_DIR / "local_final_metrics.csv"
    local_metrics = {}
    if local_file.exists():
        with open(local_file, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                local_metrics[int(row["hospital"])] = float(row["val_accuracy"])

    for hospital_id in sorted(HOSPITALS):
        local_accuracy = local_metrics.get(hospital_id, "")
        fedavg_values: list[float] = []
        fedbn_values: list[float] = []
        fedavg_by_seed = {}
        for run_dir_name in all_run_names:
            try:
                mode, seed = parse_run_name(run_dir_name)
            except ValueError:
                continue
            csv_path = runs_dir / run_dir_name / f"hospital{hospital_id}_rounds.csv"
            if not csv_path.exists():
                continue
            rows = _read_round_rows(csv_path)
            if not rows:
                continue
            last_val = max(rows, key=lambda item: item["round"])
            if mode == "baseline":
                fedavg_values.append(last_val["accuracy"])
                fedavg_by_seed[seed] = last_val["accuracy"]
                comparison_seed_rows.append(
                    {
                        "seed": seed,
                        "hospital": hospital_id,
                        "fedavg_acc": last_val["accuracy"],
                        "fedbn_acc": "",
                        "diff": "",
                    }
                )
            elif mode == "personalized":
                fedbn_values.append(last_val["accuracy"])
                for entry in comparison_seed_rows:
                    if entry["seed"] == seed and entry["hospital"] == hospital_id:
                        entry["fedbn_acc"] = last_val["accuracy"]
                        entry["diff"] = float(last_val["accuracy"]) - float(entry["fedavg_acc"]) if entry["fedavg_acc"] != "" else ""
        fedavg_mean = mean(fedavg_values) if fedavg_values else ""
        fedbn_mean = mean(fedbn_values) if fedbn_values else ""
        fedavg_std = pstdev(fedavg_values) if len(fedavg_values) > 1 else 0.0 if fedavg_values else ""
        fedbn_std = pstdev(fedbn_values) if len(fedbn_values) > 1 else 0.0 if fedbn_values else ""
        diff_fedavg_minus_local = (fedavg_mean - local_accuracy) if local_accuracy != "" and fedavg_mean != "" else ""
        diff_fedbn_minus_fedavg = (fedbn_mean - fedavg_mean) if fedbn_mean != "" and fedavg_mean != "" else ""
        diff_fedbn_minus_local = (fedbn_mean - local_accuracy) if local_accuracy != "" and fedbn_mean != "" else ""
        comparison_styles.append(
            {
                "hospital": hospital_id,
                "local_accuracy": local_accuracy,
                "fedavg_mean": fedavg_mean,
                "fedavg_std": fedavg_std,
                "fedbn_mean": fedbn_mean,
                "fedbn_std": fedbn_std,
                "diff_fedavg_minus_local": diff_fedavg_minus_local,
                "diff_fedbn_minus_fedavg": diff_fedbn_minus_fedavg,
                "diff_fedbn_minus_local": diff_fedbn_minus_local,
            }
        )

    return {
        "summary_rounds": summary_rounds,
        "summary_final": summary_final,
        "comparison_methods": comparison_styles,
        "comparison_per_seed": comparison_seed_rows,
    }


def build_comparison_table(local_file: str | Path | None, fedavg_runs: list, fedbn_runs: list, hospitals: list[int]) -> list[dict]:
    """Build the comparison table for local, FedAvg and FedBN results.

    The function is intentionally small and tolerant of missing inputs so the
    summary tests can validate the empty-fedbn behavior without a full run.
    """
    rows: list[dict] = []
    local_metrics = {}
    if local_file is not None and Path(local_file).exists():
        with open(local_file, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                local_metrics[int(row["hospital"])] = float(row["val_accuracy"])

    for hospital_id in hospitals:
        fedavg_mean = ""
        fedavg_std = ""
        fedbn_mean = ""
        fedbn_std = ""

        if fedavg_runs:
            values = []
            for run_dir in fedavg_runs:
                csv_path = Path(run_dir) / f"hospital{hospital_id}_rounds.csv"
                if not csv_path.exists():
                    continue
                rows_for_hospital = _read_round_rows(csv_path)
                if not rows_for_hospital:
                    continue
                values.append(max(rows_for_hospital, key=lambda item: item["round"])["accuracy"])
            if values:
                fedavg_mean = mean(values)
                fedavg_std = pstdev(values) if len(values) > 1 else 0.0

        if fedbn_runs:
            values = []
            for run_dir in fedbn_runs:
                csv_path = Path(run_dir) / f"hospital{hospital_id}_rounds.csv"
                if not csv_path.exists():
                    continue
                rows_for_hospital = _read_round_rows(csv_path)
                if not rows_for_hospital:
                    continue
                values.append(max(rows_for_hospital, key=lambda item: item["round"])["accuracy"])
            if values:
                fedbn_mean = mean(values)
                fedbn_std = pstdev(values) if len(values) > 1 else 0.0

        local_accuracy = local_metrics.get(hospital_id, "")
        rows.append(
            {
                "hospital": hospital_id,
                "local_accuracy": local_accuracy,
                "fedavg_mean": fedavg_mean,
                "fedavg_std": fedavg_std,
                "fedbn_mean": fedbn_mean,
                "fedbn_std": fedbn_std,
                "diff_fedavg_minus_local": (fedavg_mean - local_accuracy) if local_accuracy != "" and fedavg_mean != "" else "",
                "diff_fedbn_minus_fedavg": (fedbn_mean - fedavg_mean) if fedbn_mean != "" and fedavg_mean != "" else "",
                "diff_fedbn_minus_local": (fedbn_mean - local_accuracy) if local_accuracy != "" and fedbn_mean != "" else "",
            }
        )
    return rows


def write_summary_files(results_dir: str | Path = RESULTS_DIR, runs_dir: str | Path = RUNS_DIR) -> dict:
    """Write the summary CSV files to the project results directory."""
    results_dir = Path(results_dir)
    runs_dir = Path(runs_dir)
    summary = build_summary(runs_dir)

    rounds_path = results_dir / "summary_rounds.csv"
    with open(rounds_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "hospital", "round", "mean_val_acc", "std_val_acc", "mean_val_loss", "std_val_loss", "n_seeds"])
        writer.writeheader()
        writer.writerows(summary["summary_rounds"])

    final_path = results_dir / "summary_final.csv"
    with open(final_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "hospital", "mean_val_acc", "std_val_acc", "min_val_acc", "max_val_acc", "n_seeds"])
        writer.writeheader()
        writer.writerows(summary["summary_final"])

    comparison_path = results_dir / "comparison_methods.csv"
    with open(comparison_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["hospital", "local_accuracy", "fedavg_mean", "fedavg_std", "fedbn_mean", "fedbn_std", "diff_fedavg_minus_local", "diff_fedbn_minus_fedavg", "diff_fedbn_minus_local"],
        )
        writer.writeheader()
        for row in summary["comparison_methods"]:
            row_copy = dict(row)
            if row_copy["fedbn_mean"] == "":
                row_copy["fedbn_mean"] = ""
                row_copy["fedbn_std"] = ""
                row_copy["diff_fedbn_minus_fedavg"] = ""
                row_copy["diff_fedbn_minus_local"] = ""
            writer.writerow(row_copy)

    per_seed_path = results_dir / "comparison_per_seed.csv"
    with open(per_seed_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "hospital", "fedavg_acc", "fedbn_acc", "diff"])
        writer.writeheader()
        writer.writerows(summary["comparison_per_seed"])

    return summary


def main() -> None:
    write_summary_files()
    print("Wrote summary CSVs to", RESULTS_DIR)


if __name__ == "__main__":
    main()
