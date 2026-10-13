"""Generate the Phase 3 baseline comparison charts."""

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medifed.config import RESULTS_DIR
from medifed.plots import grouped_bar_chart, line_chart
from scripts.summarize_runs import write_summary_files


def _read_csv_rows(path: Path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_charts() -> None:
    write_summary_files(RESULTS_DIR, RESULTS_DIR / "runs")

    comparison_rows = _read_csv_rows(RESULTS_DIR / "comparison_methods.csv")
    local_values = []
    fedavg_values = []
    for row in comparison_rows:
        if row.get("local_accuracy"):
            local_values.append(float(row["local_accuracy"]))
        if row.get("fedavg_mean"):
            fedavg_values.append(float(row["fedavg_mean"]))

    comparison_path = RESULTS_DIR / "chart_methods_comparison.png"
    grouped_bar_chart(
        comparison_path,
        labels=["Hospital 1", "Hospital 2", "Hospital 3"],
        series=[
            {"label": "Local", "values": local_values, "color": (15, 107, 110)},
            {"label": "FedAvg", "values": fedavg_values, "color": (63, 94, 154)},
        ],
        title="Local vs FedAvg validation accuracy",
        xlabel="Hospital",
        ylabel="Accuracy",
    )

    rounds_rows = _read_csv_rows(RESULTS_DIR / "summary_rounds.csv")
    by_round = {}
    for row in rounds_rows:
        if row["mode"] == "baseline":
            by_round.setdefault(int(row["round"]), [])
            by_round[int(row["round"])].append(float(row["mean_val_acc"]))
    x_values = sorted(by_round)
    mean_values = [sum(values) / len(values) for values in (by_round[x] for x in x_values)]

    rounds_path = RESULTS_DIR / "chart_rounds.png"
    line_chart(
        rounds_path,
        x_values=x_values,
        series=[
            {"label": "FedAvg", "values": mean_values, "color": (63, 94, 154)},
        ],
        title="Mean validation accuracy by round",
        xlabel="Round",
        ylabel="Accuracy",
    )

    print(f"Saved: {comparison_path}")
    print(f"Saved: {rounds_path}")


if __name__ == "__main__":
    build_charts()
