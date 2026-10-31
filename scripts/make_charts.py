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
    fedbn_values = []
    for row in comparison_rows:
        if row.get("local_accuracy"):
            local_values.append(float(row["local_accuracy"]))
        if row.get("fedavg_mean"):
            fedavg_values.append(float(row["fedavg_mean"]))
        if row.get("fedbn_mean"):
            fedbn_values.append(float(row["fedbn_mean"]))

    comparison_path = RESULTS_DIR / "chart_methods_comparison.png"
    grouped_bar_chart(
        comparison_path,
        labels=["Hospital 1", "Hospital 2", "Hospital 3"],
        series=[
            {"label": "Local", "values": local_values, "color": (15, 107, 110)},
            {"label": "FedAvg", "values": fedavg_values, "color": (63, 94, 154)},
            {"label": "FedBN", "values": fedbn_values, "color": (192, 97, 40)},
        ],
        title="Local vs FedAvg vs FedBN validation accuracy",
        xlabel="Hospital",
        ylabel="Accuracy",
    )

    rounds_rows = _read_csv_rows(RESULTS_DIR / "summary_rounds.csv")
    by_mode = {"baseline": {}, "personalized": {}}
    for row in rounds_rows:
        mode = row["mode"]
        if mode not in by_mode:
            continue
        by_mode[mode].setdefault(int(row["round"]), [])
        by_mode[mode][int(row["round"])].append(float(row["mean_val_acc"]))

    x_values = sorted({key for values in by_mode.values() for key in values})
    round_series = []
    for mode, color in (("baseline", (63, 94, 154)), ("personalized", (192, 97, 40))):
        values_by_round = []
        for round_id in x_values:
            numbers = by_mode.get(mode, {}).get(round_id, [])
            values_by_round.append(sum(numbers) / len(numbers) if numbers else float("nan"))
        round_series.append({"label": "FedAvg" if mode == "baseline" else "FedBN", "values": values_by_round, "color": color})

    rounds_path = RESULTS_DIR / "chart_rounds.png"
    line_chart(
        rounds_path,
        x_values=x_values,
        series=round_series,
        title="Mean validation accuracy by round",
        xlabel="Round",
        ylabel="Accuracy",
    )

    print(f"Saved: {comparison_path}")
    print(f"Saved: {rounds_path}")


if __name__ == "__main__":
    build_charts()
