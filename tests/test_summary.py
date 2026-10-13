"""Tests for the federated baseline summary helpers."""

import csv
import tempfile
import unittest
from pathlib import Path

from scripts import summarize_runs


class TestSummary(unittest.TestCase):
    """Test the summary generation for baseline federated runs."""

    def test_summary_generation_on_tiny_runs(self):
        """TC-SUM-001 mean/std on a tiny synthetic runs folder."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir) / "results"
            runs_dir = results_dir / "runs"
            runs_dir.mkdir(parents=True)

            for seed in (42, 43):
                run_dir = runs_dir / f"baseline_seed{seed}"
                run_dir.mkdir()
                for hospital_id in (1, 2):
                    with open(run_dir / f"hospital{hospital_id}_rounds.csv", "w", newline="", encoding="utf-8") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(["round", "hospital", "split", "loss", "accuracy", "num_samples"])
                        writer.writerow([1, hospital_id, "val", 0.2, 0.8 + seed * 0.01, 10])
                        writer.writerow([2, hospital_id, "val", 0.3, 0.9 + seed * 0.01, 10])

            summary = summarize_runs.build_summary(runs_dir)
            self.assertEqual(summary["summary_rounds"][0]["mode"], "baseline")
            self.assertEqual(summary["summary_rounds"][0]["hospital"], 1)
            self.assertIn("mean_val_acc", summary["summary_rounds"][0])

    def test_mode_and_seed_parsed_from_folders(self):
        """TC-SUM-002 mode and seed parsed from folder names."""
        self.assertEqual(summarize_runs.parse_run_name("baseline_seed42"), ("baseline", 42))
        self.assertEqual(summarize_runs.parse_run_name("personalized_seed44"), ("personalized", 44))

    def test_missing_personalized_runs_leave_fedbn_empty(self):
        """TC-SUM-003 missing personalized runs leave fedbn columns empty."""
        comparison = summarize_runs.build_comparison_table(
            local_file=None,
            fedavg_runs=[],
            fedbn_runs=[],
            hospitals=[1, 2, 3],
        )
        self.assertEqual(comparison[0]["fedbn_mean"], "")


if __name__ == "__main__":
    unittest.main()
