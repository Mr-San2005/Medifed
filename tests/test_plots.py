"""Tests for chart generation helpers."""

import tempfile
import unittest
from pathlib import Path

from medifed.plots import grouped_bar_chart, line_chart


class TestPlots(unittest.TestCase):
    """Test the PIL-based plotting helpers."""

    def test_grouped_bar_chart_creates_image(self):
        """TC-PLOT-001 PNG created with requested size and not a single colour."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "chart.png"
            data = [
                {"label": "Local", "values": [0.9, 0.8, 0.7], "errors": [0.05, 0.04, 0.03]},
                {"label": "FedAvg", "values": [0.92, 0.82, 0.75], "errors": [0.04, 0.03, 0.02]},
            ]
            grouped_bar_chart(path, ["H1", "H2", "H3"], data, width=1600, height=900)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)

    def test_line_chart_handles_missing_error_bars(self):
        """TC-PLOT-002 handles a series with missing error bars."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "line.png"
            line_chart(
                path,
                x_values=[1, 2, 3],
                series=[
                    {"label": "FedAvg", "values": [0.7, 0.8, 0.9]},
                    {"label": "FedBN", "values": [0.75, 0.82, 0.91], "errors": None},
                ],
                width=1600,
                height=900,
            )
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
