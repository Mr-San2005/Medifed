"""Tests for the review-2 demonstration scripts and validations."""

import subprocess
import unittest
from pathlib import Path

from medifed.config import HOSPITALS
from scripts import demo_predict


class TestDemoScripts(unittest.TestCase):

    def test_demo_selection_is_deterministic_and_hospital_specific(self):
        """TC-DEMO-001 demo selection is deterministic and uses only classes of that hospital."""
        for hospital_id in (1, 2, 3):
            first = demo_predict.select_demo_images(hospital_id)
            second = demo_predict.select_demo_images(hospital_id)
            self.assertEqual(first, second)
            self.assertTrue(set(first).issubset(set(HOSPITALS[hospital_id]["classes"])))

    def test_unknown_method_raises_value_error(self):
        """TC-DEMO-002 an unknown method name raises ValueError."""
        with self.assertRaises(ValueError):
            demo_predict.load_model("not_real", 1)

    def test_demo_review2_script_parses(self):
        """TC-DEMO-003 scripts\demo_review2.ps1 parses."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "scripts" / "demo_review2.ps1"
        command = (
            "$errors=$null; "
            "[void][System.Management.Automation.Language.Parser]::ParseFile("
            f"'{script_path}', [ref]$null, [ref]$errors); "
            "if ($errors) { $errors | Format-List *; exit 1 } "
            "else { Write-Output 'syntax OK' }"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("syntax OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
