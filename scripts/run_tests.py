"""Discover and run all tests in tests/, save output with timestamp, exit non-zero on failure."""

import sys
import unittest
import io
from datetime import datetime
from pathlib import Path

# Ensure the project root is on sys.path regardless of where this script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TESTS_DIR    = PROJECT_ROOT / "tests"
OUTPUT_DIR   = PROJECT_ROOT / "docs" / "04_testing" / "test_runs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = OUTPUT_DIR / f"test_run_{timestamp}.txt"

# Capture the runner output so it can be written to file and printed.
buffer = io.StringIO()
runner = unittest.TextTestRunner(stream=buffer, verbosity=2)
loader = unittest.TestLoader()
suite  = loader.discover(start_dir=str(TESTS_DIR), pattern="test_*.py")
result = runner.run(suite)

output = buffer.getvalue()
output_file.write_text(output, encoding="utf-8")

print(output)
print("=" * 60)
print(f"Run:    {result.testsRun}")
print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
print(f"Failed: {len(result.failures)}")
print(f"Errors: {len(result.errors)}")
print(f"Output: {output_file}")
print("=" * 60)

sys.exit(0 if result.wasSuccessful() else 1)
