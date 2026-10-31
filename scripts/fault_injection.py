"""Apply deliberate code faults to a temporary copy and report which tests fail."""

from __future__ import annotations

import csv
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"

FAULTS = [
    ("global-order-change", "medifed/config.py", 'CLASS_NAMES = ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]', 'CLASS_NAMES = ["NORMAL", "COVID19", "PNEUMONIA", "TUBERCULOSIS"]'),
    ("merge-local-bn-from-global", "medifed/personalization.py", "for key in keys:\n        merged[key] = local_bn[key]", "for key in keys:\n        merged[key] = global_state[key]"),
    ("remove-junk-filter", "medifed/data.py", 'if filepath.name.startswith("._"):\n            skipped["junk"] += 1\n            continue\n', ""),
    ("cap-off-by-one", "medifed/data.py", '        n_val = max(1, round(len(shuffled) * val_fraction))\n', '        n_val = max(1, round(len(shuffled) * val_fraction) + 1)\n'),
    ("partition-swap", "medifed/federated.py", "    mapping = {0: 1, 1: 2, 2: 3}\n", "    mapping = {0: 2, 1: 1, 2: 3}\n"),
    ("skip-bn-key-check", "medifed/personalization.py", '    missing = [key for key in keys if key not in local_bn]\n    if missing:\n        raise KeyError(f"Missing local BN keys: {missing[:3]}" )\n', ""),
    ("drop-round-reset", "medifed/federated.py", '    if server_round == 1:\n        Path(csv_path).unlink(missing_ok=True)\n', ""),
    ("validation-fraction-twice", "medifed/config.py", 'VAL_FRACTION = 0.2', 'VAL_FRACTION = 0.5'),
]


def apply_fault(copy_root: Path, file_rel: str, old: str, new: str) -> bool:
    path = copy_root / file_rel
    if not path.exists():
        return False
    text = path.read_text(encoding='utf-8')
    if old not in text:
        return False
    path.write_text(text.replace(old, new), encoding='utf-8')
    return True


def discover_tests() -> list[str]:
    return [
        "tests.test_config",
        "tests.test_data",
        "tests.test_federated",
        "tests.test_model",
        "tests.test_personalization",
        "tests.test_plots",
        "tests.test_summary",
    ]


def run_faults() -> list[dict]:
    results: list[dict] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        shutil.copytree(PROJECT_ROOT / "medifed", tmp_root / "medifed")
        shutil.copytree(PROJECT_ROOT / "tests", tmp_root / "tests")
        for fault_name, file_rel, old, new in FAULTS:
            temp_copy = tmp_root / "copy"
            if temp_copy.exists():
                shutil.rmtree(temp_copy)
            temp_copy.mkdir(parents=True, exist_ok=True)
            shutil.copytree(tmp_root / "medifed", temp_copy / "medifed")
            shutil.copytree(tmp_root / "tests", temp_copy / "tests")
            if not apply_fault(temp_copy, file_rel, old, new):
                results.append({"fault": fault_name, "status": "SKIPPED", "failing_tests": "text not found"})
                continue
            cmd = [str(VENV_PYTHON), "-m", "unittest"]
            cmd.extend(discover_tests())
            proc = subprocess.run(cmd, cwd=temp_copy, capture_output=True, text=True, timeout=120)
            status = "CAUGHT" if proc.returncode != 0 else "SURVIVED"
            failing = ""
            for line in (proc.stdout + proc.stderr).splitlines():
                if line.startswith("FAIL:") or "FAIL:" in line:
                    failing = line
            results.append({"fault": fault_name, "status": status, "failing_tests": failing})
    return results


def main() -> None:
    rows = run_faults()
    csv_path = PROJECT_ROOT / "results" / "fault_injection.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fault", "status", "failing_tests"])
        writer.writeheader()
        writer.writerows(rows)
    print(csv_path)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
