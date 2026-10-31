"""Verify and generate all result files needed for the demo.

Run this once before the guide review:
    python scripts/demo_prep.py

It will:
1. Check all required checkpoints exist.
2. Generate crosshospital_fedbn.csv if missing.
3. Regenerate summary CSVs and charts.
4. Print a go/no-go checklist.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medifed.config import CHECKPOINT_DIR, RESULTS_DIR

REQUIRED_CHECKPOINTS = [
    # FedAvg global is optional — dashboard metrics come from saved CSVs.
    # CHECKPOINT_DIR / "baseline_seed42_global.pt",
    CHECKPOINT_DIR / "personalized_seed42_global.pt",
    CHECKPOINT_DIR / "personalized_seed42" / "hospital1_bn.pt",
    CHECKPOINT_DIR / "personalized_seed42" / "hospital2_bn.pt",
    CHECKPOINT_DIR / "personalized_seed42" / "hospital3_bn.pt",
    CHECKPOINT_DIR / "local_hospital1.pt",
    CHECKPOINT_DIR / "local_hospital2.pt",
    CHECKPOINT_DIR / "local_hospital3.pt",
]

REQUIRED_RESULTS = [
    RESULTS_DIR / "comparison_methods.csv",
    RESULTS_DIR / "summary_rounds.csv",
    RESULTS_DIR / "summary_final.csv",
    RESULTS_DIR / "local_final_metrics.csv",
    RESULTS_DIR / "baseline_seed42_final_metrics.csv",
    RESULTS_DIR / "personalized_seed42_final_metrics.csv",
    RESULTS_DIR / "data_counts.csv",
    RESULTS_DIR / "local_cross_eval.csv",
]


def check(label: str, ok: bool) -> bool:
    status = "  OK " if ok else "  MISSING"
    print(f"{status}  {label}")
    return ok


def main() -> None:
    print("=" * 60)
    print("MediFed demo preparation check")
    print("=" * 60)

    print("\n[Checkpoints]")
    ckpt_ok = all(check(str(p.relative_to(PROJECT_ROOT)), p.exists()) for p in REQUIRED_CHECKPOINTS)

    print("\n[Result files]")
    results_ok = all(check(str(p.relative_to(PROJECT_ROOT)), p.exists()) for p in REQUIRED_RESULTS)

    # Generate crosshospital_fedbn.csv if missing
    cross_path = RESULTS_DIR / "crosshospital_fedbn.csv"
    if not cross_path.exists():
        print("\n[Generating crosshospital_fedbn.csv…]")
        if ckpt_ok:
            from scripts.cross_hospital_eval import main as run_cross
            run_cross()
        else:
            print("  SKIP — checkpoints missing, cannot generate cross-hospital results.")
    check(str(cross_path.relative_to(PROJECT_ROOT)), cross_path.exists())

    # Regenerate summary CSVs and charts
    print("\n[Regenerating summary CSVs and charts…]")
    try:
        from scripts.summarize_runs import write_summary_files
        write_summary_files(RESULTS_DIR, RESULTS_DIR / "runs")
        print("  OK  summary CSVs written")
    except Exception as exc:
        print(f"  WARN  summarize_runs failed: {exc}")

    try:
        from scripts.make_charts import build_charts
        build_charts()
        print("  OK  charts written")
    except Exception as exc:
        print(f"  WARN  make_charts failed: {exc}")

    print("\n" + "=" * 60)
    all_ok = ckpt_ok and results_ok and cross_path.exists()
    if all_ok:
        print("READY — start the app with:  .\\scripts\\run_app.ps1")
    else:
        print("NOT READY — fix missing items above before the demo.")
    print("=" * 60)


if __name__ == "__main__":
    main()
