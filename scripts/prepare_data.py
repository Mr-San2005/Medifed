"""Build manifests for all three hospitals and save per-class count summary."""

import csv
import sys
from pathlib import Path

# Ensure the project root is importable regardless of launch directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medifed.config import (
    CLASS_NAMES,
    FOLDER_TO_CLASS,
    HOSPITAL_RAW_DIRS,
    HOSPITALS,
    MAX_IMAGES_PER_CLASS,
    PROCESSED_DIR,
    RESULTS_DIR,
)
from medifed.data import build_manifest, find_images, write_manifest


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for hid in (1, 2, 3):
        raw_dir = HOSPITAL_RAW_DIRS[hid]
        f2c = FOLDER_TO_CLASS[hid]
        hospital_name = HOSPITALS[hid]["name"]

        print(f"\n{'='*60}")
        print(f"Hospital {hid}: {hospital_name}")
        print(f"Raw dir: {raw_dir}")

        # Verify expected class folders exist and are non-empty.
        for cls in HOSPITALS[hid]["classes"]:
            # Find the folder whose lowercase name matches the class key.
            matching = [d for d in raw_dir.iterdir()
                        if d.is_dir() and d.name.lower() in f2c
                        and f2c[d.name.lower()] == cls]
            if not matching:
                sys.exit(f"ERROR: Expected class folder for '{cls}' not found under {raw_dir}. "
                         f"Folders found: {[d.name for d in raw_dir.iterdir() if d.is_dir()]}")
            folder = matching[0]
            files = list(folder.rglob("*"))
            if not any(f.is_file() for f in files):
                sys.exit(f"ERROR: Class folder '{folder}' exists but contains no files.")

        # Scan for images, applying all filters.
        images_by_class, skipped = find_images(raw_dir, f2c)

        # Verify no unexpected folder names exist (stop rather than guess).
        actual_dirs = {d.name.lower() for d in raw_dir.iterdir() if d.is_dir()}
        unknown = actual_dirs - set(f2c.keys())
        if unknown:
            sys.exit(
                f"ERROR: Hospital {hid} has folder(s) not in FOLDER_TO_CLASS: {unknown}. "
                f"Update FOLDER_TO_CLASS[{hid}] before proceeding."
            )

        print(f"\nSkipped files:")
        print(f"  junk (._):        {skipped['junk']}")
        print(f"  wrong extension:  {skipped['extension']}")
        print(f"  excluded folder:  {skipped['excluded']}")
        print(f"  unreadable (PIL): {skipped['unreadable']}")

        print(f"\nPer-class counts (cap={MAX_IMAGES_PER_CLASS}):")
        print(f"  {'Class':<16} {'Found':>6} {'Kept':>6}")
        print(f"  {'-'*30}")

        for cls in CLASS_NAMES:
            if cls not in images_by_class:
                continue
            found = len(images_by_class[cls])
            kept  = min(found, MAX_IMAGES_PER_CLASS)
            print(f"  {cls:<16} {found:>6} {kept:>6}")

        # Build and save the manifest.
        rows = build_manifest(images_by_class, raw_dir)
        manifest_path = PROCESSED_DIR / f"hospital{hid}_manifest.csv"
        write_manifest(rows, manifest_path)

        # Compute train/val counts per class for the summary.
        print(f"\nTrain / Val split:")
        print(f"  {'Class':<16} {'Train':>6} {'Val':>6}")
        print(f"  {'-'*30}")

        for cls in CLASS_NAMES:
            if cls not in images_by_class:
                continue
            found = len(images_by_class[cls])
            kept  = min(found, MAX_IMAGES_PER_CLASS)
            train_n = sum(1 for r in rows if r["class_name"] == cls and r["split"] == "train")
            val_n   = sum(1 for r in rows if r["class_name"] == cls and r["split"] == "val")
            print(f"  {cls:<16} {train_n:>6} {val_n:>6}")
            summary_rows.append({
                "hospital": hid,
                "class":    cls,
                "found":    found,
                "kept":     kept,
                "train":    train_n,
                "val":      val_n,
            })

        print(f"\nManifest saved: {manifest_path}")

    # Save the cross-hospital summary CSV.
    counts_path = RESULTS_DIR / "data_counts.csv"
    with open(counts_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["hospital", "class", "found", "kept", "train", "val"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSummary saved: {counts_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
