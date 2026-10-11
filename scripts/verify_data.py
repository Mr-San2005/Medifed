"""Verify manifests and data integrity; save a sample grid image."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw

from medifed.config import (
    CLASS_TO_INDEX,
    HOSPITAL_RAW_DIRS,
    PROCESSED_DIR,
    RESULTS_DIR,
)
from medifed.data import make_loader, read_manifest


# Grid layout constants.
SAMPLES_PER_ROW = 4
TILE_SIZE       = 160   # pixels per tile in the output grid
LABEL_HEIGHT    = 20    # pixels reserved for the class label below each tile
BG_COLOR        = (245, 247, 248)
TEXT_COLOR      = (27, 42, 51)


def verify_hospital(hid: int) -> list[dict]:
    """Read the manifest for one hospital, check files exist and indices are valid."""
    manifest_path = PROCESSED_DIR / f"hospital{hid}_manifest.csv"
    if not manifest_path.exists():
        sys.exit(f"FAIL: manifest not found: {manifest_path}")

    rows = read_manifest(manifest_path)
    # Paths in the manifest are relative to the hospital's own raw dir.
    raw_dir = HOSPITAL_RAW_DIRS[hid]
    errors: list[str] = []

    for row in rows:
        abs_path = raw_dir / Path(row["relative_path"])
        if not abs_path.exists():
            errors.append(f"Missing file: {abs_path}")
        if row["class_index"] not in CLASS_TO_INDEX.values():
            errors.append(f"Invalid class_index {row['class_index']} in {row['relative_path']}")

    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(f"FAIL: {len(errors)} error(s) in hospital {hid} manifest.")

    return rows


def print_manifest_table(hid: int, rows: list[dict]) -> None:
    """Print a per-class count table for one hospital."""
    from collections import Counter
    counts = Counter((r["class_name"], r["split"]) for r in rows)
    classes = sorted({r["class_name"] for r in rows})
    print(f"\n  Hospital {hid}:")
    print(f"  {'Class':<16} {'Train':>6} {'Val':>6} {'Total':>6}")
    print(f"  {'-'*36}")
    for cls in classes:
        tr = counts.get((cls, "train"), 0)
        va = counts.get((cls, "val"),   0)
        print(f"  {cls:<16} {tr:>6} {va:>6} {tr+va:>6}")


def sample_tile(hid: int, rows: list[dict]) -> list[tuple]:
    """Return up to SAMPLES_PER_ROW (image, label) pairs from the val split."""
    raw_dir = HOSPITAL_RAW_DIRS[hid]
    val_rows = [r for r in rows if r["split"] == "val"]
    # Take one from each class if possible, then fill up to SAMPLES_PER_ROW.
    seen_classes: set = set()
    selected = []
    for r in val_rows:
        if r["class_name"] not in seen_classes:
            seen_classes.add(r["class_name"])
            selected.append(r)
        if len(selected) == SAMPLES_PER_ROW:
            break
    # Fill remaining slots if fewer than SAMPLES_PER_ROW distinct classes.
    for r in val_rows:
        if len(selected) >= SAMPLES_PER_ROW:
            break
        if r not in selected:
            selected.append(r)

    tiles = []
    for r in selected:
        img_path = raw_dir / Path(r["relative_path"])
        img = Image.open(img_path).convert("RGB").resize((TILE_SIZE, TILE_SIZE))
        tiles.append((img, r["class_name"]))
    return tiles


def build_grid(all_tiles: list[list[tuple]]) -> Image.Image:
    """Compose a grid image: one row per hospital, SAMPLES_PER_ROW tiles per row."""
    n_rows = len(all_tiles)
    tile_h = TILE_SIZE + LABEL_HEIGHT
    grid_w = TILE_SIZE * SAMPLES_PER_ROW
    grid_h = tile_h * n_rows

    grid = Image.new("RGB", (grid_w, grid_h), BG_COLOR)
    draw = ImageDraw.Draw(grid)

    for row_idx, tiles in enumerate(all_tiles):
        for col_idx, (img, label) in enumerate(tiles):
            x = col_idx * TILE_SIZE
            y = row_idx * tile_h
            grid.paste(img, (x, y))
            # Label below the tile.
            draw.text((x + 4, y + TILE_SIZE + 2), label, fill=TEXT_COLOR)

    return grid


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Verifying manifests...")
    all_tiles = []

    for hid in (1, 2, 3):
        rows = verify_hospital(hid)
        print_manifest_table(hid, rows)

        # Load one batch and print tensor shape + label set.
        loader = make_loader(hid, split="val", batch_size=4, shuffle=False)
        images, labels = next(iter(loader))
        print(f"\n  Hospital {hid} batch — tensor shape: {tuple(images.shape)}, "
              f"labels: {sorted(set(labels.tolist()))}")

        all_tiles.append(sample_tile(hid, rows))

    # Save the sample grid.
    grid = build_grid(all_tiles)
    grid_path = RESULTS_DIR / "sample_grid.png"
    grid.save(grid_path)
    print(f"\nSample grid saved: {grid_path}")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
