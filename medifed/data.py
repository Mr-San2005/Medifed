"""Data pipeline for MediFed: find, filter, cap, split, and load chest X-ray images."""

import csv
import random
from pathlib import Path, PurePosixPath

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from medifed.config import (
    CLASS_TO_INDEX,
    EXCLUDE_FOLDER_NAMES,
    IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    MAX_IMAGES_PER_CLASS,
    PROCESSED_DIR,
    SEED,
    VAL_FRACTION,
    HOSPITAL_RAW_DIRS,
    FOLDER_TO_CLASS,
)


def _is_excluded(path: Path) -> bool:
    """Return True if any component of path matches an excluded folder name (any case)."""
    exclude_lower = {n.lower() for n in EXCLUDE_FOLDER_NAMES}
    return any(part.lower() in exclude_lower for part in path.parts)


def find_images(raw_dir: Path, folder_to_class: dict) -> tuple[dict, dict]:
    """Walk raw_dir and collect valid image paths grouped by class.

    The class of an image is determined by the nearest ancestor folder whose
    lowercase name appears in folder_to_class.  This handles layouts where
    images sit inside a sub-folder (e.g. COVID19/images/x.png).

    Returns:
        images_by_class: dict[class_name -> sorted list of Path]
        skipped:         dict with counts for each skip reason
    """
    images_by_class: dict[str, list[Path]] = {cls: [] for cls in folder_to_class.values()}
    skipped = {"junk": 0, "extension": 0, "excluded": 0, "unreadable": 0}

    for filepath in sorted(raw_dir.rglob("*")):
        if not filepath.is_file():
            continue

        # Skip macOS AppleDouble junk files.
        if filepath.name.startswith("._"):
            skipped["junk"] += 1
            continue

        # Skip files with disallowed extensions.
        if filepath.suffix.lower() not in IMAGE_EXTENSIONS:
            skipped["extension"] += 1
            continue

        # Skip files inside excluded folders (masks, Lung_Opacity, __MACOSX, etc.).
        # Check the path relative to raw_dir so we only look at sub-components.
        rel = filepath.relative_to(raw_dir)
        if _is_excluded(rel):
            skipped["excluded"] += 1
            continue

        # Determine class from the nearest matching ancestor folder.
        class_name = None
        for part in reversed(rel.parts[:-1]):  # walk ancestors from nearest to root
            if part.lower() in folder_to_class:
                class_name = folder_to_class[part.lower()]
                break
        if class_name is None:
            # No matching ancestor — not part of any known class folder.
            skipped["excluded"] += 1
            continue

        # Skip files that PIL cannot open (corrupt or truncated).
        try:
            with Image.open(filepath) as img:
                img.verify()
        except Exception:
            skipped["unreadable"] += 1
            continue

        images_by_class[class_name].append(filepath)

    # Sort for determinism before the random cap is applied.
    for cls in images_by_class:
        images_by_class[cls].sort()

    return images_by_class, skipped


def build_manifest(
    images_by_class: dict,
    raw_dir: Path,
    cap: int = MAX_IMAGES_PER_CLASS,
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
) -> list[dict]:
    """Cap each class, then split into train/val.  Returns a list of row dicts.

    Paths are stored relative to RAW_DIR with forward slashes so the manifest
    is portable across machines.  Class indices come from the global CLASS_TO_INDEX.
    """
    rng = random.Random(seed)
    rows: list[dict] = []

    for class_name, paths in sorted(images_by_class.items()):
        # Cap: sample at most `cap` images per class using the fixed seed.
        kept = paths if len(paths) <= cap else rng.sample(paths, cap)
        kept = sorted(kept)  # sort after sample for determinism

        # Stratified split: shuffle then cut at val_fraction boundary.
        shuffled = kept[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_fraction))
        val_paths  = set(shuffled[:n_val])
        train_paths = shuffled[n_val:]

        for p in train_paths:
            rows.append(_make_row(p, class_name, "train", raw_dir))
        for p in val_paths:
            rows.append(_make_row(p, class_name, "val", raw_dir))

    return rows


def _make_row(path: Path, class_name: str, split: str, raw_dir: Path) -> dict:
    """Build one manifest row with a portable relative path."""
    rel = path.relative_to(raw_dir)
    # Forward slashes so the manifest is readable on any OS.
    rel_str = str(PurePosixPath(rel))
    return {
        "relative_path": rel_str,
        "class_name":    class_name,
        "class_index":   CLASS_TO_INDEX[class_name],
        "split":         split,
    }


def write_manifest(rows: list[dict], path: Path) -> None:
    """Write manifest rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "class_name", "class_index", "split"])
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> list[dict]:
    """Read a manifest CSV and return rows as dicts (class_index as int)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["class_index"] = int(row["class_index"])
            rows.append(row)
    return rows


def get_transforms(train: bool) -> transforms.Compose:
    """Return the image transform pipeline.

    RandomRotation(5) is added for training only.  No horizontal flip because
    chest X-rays have a fixed left-right orientation (heart position matters).
    """
    ops = []
    if train:
        ops.append(transforms.RandomRotation(5))
    ops += [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        # ImageNet mean/std — the pretrained ResNet18 backbone expects this normalisation.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(ops)


class ChestXrayDataset(Dataset):
    """PyTorch Dataset built from a manifest.  Returns (image_tensor, class_index)."""

    def __init__(self, rows: list[dict], raw_dir: Path, transform) -> None:
        self.rows = rows
        self.raw_dir = raw_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple:
        row = self.rows[idx]
        # Reconstruct the absolute path from the stored relative path.
        img_path = self.raw_dir / Path(row["relative_path"])
        # Convert to RGB so grayscale X-rays work with the 3-channel ResNet backbone.
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, row["class_index"]


def make_loader(
    hospital_id: int,
    split: str,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Build a DataLoader for one hospital and split from its saved manifest."""
    manifest_path = PROCESSED_DIR / f"hospital{hospital_id}_manifest.csv"
    rows = read_manifest(manifest_path)
    split_rows = [r for r in rows if r["split"] == split]
    transform = get_transforms(train=(split == "train"))
    # Paths in the manifest are relative to the hospital's own raw dir.
    raw_dir = HOSPITAL_RAW_DIRS[hospital_id]
    dataset = ChestXrayDataset(split_rows, raw_dir, transform)
    # num_workers=0 is required on Windows to avoid multiprocessing issues.
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
