"""Unit tests for medifed/data.py using a temporary fake raw directory."""

import io
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Helpers to build a minimal fake raw directory in a temp folder.
# ---------------------------------------------------------------------------

def _make_rgb_image(path: Path, size: int = 8) -> None:
    """Save a tiny valid RGB PNG to path (creates parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color=(128, 64, 32)).save(path)


def _make_fake_raw(base: Path) -> dict:
    """Build a fake raw directory that exercises every filter and layout case.

    Layout:
      base/
        NORMAL/
          good1.png          <- valid
          ._junk.jpg         <- junk (starts with ._)
          corrupt.jpg        <- corrupt (not a real image)
          masks/
            masked.png       <- excluded folder
          Lung_Opacity/
            lo.png           <- excluded folder
          __MACOSX/
            mac.png          <- excluded folder
        TUBERCULOSIS/
          good2.png          <- valid
          good3.png          <- valid
          good4.png          <- valid
        COVID19/
          images/
            nested.png       <- valid; class comes from COVID19, not images
    """
    # NORMAL class images.
    _make_rgb_image(base / "NORMAL" / "good1.png")

    # Junk file (starts with ._).
    junk = base / "NORMAL" / "._junk.jpg"
    junk.parent.mkdir(parents=True, exist_ok=True)
    junk.write_bytes(b"fake bytes")

    # Corrupt image (valid extension, invalid content).
    corrupt = base / "NORMAL" / "corrupt.jpg"
    corrupt.write_bytes(b"\xff\xd8\xff not a real jpeg")

    # Excluded folders.
    _make_rgb_image(base / "NORMAL" / "masks" / "masked.png")
    _make_rgb_image(base / "NORMAL" / "Lung_Opacity" / "lo.png")
    _make_rgb_image(base / "NORMAL" / "__MACOSX" / "mac.png")

    # TUBERCULOSIS class — 3 valid images (for cap test).
    _make_rgb_image(base / "TUBERCULOSIS" / "good2.png")
    _make_rgb_image(base / "TUBERCULOSIS" / "good3.png")
    _make_rgb_image(base / "TUBERCULOSIS" / "good4.png")

    # COVID19 with nested images/ sub-folder (hospital-3 style layout).
    _make_rgb_image(base / "COVID19" / "images" / "nested.png")

    return {
        "folder_to_class": {
            "normal":       "NORMAL",
            "tuberculosis": "TUBERCULOSIS",
            "covid19":      "COVID19",
        }
    }


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestFindImages(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        info = _make_fake_raw(self.tmp)
        self.f2c = info["folder_to_class"]

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _run(self):
        from medifed.data import find_images
        return find_images(self.tmp, self.f2c)

    def test_junk_skipped(self):
        """TC-DATA-001: '._' junk file is not returned and is counted in skipped['junk']."""
        _, skipped = self._run()
        self.assertGreaterEqual(skipped["junk"], 1)

    def test_corrupt_skipped(self):
        """TC-DATA-002: corrupt image is not returned and is counted in skipped['unreadable']."""
        images, skipped = self._run()
        all_names = [p.name for paths in images.values() for p in paths]
        self.assertNotIn("corrupt.jpg", all_names)
        self.assertGreaterEqual(skipped["unreadable"], 1)

    def test_masks_excluded(self):
        """TC-DATA-003: images inside a 'masks' folder are excluded and counted."""
        images, skipped = self._run()
        all_names = [p.name for paths in images.values() for p in paths]
        self.assertNotIn("masked.png", all_names)
        self.assertGreaterEqual(skipped["excluded"], 1)

    def test_lung_opacity_excluded(self):
        """TC-DATA-004: images inside 'Lung_Opacity' folder are excluded."""
        images, skipped = self._run()
        all_names = [p.name for paths in images.values() for p in paths]
        self.assertNotIn("lo.png", all_names)

    def test_macosx_excluded(self):
        """TC-DATA-005: images inside '__MACOSX' folder are excluded."""
        images, skipped = self._run()
        all_names = [p.name for paths in images.values() for p in paths]
        self.assertNotIn("mac.png", all_names)

    def test_nearest_ancestor_class(self):
        """TC-DATA-006: class comes from the nearest matching ancestor (COVID19/images/x.png -> COVID19)."""
        images, _ = self._run()
        covid_names = [p.name for p in images.get("COVID19", [])]
        self.assertIn("nested.png", covid_names)


class TestBuildManifest(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        info = _make_fake_raw(self.tmp)
        self.f2c = info["folder_to_class"]

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _images(self):
        from medifed.data import find_images
        images, _ = find_images(self.tmp, self.f2c)
        return images

    def test_cap_respected(self):
        """TC-DATA-007: per-class cap is respected; no class exceeds the cap."""
        from medifed.data import build_manifest
        images = self._images()
        rows = build_manifest(images, self.tmp, cap=2)
        from collections import Counter
        counts = Counter(r["class_name"] for r in rows)
        for cls, count in counts.items():
            self.assertLessEqual(count, 2, f"Class {cls} has {count} rows, cap is 2")

    def test_same_seed_same_sample(self):
        """TC-DATA-008: the same seed produces an identical sample on two calls."""
        from medifed.data import build_manifest
        images = self._images()
        rows1 = build_manifest(images, self.tmp, cap=2, seed=99)
        rows2 = build_manifest(images, self.tmp, cap=2, seed=99)
        self.assertEqual(
            [(r["relative_path"], r["split"]) for r in rows1],
            [(r["relative_path"], r["split"]) for r in rows2],
        )

    def test_global_indices_hospital2_style(self):
        """TC-DATA-009: hospital-2-style folder gives NORMAL=1 and TUBERCULOSIS=3 (global indices)."""
        from medifed.data import build_manifest, find_images
        # Use only NORMAL and TUBERCULOSIS (hospital-2 style).
        f2c = {"normal": "NORMAL", "tuberculosis": "TUBERCULOSIS"}
        images, _ = find_images(self.tmp, f2c)
        rows = build_manifest(images, self.tmp)
        index_map = {r["class_name"]: r["class_index"] for r in rows}
        self.assertEqual(index_map.get("NORMAL"),       1)
        self.assertEqual(index_map.get("TUBERCULOSIS"), 3)

    def test_train_val_no_overlap(self):
        """TC-DATA-010: train and val splits are stratified per class and have no path overlap."""
        from medifed.data import build_manifest
        images = self._images()
        rows = build_manifest(images, self.tmp)
        train_paths = {r["relative_path"] for r in rows if r["split"] == "train"}
        val_paths   = {r["relative_path"] for r in rows if r["split"] == "val"}
        self.assertEqual(train_paths & val_paths, set(), "Train and val sets overlap")
        # Both splits must be non-empty (we have enough images).
        self.assertGreater(len(train_paths), 0)
        self.assertGreater(len(val_paths),   0)


class TestManifestIO(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_write_read_roundtrip(self):
        """TC-DATA-011: manifest write/read round-trip is lossless; relative paths use forward slashes."""
        from medifed.data import read_manifest, write_manifest
        rows = [
            {"relative_path": "hospital1/NORMAL/img.png", "class_name": "NORMAL",
             "class_index": 1, "split": "train"},
            {"relative_path": "hospital2/TUBERCULOSIS/img.png", "class_name": "TUBERCULOSIS",
             "class_index": 3, "split": "val"},
        ]
        path = self.tmp / "test_manifest.csv"
        write_manifest(rows, path)
        loaded = read_manifest(path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["relative_path"], "hospital1/NORMAL/img.png")
        self.assertEqual(loaded[1]["class_index"],   3)
        # Forward slashes only.
        for row in loaded:
            self.assertNotIn("\\", row["relative_path"])


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        info = _make_fake_raw(self.tmp)
        self.f2c = info["folder_to_class"]

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_dataset_item_shape_and_type(self):
        """TC-DATA-012: dataset item is a float tensor of shape 3x224x224 with an integer label."""
        import torch
        from medifed.data import ChestXrayDataset, build_manifest, find_images, get_transforms
        images, _ = find_images(self.tmp, self.f2c)
        rows = build_manifest(images, self.tmp)
        dataset = ChestXrayDataset(rows, self.tmp, get_transforms(train=False))
        tensor, label = dataset[0]
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertIsInstance(label, int)


if __name__ == "__main__":
    unittest.main()
