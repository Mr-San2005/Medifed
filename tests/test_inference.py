"""Tests for checkpoint loading and image prediction variants."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from PIL import Image

from medifed import inference
from medifed.personalization import bn_keys, split_state


class TinyInferenceModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.layer4 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 4)

    def forward(self, image):
        return self.fc(self.pool(self.layer4(self.conv1(image))).flatten(1))


def make_png() -> bytes:
    image = Image.new("RGB", (40, 32), (130, 110, 100))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class TestInference(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(19)
        inference._load_variant_cached.cache_clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        model = TinyInferenceModel()
        torch.save(model.state_dict(), self.root / "baseline_seed42_global.pt")
        torch.save(model.state_dict(), self.root / "personalized_seed42_global.pt")
        for hospital_id in (1, 2):
            _, bn_state = split_state(model.state_dict(), bn_keys(model))
            if hospital_id == 2:
                bn_state["layer4.1.running_mean"] = bn_state["layer4.1.running_mean"] + 3
            bn_dir = self.root / "personalized_seed42"
            bn_dir.mkdir(exist_ok=True)
            torch.save(bn_state, bn_dir / f"hospital{hospital_id}_bn.pt")

    def tearDown(self):
        inference._load_variant_cached.cache_clear()
        self.temp_dir.cleanup()

    def test_prediction_returns_four_sorted_probabilities(self):
        """TC-INF-001 prediction ranks four probabilities that sum to one."""
        with patch("medifed.inference.build_model", side_effect=TinyInferenceModel):
            result = inference.predict("fedavg_global", make_png(), self.root, torch.device("cpu"))
        ranked = result["ranked"]
        values = [item["probability"] for item in ranked]
        self.assertEqual(len(ranked), 4)
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertAlmostEqual(sum(values), 1.0, places=5)
        self.assertTrue(result["heatmap_png"])

    def test_unknown_variant_raises_value_error(self):
        """TC-INF-002 unknown variant names raise ValueError."""
        with self.assertRaises(ValueError):
            inference.load_variant("unknown", self.root, torch.device("cpu"))

    def test_missing_checkpoint_names_the_file(self):
        """TC-INF-003 a missing model checkpoint raises a named FileNotFoundError."""
        with patch("medifed.inference.build_model", side_effect=TinyInferenceModel):
            with self.assertRaisesRegex(FileNotFoundError, "baseline_seed42_global.pt"):
                inference.load_variant("fedavg_global", self.root / "missing", torch.device("cpu"))

    def test_invalid_image_bytes_raise_value_error(self):
        """TC-INF-004 undecodable image bytes raise the documented ValueError."""
        with self.assertRaisesRegex(ValueError, "not a valid image"):
            inference.predict("fedavg_global", b"bad image", self.root, torch.device("cpu"))

    def test_hospital_variant_uses_its_saved_batchnorm_state(self):
        """TC-INF-005 personalized variants use the selected hospital BatchNorm checkpoint."""
        with patch("medifed.inference.build_model", side_effect=TinyInferenceModel):
            first = inference.load_variant("fedbn_hospital1", self.root, torch.device("cpu"))
            second = inference.load_variant("fedbn_hospital2", self.root, torch.device("cpu"))
        tensor = torch.ones(1, 3, 32, 32)
        with torch.no_grad():
            first_output = first(tensor)
            second_output = second(tensor)
        self.assertFalse(torch.allclose(first_output, second_output))


if __name__ == "__main__":
    unittest.main()