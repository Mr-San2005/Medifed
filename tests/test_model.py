"""Unit tests for medifed/model.py and medifed/train.py."""

import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from medifed.config import NUM_CLASSES
from medifed.model import build_model, get_device, set_seed
from medifed.train import evaluate, fit, train_one_epoch


def _synthetic_loader(n: int = 8, seed: int = 0) -> DataLoader:
    """Return a DataLoader with n random 3×224×224 images and fixed labels."""
    torch.manual_seed(seed)
    images = torch.randn(n, 3, 224, 224)
    # Assign labels cycling through all 4 classes so every class is represented.
    labels = torch.tensor([i % NUM_CLASSES for i in range(n)])
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=n, shuffle=False)


class TestBuildModel(unittest.TestCase):

    def test_output_shape(self):
        """TC-MODEL-001: build_model output shape is (batch, 4) for a 224x224 input."""
        model = build_model(pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(tuple(out.shape), (2, NUM_CLASSES))

    def test_final_layer_only_changed(self):
        """TC-MODEL-002: the final layer has exactly 4 outputs and no other layer was changed in shape."""
        model = build_model(pretrained=False)
        # Final FC must have NUM_CLASSES outputs.
        self.assertEqual(model.fc.out_features, NUM_CLASSES)
        # The in_features of the FC must be 512 (ResNet18 standard).
        self.assertEqual(model.fc.in_features, 512)

    def test_get_device_type(self):
        """TC-MODEL-003: get_device returns cuda when available and cpu otherwise."""
        device = get_device()
        if torch.cuda.is_available():
            self.assertEqual(device.type, "cuda")
        else:
            self.assertEqual(device.type, "cpu")

    def test_set_seed_identical_weights(self):
        """TC-MODEL-004: set_seed makes two freshly built models have identical weights."""
        set_seed(42)
        m1 = build_model(pretrained=False)
        set_seed(42)
        m2 = build_model(pretrained=False)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            self.assertTrue(torch.equal(p1, p2))


class TestTrainFunctions(unittest.TestCase):

    def setUp(self):
        # Use CPU for tests — no GPU required, keeps tests fast and portable.
        self.device = torch.device("cpu")

    def test_loss_decreases(self):
        """TC-TRAIN-001: on 8 synthetic images with fixed labels, 30 training steps lower the loss."""
        set_seed(0)
        model = build_model(pretrained=False).to(self.device)
        loader = _synthetic_loader(n=8, seed=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        first_loss, _ = train_one_epoch(model, loader, optimizer, loss_fn, self.device)
        for _ in range(29):
            last_loss, _ = train_one_epoch(model, loader, optimizer, loss_fn, self.device)

        self.assertLess(last_loss, first_loss,
                        f"Loss did not decrease: first={first_loss:.4f}, last={last_loss:.4f}")

    def test_evaluate_ranges_and_counts(self):
        """TC-TRAIN-002: evaluate returns accuracy in [0,1], num_samples equals dataset size, confusion sum equals num_samples."""
        model = build_model(pretrained=False).to(self.device)
        loader = _synthetic_loader(n=8, seed=1)
        metrics = evaluate(model, loader, self.device)

        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["num_samples"], 8)

        cm_sum = sum(metrics["confusion_matrix"][i][j]
                     for i in range(NUM_CLASSES) for j in range(NUM_CLASSES))
        self.assertEqual(cm_sum, metrics["num_samples"])

    def test_per_class_recall_none_for_missing_class(self):
        """TC-TRAIN-003: per_class_recall is None for a class with no true samples."""
        model = build_model(pretrained=False).to(self.device)
        # Only classes 0 and 1 present — classes 2 and 3 should have None recall.
        images = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 0, 1])
        loader = DataLoader(TensorDataset(images, labels), batch_size=4)
        metrics = evaluate(model, loader, self.device)

        self.assertIsNone(metrics["per_class_recall"][2])
        self.assertIsNone(metrics["per_class_recall"][3])
        # Classes 0 and 1 are present so recall must be a float.
        self.assertIsInstance(metrics["per_class_recall"][0], float)
        self.assertIsInstance(metrics["per_class_recall"][1], float)

    def test_checkpoint_save_load_identical_output(self):
        """TC-TRAIN-004: a checkpoint saved with torch.save and loaded into a fresh model gives identical outputs."""
        import tempfile, os
        set_seed(7)
        model = build_model(pretrained=False).to(self.device)
        model.eval()

        x = torch.randn(2, 3, 224, 224).to(self.device)
        with torch.no_grad():
            out_original = model(x)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name
        try:
            torch.save(model.state_dict(), tmp_path)
            model2 = build_model(pretrained=False).to(self.device)
            model2.load_state_dict(torch.load(tmp_path, map_location=self.device, weights_only=True))
            model2.eval()
            with torch.no_grad():
                out_loaded = model2(x)
            self.assertTrue(torch.allclose(out_original, out_loaded),
                            "Outputs differ after save/load round-trip")
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
