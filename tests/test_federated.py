"""Unit tests for federated helpers and app imports. CPU only; no Flower runtime."""

import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from medifed.config import NUM_CLASSES, resolve_project_root
from medifed.federated import (
    append_round_row,
    local_update,
    partition_to_hospital,
    state_dict_to_cpu,
)
from medifed.model import build_model


class TestResolveProjectRoot(unittest.TestCase):

    def test_env_path_when_set(self):
        """TC-FED-001 resolve_project_root returns the environment path when it is set"""
        fallback = Path("fallback-dir")
        result = resolve_project_root("E:/custom/root", fallback)
        self.assertEqual(result, Path("E:/custom/root"))

    def test_fallback_when_missing(self):
        """TC-FED-002 resolve_project_root returns the fallback when the value is None or empty"""
        fallback = Path("fallback-dir")
        self.assertEqual(resolve_project_root(None, fallback), fallback)
        self.assertEqual(resolve_project_root("", fallback), fallback)


class TestPartitionToHospital(unittest.TestCase):

    def test_mapping_and_errors(self):
        """TC-FED-003 partition_to_hospital maps 0,1,2 to 1,2,3 and raises ValueError for 3 and -1"""
        self.assertEqual(partition_to_hospital(0), 1)
        self.assertEqual(partition_to_hospital(1), 2)
        self.assertEqual(partition_to_hospital(2), 3)
        with self.assertRaises(ValueError):
            partition_to_hospital(3)
        with self.assertRaises(ValueError):
            partition_to_hospital(-1)


class TestLocalUpdate(unittest.TestCase):

    def test_state_dict_shape_and_finite_loss(self):
        """TC-FED-004 after local_update on 8 synthetic images with a pretrained=False model, the state_dict keys and tensor shapes are identical to those before, and the returned train_loss is a finite number"""
        device = torch.device("cpu")
        model = build_model(pretrained=False).to(device)
        before = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        images = torch.randn(8, 3, 224, 224)
        labels = torch.tensor([i % NUM_CLASSES for i in range(8)])
        loader = DataLoader(TensorDataset(images, labels), batch_size=4)
        metrics = local_update(model, loader, epochs=1, lr=1e-4, device=device)
        after = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        self.assertEqual(list(before.keys()), list(after.keys()))
        self.assertEqual(before, after)
        self.assertTrue(math.isfinite(metrics["train_loss"]))


class TestAppendRoundRow(unittest.TestCase):

    def test_header_once_then_append(self):
        """TC-FED-005 append_round_row writes the header exactly once and appends rows (use a temp folder)"""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "nested" / "rounds.csv"
            row1 = {
                "round": 1,
                "hospital": 1,
                "split": "train",
                "loss": 0.5,
                "accuracy": 0.8,
                "num_samples": 8,
            }
            row2 = {
                "round": 1,
                "hospital": 1,
                "split": "val",
                "loss": 0.4,
                "accuracy": 0.9,
                "num_samples": 4,
            }
            append_round_row(csv_path, row1)
            append_round_row(csv_path, row2)
            lines = csv_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "round,hospital,split,loss,accuracy,num_samples")
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines.count("round,hospital,split,loss,accuracy,num_samples"), 1)


class TestAppImports(unittest.TestCase):

    def test_client_and_server_expose_app(self):
        """TC-FED-006 medifed.client_app and medifed.server_app import successfully and each exposes an object named app"""
        import medifed.client_app as client_app
        import medifed.server_app as server_app

        self.assertTrue(hasattr(client_app, "app"))
        self.assertTrue(hasattr(server_app, "app"))
        self.assertIsNotNone(client_app.app)
        self.assertIsNotNone(server_app.app)


class TestStateDictToCpu(unittest.TestCase):

    def test_all_tensors_on_cpu(self):
        """TC-FED-007 state_dict_to_cpu returns tensors that all live on cpu"""
        model = build_model(pretrained=False)
        cpu_sd = state_dict_to_cpu(model)
        for key, tensor in cpu_sd.items():
            self.assertEqual(
                tensor.device.type,
                "cpu",
                f"{key} is on {tensor.device}, expected cpu",
            )


if __name__ == "__main__":
    unittest.main()
