"""Tests for the FedBN personalization helpers."""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from medifed.model import build_model
from medifed.personalization import (
    bn_keys,
    bn_path,
    load_bn_state,
    merge_state,
    prepare_local_bn,
    save_bn_state,
    split_state,
)


class TestFedBNHelpers(unittest.TestCase):
    """Test the BatchNorm-specific state helpers."""

    def test_bn_keys_cover_batchnorm_only(self):
        """TC-FBN-001 bn_keys covers every BatchNorm module and excludes conv/fc keys."""
        model = build_model(pretrained=False)
        keys = bn_keys(model)
        self.assertEqual(len(keys), 20 * 5)
        for key in keys:
            self.assertTrue(key.endswith(("weight", "bias", "running_mean", "running_var", "num_batches_tracked")))
            self.assertNotIn("fc.", key)
            self.assertNotIn("conv1.", key)
            self.assertNotIn("layer1.0.conv1.", key)

    def test_merge_state_keeps_local_bn(self):
        """TC-FBN-002 merge takes BatchNorm entries from the local state and all others from the global state."""
        model = build_model(pretrained=False)
        global_state = model.state_dict()
        keys = bn_keys(model)
        local_bn = {key: tensor.clone() + 1.0 for key, tensor in global_state.items() if key in keys}
        merged = merge_state(global_state, local_bn, keys)
        for key in keys:
            self.assertTrue(torch.allclose(merged[key], local_bn[key]))
        for key in global_state:
            if key not in keys:
                self.assertTrue(torch.equal(merged[key], global_state[key]))

    def test_save_and_load_round_trip(self):
        """TC-FBN-003 save/load round trip for a local BN state."""
        model = build_model(pretrained=False)
        keys = bn_keys(model)
        state = {key: model.state_dict()[key].clone() for key in keys}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bn.pt"
            save_bn_state(path, state)
            loaded = load_bn_state(path)
            for key in keys:
                self.assertTrue(torch.equal(loaded[key], state[key]))

    def test_merge_raises_if_bn_missing(self):
        """TC-FBN-004 merge raises when a BatchNorm key is missing."""
        model = build_model(pretrained=False)
        keys = bn_keys(model)
        missing = {key: value.clone() for key, value in model.state_dict().items() if key in keys[:10]}
        with self.assertRaises(KeyError):
            merge_state(model.state_dict(), missing, keys)

    def test_prepare_local_bn_round1_and_round2(self):
        """TC-FBN-005 round 1 deletes stale file; round 2 loads the state back."""
        run_name = "personalized_seed42"
        hospital_id = 1
        path = bn_path(run_name, hospital_id)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            path = checkpoint_dir / run_name / f"hospital{hospital_id}_bn.pt"
            state = {"layer1.0.bn1.weight": torch.ones(1)}
            path.parent.mkdir(parents=True, exist_ok=True)
            save_bn_state(path, state)
            result = prepare_local_bn(1, path)
            self.assertIsNone(result)
            self.assertFalse(path.exists())

            save_bn_state(path, state)
            loaded = prepare_local_bn(2, path)
            self.assertEqual(set(loaded.keys()), set(state.keys()))
            self.assertTrue(torch.equal(loaded["layer1.0.bn1.weight"], state["layer1.0.bn1.weight"]))

    def test_bn_paths_are_hospital_specific(self):
        """TC-FBN-006 different hospitals yield different BN checkpoint paths."""
        self.assertNotEqual(bn_path("baseline_seed42", 1), bn_path("baseline_seed42", 2))
        self.assertNotEqual(bn_path("baseline_seed42", 1), bn_path("personalized_seed42", 1))

    def test_split_state(self):
        """Split state retains the BN subset separately from the shared state."""
        model = build_model(pretrained=False)
        state = model.state_dict()
        keys = bn_keys(model)
        shared, local_bn = split_state(state, keys)
        self.assertEqual(set(shared.keys()) | set(local_bn.keys()), set(state.keys()))
        self.assertEqual(set(local_bn.keys()), set(keys))


if __name__ == "__main__":
    unittest.main()
