"""Tests for MediFed Grad-CAM heatmaps and overlays."""

import unittest

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from medifed.gradcam import colormap, gradcam, overlay


class TinyCamModel(nn.Module):
    def __init__(self):
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
        features = self.layer4(self.conv1(image))
        return self.fc(self.pool(features).flatten(1))


class TestGradCam(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.model = TinyCamModel().eval()
        self.image = torch.rand(1, 3, 32, 32)

    def test_cam_shape_and_range(self):
        """TC-CAM-001 Grad-CAM output matches input size and stays in [0, 1]."""
        cam, probabilities, target = gradcam(self.model, self.image)
        self.assertEqual(cam.shape, (32, 32))
        self.assertGreaterEqual(float(cam.min()), 0.0)
        self.assertLessEqual(float(cam.max()), 1.0)
        self.assertEqual(len(probabilities), 4)
        self.assertIn(target, range(4))

    def test_hooks_are_removed_after_computation(self):
        """TC-CAM-002 Grad-CAM removes its forward hook after computation."""
        target_layer = self.model.layer4[-1]
        gradcam(self.model, self.image)
        self.assertEqual(len(target_layer._forward_hooks), 0)
        self.assertEqual(len(target_layer._backward_hooks), 0)

    def test_target_classes_can_produce_different_maps(self):
        """TC-CAM-003 distinct target classes produce class-specific maps."""
        first, _, _ = gradcam(self.model, self.image, target_class=0)
        second, _, _ = gradcam(self.model, self.image, target_class=1)
        self.assertFalse(np.allclose(first, second))

    def test_constant_activation_map_becomes_zeros(self):
        """TC-CAM-004 constant activation maps become finite zeros."""
        for parameter in self.model.parameters():
            nn.init.zeros_(parameter)
        cam, _, _ = gradcam(self.model, self.image)
        self.assertTrue(np.isfinite(cam).all())
        self.assertTrue(np.array_equal(cam, np.zeros_like(cam)))

    def test_overlay_is_rgb_and_keeps_image_size(self):
        """TC-CAM-005 overlay returns an RGB image at the original dimensions."""
        result = overlay(Image.new("L", (27, 19), 120), np.ones((32, 32), dtype=np.float32))
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (27, 19))

    def test_colormap_has_three_channels_and_distinct_endpoints(self):
        """TC-CAM-006 colormap output is RGB and maps cool and warm endpoints differently."""
        colors = colormap(np.array([[0.0, 1.0]], dtype=np.float32))
        self.assertEqual(colors.shape, (1, 2, 3))
        self.assertFalse(np.array_equal(colors[0, 0], colors[0, 1]))


if __name__ == "__main__":
    unittest.main()