"""Grad-CAM heatmaps and image overlays for MediFed predictions."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


def gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int | None = None,
    device: torch.device | None = None,
) -> tuple[np.ndarray, list[float], int]:
    """Return a normalized Grad-CAM map, class probabilities, and target index."""
    layer = model.layer4[-1]
    captured: dict[str, torch.Tensor] = {}

    def capture_activation(_module, _inputs, output):
        captured["activation"] = output
        output.register_hook(lambda gradient: captured.__setitem__("gradient", gradient))

    handle = layer.register_forward_hook(capture_activation)
    model.zero_grad(set_to_none=True)
    try:
        if device is not None:
            image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        selected_class = int(logits.argmax(dim=1).item()) if target_class is None else target_class
        if selected_class < 0 or selected_class >= logits.shape[1]:
            raise ValueError(f"target_class must be between 0 and {logits.shape[1] - 1}")
        probabilities = torch.softmax(logits.detach(), dim=1)[0].cpu().tolist()
        logits[0, selected_class].backward()

        activation = captured["activation"]
        gradient = captured["gradient"]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        minimum = float(cam.min())
        maximum = float(cam.max())
        if maximum - minimum <= 1e-12:
            cam = np.zeros_like(cam, dtype=np.float32)
        else:
            cam = ((cam - minimum) / (maximum - minimum)).astype(np.float32)
        return cam, [float(value) for value in probabilities], selected_class
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def colormap(values: np.ndarray) -> np.ndarray:
    """Map normalized activation values from blue through cyan/yellow to red."""
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    stops = np.array(
        [[0, 0, 255], [0, 255, 255], [255, 255, 0], [255, 0, 0]],
        dtype=np.float32,
    )
    positions = values * (len(stops) - 1)
    lower = np.floor(positions).astype(np.int32)
    upper = np.minimum(lower + 1, len(stops) - 1)
    fraction = (positions - lower)[..., None]
    colors = stops[lower] * (1.0 - fraction) + stops[upper] * fraction
    return np.rint(colors).astype(np.uint8)


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a blue-to-red activation map over an RGB image."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    rgb = image.convert("RGB")
    if cam.shape != (rgb.height, rgb.width):
        cam_image = Image.fromarray(np.rint(np.clip(cam, 0, 1) * 255).astype(np.uint8))
        cam_image = cam_image.resize(rgb.size, Image.Resampling.BILINEAR)
        cam = np.asarray(cam_image, dtype=np.float32) / 255.0
    heatmap = Image.fromarray(colormap(cam), mode="RGB")
    return Image.blend(rgb, heatmap, alpha).convert("RGB")