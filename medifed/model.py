"""ResNet18 model construction and device utilities for MediFed."""

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from medifed.config import NUM_CLASSES


def build_model(pretrained: bool = True) -> nn.Module:
    """Return a ResNet18 with the final FC layer replaced for NUM_CLASSES outputs.

    Using the pretrained ImageNet backbone gives a strong feature extractor for
    chest X-rays without requiring a large local dataset per hospital.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    # Replace only the final layer; all other layers keep their pretrained weights.
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def get_device() -> torch.device:
    """Return the requested device, or choose CUDA when available."""
    requested_device = os.environ.get("MEDIFED_DEVICE")
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("MEDIFED_DEVICE is 'cuda' but CUDA is not available")
        return torch.device("cuda")
    if requested_device is not None:
        raise ValueError("MEDIFED_DEVICE must be 'cpu' or 'cuda'")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed Python random, NumPy and PyTorch (CPU and CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
