"""Checkpoint-backed prediction variants shared by the demo and local app."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image, UnidentifiedImageError

from medifed.config import CHECKPOINT_DIR, CLASS_NAMES
from medifed.data import get_transforms
from medifed.gradcam import gradcam, overlay
from medifed.model import build_model, get_device
from medifed.personalization import bn_keys, load_bn_state, merge_state


def list_variants(checkpoint_dir: str | Path = CHECKPOINT_DIR) -> list[str]:
    """Return only prediction variants whose required checkpoints exist."""
    checkpoint_dir = Path(checkpoint_dir)
    available: list[str] = []
    if (checkpoint_dir / "baseline_seed42_global.pt").is_file():
        available.append("fedavg_global")
    personalized = checkpoint_dir / "personalized_seed42_global.pt"
    for hospital_id in (1, 2, 3):
        if personalized.is_file() and (
            checkpoint_dir / "personalized_seed42" / f"hospital{hospital_id}_bn.pt"
        ).is_file():
            available.append(f"fedbn_hospital{hospital_id}")
    for hospital_id in (1, 2, 3):
        if (checkpoint_dir / f"local_hospital{hospital_id}.pt").is_file():
            available.append(f"local_hospital{hospital_id}")
    return available


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def _hospital_id(name: str, prefix: str) -> int:
    suffix = name.removeprefix(prefix)
    if suffix not in {"1", "2", "3"}:
        raise ValueError(f"Unknown variant: {name}")
    return int(suffix)


@lru_cache(maxsize=16)
def _load_variant_cached(name: str, checkpoint_dir: str, device_name: str) -> torch.nn.Module:
    root = Path(checkpoint_dir)
    model = build_model(pretrained=False)

    if name == "fedavg_global":
        state = _load_state(root / "baseline_seed42_global.pt")
    elif name.startswith("fedbn_hospital"):
        hospital_id = _hospital_id(name, "fedbn_hospital")
        global_path = root / "personalized_seed42_global.pt"
        bn_path = root / "personalized_seed42" / f"hospital{hospital_id}_bn.pt"
        state = _load_state(global_path)
        local_bn = load_bn_state(bn_path) if bn_path.is_file() else _load_state(bn_path)
        state = merge_state(state, local_bn, bn_keys(model))
    elif name.startswith("local_hospital"):
        hospital_id = _hospital_id(name, "local_hospital")
        state = _load_state(root / f"local_hospital{hospital_id}.pt")
    else:
        raise ValueError(f"Unknown variant: {name}")

    model.load_state_dict(state)
    model.to(torch.device(device_name))
    model.eval()
    return model


def load_variant(
    name: str,
    checkpoint_dir: str | Path = CHECKPOINT_DIR,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Load and cache one named model variant from a checkpoint directory."""
    if name != "fedavg_global" and not name.startswith(("fedbn_hospital", "local_hospital")):
        raise ValueError(f"Unknown variant: {name}")
    selected_device = device or get_device()
    return _load_variant_cached(name, str(Path(checkpoint_dir).resolve()), str(selected_device))


def predict(
    variant: str,
    image_bytes: bytes,
    checkpoint_dir: str | Path = CHECKPOINT_DIR,
    device: torch.device | None = None,
) -> dict:
    """Predict an in-memory image and return ranked classes and PNG previews."""
    try:
        with Image.open(BytesIO(image_bytes)) as opened:
            image = opened.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("not a valid image") from error

    image_tensor = get_transforms(train=False)(image).unsqueeze(0)
    selected_device = device or get_device()
    model = load_variant(variant, checkpoint_dir, selected_device)
    started = perf_counter()
    cam, probabilities, target = gradcam(model, image_tensor, device=selected_device)
    elapsed_ms = (perf_counter() - started) * 1000.0

    resized = image.resize((224, 224), Image.Resampling.BILINEAR)
    heatmap = overlay(resized, cam, alpha=0.5)
    ranked = sorted(
        (
            {"class": CLASS_NAMES[index], "probability": probabilities[index]}
            for index in range(len(CLASS_NAMES))
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )
    original_buffer = BytesIO()
    heatmap_buffer = BytesIO()
    resized.save(original_buffer, format="PNG")
    heatmap.save(heatmap_buffer, format="PNG")
    return {
        "ranked": ranked,
        "top_class": CLASS_NAMES[target],
        "heatmap_png": heatmap_buffer.getvalue(),
        "original_png": original_buffer.getvalue(),
        "milliseconds": elapsed_ms,
    }