"""FedBN helper functions for keeping BatchNorm state local to each hospital."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from medifed.config import CHECKPOINT_DIR


def bn_keys(model: nn.Module) -> list[str]:
    """Return sorted BN state_dict keys for every BatchNorm module in the model."""
    keys: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue
        prefix = name
        for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            key = f"{prefix}.{suffix}" if prefix else suffix
            if key in model.state_dict():
                keys.append(key)
    return sorted(keys)


def split_state(state_dict: dict[str, torch.Tensor], keys: list[str]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split a state_dict into the shared part and the local BatchNorm part."""
    shared: dict[str, torch.Tensor] = {}
    local_bn: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key in keys:
            local_bn[key] = value
        else:
            shared[key] = value
    return shared, local_bn


def merge_state(global_state: dict[str, torch.Tensor], local_bn: dict[str, torch.Tensor], keys: list[str]) -> dict[str, torch.Tensor]:
    """Create a merged state dict that keeps local BN entries but uses the global state elsewhere."""
    merged = dict(global_state)
    missing = [key for key in keys if key not in local_bn]
    if missing:
        raise KeyError(f"Missing local BN keys: {missing[:3]}" )
    for key in keys:
        merged[key] = local_bn[key]
    return merged


def bn_path(run_name: str, hospital_id: int) -> Path:
    """Return the path for a hospital's local BN checkpoint."""
    return CHECKPOINT_DIR / run_name / f"hospital{hospital_id}_bn.pt"


def save_bn_state(path: str | Path, state_dict: dict[str, torch.Tensor]) -> Path:
    """Write a local BN checkpoint to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)
    return path


def load_bn_state(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a local BN checkpoint from disk."""
    return torch.load(Path(path), map_location="cpu")


def prepare_local_bn(server_round: int, path: str | Path) -> dict[str, torch.Tensor] | None:
    """Delete a stale local BN file on round 1; otherwise load the round's saved BN state."""
    path = Path(path)
    if server_round == 1:
        if path.exists():
            path.unlink()
        return None
    if path.exists():
        return load_bn_state(path)
    return None
