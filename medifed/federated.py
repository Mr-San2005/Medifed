"""Plain federated helpers with no Flower imports.

These functions are used by the client app and by unit tests. Keeping Flower
out of this module means the helpers can be tested without the deployment runtime.
"""

import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from medifed.train import train_one_epoch

ROUND_CSV_FIELDS = ["round", "hospital", "split", "loss", "accuracy", "num_samples"]


def partition_to_hospital(partition_id: int) -> int:
    """Map Flower partition-id (0, 1, 2) to hospital id (1, 2, 3)."""
    mapping = {0: 1, 1: 2, 2: 3}
    if partition_id not in mapping:
        raise ValueError(
            f"partition_id must be 0, 1 or 2, got {partition_id!r}"
        )
    return mapping[partition_id]


def validate_mode(mode: str, allowed: tuple[str, ...]) -> str:
    """Return mode when it is allowed, otherwise list the accepted values."""
    if mode not in allowed:
        raise ValueError(f"mode must be one of: {', '.join(allowed)}")
    return mode


def reset_round_csv_for_first_round(csv_path: Path, server_round: int) -> None:
    """Remove an earlier run's rows when a new run reaches its first round."""
    if server_round == 1:
        Path(csv_path).unlink(missing_ok=True)


def local_update(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> dict:
    """Train `epochs` local epochs with a fresh Adam optimizer.

    Returns the last epoch's train_loss and train_acc. A new optimizer is
    created each call so federated rounds do not carry Adam state across rounds.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loss = 0.0
    train_acc = 0.0
    for _ in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, loader, optimizer, loss_fn, device
        )
    return {"train_loss": train_loss, "train_acc": train_acc}


def state_dict_to_cpu(model: nn.Module) -> dict:
    """Return a state_dict whose tensors all live on CPU.

    Flower serialises arrays independently of the client GPU, so weights must
    leave the device before they are packed into an ArrayRecord.
    """
    return {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}


def append_round_row(csv_path: Path, row_dict: dict) -> None:
    """Append one metrics row, writing the header only when the file is new.

    Each hospital writes its own CSV so concurrent client processes never share
    a file.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROUND_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row_dict[field] for field in ROUND_CSV_FIELDS})
