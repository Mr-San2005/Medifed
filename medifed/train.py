"""Local training and evaluation functions for MediFed."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from medifed.config import NUM_CLASSES


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one full pass over the training loader.

    Returns:
        mean_loss: average cross-entropy loss over all batches
        accuracy:  fraction of correctly classified samples
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    mean_loss = total_loss / total if total > 0 else 0.0
    accuracy  = correct / total if total > 0 else 0.0
    return mean_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate the model on a DataLoader without updating weights.

    Returns a dict with:
        loss             – mean cross-entropy loss
        accuracy         – fraction correctly classified
        num_samples      – total number of samples evaluated
        confusion_matrix – NUM_CLASSES × NUM_CLASSES list of lists
                           rows = true class, columns = predicted class
        per_class_recall – list of NUM_CLASSES values; None for classes with
                           no true samples in this loader
    """
    loss_fn = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    total = 0
    # confusion[true][pred] counts
    confusion = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * images.size(0)
            total += images.size(0)

            preds = logits.argmax(dim=1)
            for true, pred in zip(labels.tolist(), preds.tolist()):
                confusion[true][pred] += 1

    mean_loss = total_loss / total if total > 0 else 0.0
    correct   = sum(confusion[i][i] for i in range(NUM_CLASSES))
    accuracy  = correct / total if total > 0 else 0.0

    # Per-class recall: TP / (TP + FN).  None when the class has no true samples.
    per_class_recall = []
    for i in range(NUM_CLASSES):
        true_count = sum(confusion[i])
        if true_count == 0:
            per_class_recall.append(None)
        else:
            per_class_recall.append(confusion[i][i] / true_count)

    return {
        "loss":             mean_loss,
        "accuracy":         accuracy,
        "num_samples":      total,
        "confusion_matrix": confusion,
        "per_class_recall": per_class_recall,
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> list[dict]:
    """Train for `epochs` epochs and return per-epoch metrics.

    Uses Adam optimiser.  Returns a list of dicts, one per epoch, each with:
        epoch, train_loss, train_acc, val_loss, val_acc
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss()
    history   = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, device)
        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "val_loss":   val_metrics["loss"],
            "val_acc":    val_metrics["accuracy"],
        })

    return history
