"""Constants for MediFed. One source of truth for paths, seeds, classes and hospital metadata."""

import os
from pathlib import Path


def resolve_project_root(env_value, fallback):
    """Return Path(env_value) when it is a non-empty string, otherwise fallback.

    Client processes started by Flower run from a copy of the app under the user
    profile, so they cannot locate the project by this file's location. The
    launcher sets MEDIFED_ROOT; local scripts and tests leave it unset and use
    the folder that contains medifed/.
    """
    if isinstance(env_value, str) and env_value != "":
        return Path(env_value)
    return fallback


# Anchor every path to this file so the project is relocatable when MEDIFED_ROOT is unset.
PROJECT_ROOT = resolve_project_root(
    os.environ.get("MEDIFED_ROOT"),
    Path(__file__).resolve().parent.parent,
)

DATA_DIR        = PROJECT_ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
CHECKPOINT_DIR  = PROJECT_ROOT / "checkpoints"
RESULTS_DIR     = PROJECT_ROOT / "results"
RUNS_DIR        = RESULTS_DIR / "runs"

# Training reproducibility.
SEED = 42

# All images are resized to this square before entering the model.
IMAGE_SIZE = 224

# Per-class cap applied during data loading (keeps GPU memory and training time bounded).
MAX_IMAGES_PER_CLASS = 500

# Fraction of each class held out for validation.
VAL_FRACTION = 0.2

# Only files with these extensions are treated as images.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Global class order — defined once, used identically by every hospital.
# A hospital that lacks a class still uses these global indices.
CLASS_NAMES = ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Derived from CLASS_NAMES so the model head always matches the class registry.
NUM_CLASSES = len(CLASS_NAMES)

# Folder names that are never treated as class folders, regardless of hospital.
# Checked case-insensitively at walk time; __MACOSX is also excluded.
EXCLUDE_FOLDER_NAMES = ["masks", "Lung_Opacity", "__MACOSX"]

# Per-hospital raw data directories.
HOSPITAL_RAW_DIRS = {
    1: RAW_DIR / "hospital1",
    2: RAW_DIR / "hospital2",
    3: RAW_DIR / "hospital3",
}

# Maps lowercase folder name -> global class name, per hospital.
# Used by find_images to assign a class to each image without relying on ImageFolder.
FOLDER_TO_CLASS = {
    1: {
        "normal":    "NORMAL",
        "pneumonia": "PNEUMONIA",
    },
    2: {
        "normal":       "NORMAL",
        "tuberculosis": "TUBERCULOSIS",
    },
    3: {
        "covid19":   "COVID19",
        "normal":    "NORMAL",
        "pneumonia": "PNEUMONIA",
    },
}

# Hospital registry: display name, Kaggle dataset slug, and the classes present locally.
HOSPITALS = {
    1: {
        "name":    "Hospital 1 (Guangzhou, China)",
        "slug":    "paultimothymooney/chest-xray-pneumonia",
        "classes": ["NORMAL", "PNEUMONIA"],
    },
    2: {
        "name":    "Hospital 2 (Qatar Univ + Dhaka)",
        "slug":    "tawsifurrahman/tuberculosis-tb-chest-xray-dataset",
        "classes": ["NORMAL", "TUBERCULOSIS"],
    },
    3: {
        "name":    "Hospital 3 (Qatar Univ et al.)",
        "slug":    "tawsifurrahman/covid19-radiography-database",
        "classes": ["COVID19", "NORMAL", "PNEUMONIA"],
    },
}

# Local training hyperparameters.
# Batch size of 16 keeps peak GPU memory well under 4 GB at 224×224.
BATCH_SIZE = 16
# 5 epochs is enough for a sanity check that the model learns from each hospital's data.
LOCAL_EPOCHS = 5
# 1e-4 is a conservative learning rate for fine-tuning a pretrained backbone.
LEARNING_RATE = 1e-4

# Federated baseline (plain FedAvg) hyperparameters.
FL_NUM_HOSPITALS = 3
FL_NUM_ROUNDS = 10
# One local epoch per round keeps each Flower round short on this machine.
FL_LOCAL_EPOCHS = 1
DEFAULT_RUN_NAME = "baseline_seed42"
