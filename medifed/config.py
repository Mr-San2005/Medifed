"""Constants for MediFed. One source of truth for paths, seeds, classes and hospital metadata."""

from pathlib import Path

# Anchor every path to this file so the project is relocatable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR        = PROJECT_ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
CHECKPOINT_DIR  = PROJECT_ROOT / "checkpoints"
RESULTS_DIR     = PROJECT_ROOT / "results"

# Training reproducibility.
SEED = 42

# All images are resized to this square before entering the model.
IMAGE_SIZE = 224

# Per-class cap applied during data loading (keeps GPU memory and training time bounded).
MAX_IMAGES_PER_CLASS = 500

# Global class order — defined once, used identically by every hospital.
# A hospital that lacks a class still uses these global indices.
CLASS_NAMES = ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Folder names that are never treated as class folders, regardless of hospital.
EXCLUDE_FOLDER_NAMES = ["masks", "Lung_Opacity"]

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
