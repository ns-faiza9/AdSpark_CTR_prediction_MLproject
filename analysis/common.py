"""Shared configuration and helpers for the AdSpark analysis pipeline.

Every stage script (01_data_loading ... 08_ensemble) imports from here so
that paths, seeds and output locations stay consistent across the project.
"""
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")  # headless-safe figure rendering
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "Data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# Mirror paths: React/Vite serves from webpage/public/ (dev) and webpage/dist/ (prod)
WEBPAGE_DIR = os.path.join(ROOT_DIR, "webpage")
PUBLIC_DIR = os.path.join(WEBPAGE_DIR, "public")
PUBLIC_FIGURES_DIR = os.path.join(PUBLIC_DIR, "figures")
PUBLIC_DATA_DIR = os.path.join(PUBLIC_DIR, "data")
DIST_DIR = os.path.join(WEBPAGE_DIR, "dist")
DIST_FIGURES_DIR = os.path.join(DIST_DIR, "figures")
DIST_DATA_DIR = os.path.join(DIST_DIR, "data")

# ---------------------------------------------------------------------------
# Dataset files
# ---------------------------------------------------------------------------
TRAIN_GZ = os.path.join(DATA_DIR, "train.gz")
TEST_GZ = os.path.join(DATA_DIR, "test.gz")
TRAIN_SAMPLE = os.path.join(DATA_DIR, "train_sample.csv")
TEST_SAMPLE = os.path.join(DATA_DIR, "test_sample.csv")
PROCESSED_TRAIN = os.path.join(PROCESSED_DIR, "train_processed.csv")
PROCESSED_TEST = os.path.join(PROCESSED_DIR, "test_processed.csv")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
TARGET = "click"
# The full Avazu train set has ~40.4M rows; we sample ~2.5% (~1M rows) so the
# whole pipeline runs quickly on a laptop while staying statistically sound.
TRAIN_FRAC = 0.025
TEST_FRAC = 0.045  # test set has ~4.5M rows -> ~200k sample

# Columns that are high-cardinality identifiers -> frequency encoding
HIGH_CARDINALITY = [
    "site_id", "site_domain", "app_id", "app_domain",
    "device_id", "device_ip", "device_model",
]
# Low-cardinality categoricals -> label encoding
LOW_CARDINALITY = [
    "C1", "banner_pos", "site_category", "app_category",
    "device_type", "device_conn_type",
    "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
]


def ensure_dirs() -> None:
    """Create every directory the pipeline writes into."""
    for d in (PROCESSED_DIR, OUTPUT_DIR, FIGURES_DIR,
              PUBLIC_FIGURES_DIR, PUBLIC_DATA_DIR,
              DIST_FIGURES_DIR, DIST_DATA_DIR):
        os.makedirs(d, exist_ok=True)


def load_sample(path: str = TRAIN_SAMPLE) -> pd.DataFrame:
    """Load a prepared sample CSV."""
    return pd.read_csv(path)


def load_processed(path: str = PROCESSED_TRAIN) -> pd.DataFrame:
    """Load the feature-engineered dataset."""
    return pd.read_csv(path)


def save_json(name: str, data: dict) -> str:
    """Write a JSON summary into analysis/output/ AND mirror to public/ & dist/."""
    ensure_dirs()
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    # Mirror to public/ and dist/
    for dest in (PUBLIC_DATA_DIR, DIST_DATA_DIR):
        shutil.copy2(path, os.path.join(dest, name))
    print(f"[saved] {path} -> copied to public/ & dist/")
    return path


def save_fig(fig, name: str) -> str:
    """Save a matplotlib figure into analysis/output/figures/ AND mirror to public/ & dist/."""
    ensure_dirs()
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    # Mirror to public/ and dist/
    for dest in (PUBLIC_FIGURES_DIR, DIST_FIGURES_DIR):
        shutil.copy2(path, os.path.join(dest, name))
    print(f"[saved] {path} -> copied to public/ & dist/")
    return path


def save_table(df: pd.DataFrame, name: str) -> str:
    """Write a small table as CSV into analysis/output/."""
    ensure_dirs()
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, index=False)
    print(f"[saved] {path}")
    return path