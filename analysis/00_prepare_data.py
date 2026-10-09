"""Stage 00 — Prepare a workable random sample of the Avazu CTR dataset.

The raw train.gz is ~1.1 GB / ~40.4M rows. Every later stage reads the small
sample produced here, so the whole pipeline runs in minutes on a laptop.

Usage:
    python analysis/00_prepare_data.py
"""
import gzip
import os

import pandas as pd

from common import (
    DATA_DIR, TRAIN_GZ, TEST_GZ, TRAIN_SAMPLE, TEST_SAMPLE,
    TRAIN_FRAC, TEST_FRAC, SEED, ensure_dirs,
)


def sample_gz(gz_path: str, out_path: str, frac: float, seed: int) -> None:
    """Read a gzipped CSV in chunks and write a random sample to out_path."""
    if os.path.exists(out_path):
        print(f"[skip] {out_path} already exists")
        return
    ensure_dirs()
    print(f"Reading {gz_path} (sampling {frac:.2%}) ...")
    chunks = []
    with gzip.open(gz_path, "rt") as f:
        for chunk in pd.read_csv(f, chunksize=500_000):
            chunks.append(chunk.sample(frac=frac, random_state=seed))
    df = pd.concat(chunks, ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"[saved] {out_path} ({len(df):,} rows)")


def main() -> None:
    sample_gz(TRAIN_GZ, TRAIN_SAMPLE, TRAIN_FRAC, SEED)
    sample_gz(TEST_GZ, TEST_SAMPLE, TEST_FRAC, SEED)
    print("Done. Samples ready in", DATA_DIR)


if __name__ == "__main__":
    main()