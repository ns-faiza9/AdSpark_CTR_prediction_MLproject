"""Stage 03 — Feature Engineering / Preprocessing.

Transforms the raw sample into a model-ready dataset:
  * parse `hour` -> hour_of_day + day_of_week
  * frequency-encode high-cardinality identifiers (site_id, device_id, ...)
  * label-encode low-cardinality categoricals
  * drop id / target / raw text columns

Produces:
    Data/processed/train_processed.csv, test_processed.csv
    analysis/output/03_feature_engineering_summary.json
    analysis/output/figures/fe_*.png

Usage:
    python analysis/03_feature_engineering.py
"""
import matplotlib.pyplot as plt
import pandas as pd

from common import (
    load_sample, save_fig, save_json, save_table,
    TRAIN_SAMPLE, TEST_SAMPLE, PROCESSED_TRAIN, PROCESSED_TEST,
    HIGH_CARDINALITY, LOW_CARDINALITY, TARGET, SEED,
)

DROP_COLS = ["id", "hour", "device_id", "device_ip"]  # raw identifiers


def parse_hour(df: pd.DataFrame) -> pd.DataFrame:
    """hour is YYMMDDHH (e.g. 14102100 = 2014-10-21 00:00)."""
    df = df.copy()
    df["hour_of_day"] = df["hour"] % 100
    df["day_of_week"] = pd.to_datetime(
        df["hour"].astype(str).str.zfill(8), format="%y%m%d%H"
    ).dt.dayofweek
    return df


def frequency_encode(train: pd.DataFrame, test: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Replace each high-cardinality column with its frequency in train."""
    for col in cols:
        freq = train[col].value_counts(normalize=True)
        train[f"{col}_freq"] = train[col].map(freq).fillna(0.0)
        test[f"{col}_freq"] = test[col].map(freq).fillna(0.0)
        train = train.drop(columns=[col])
        test = test.drop(columns=[col])
    return train, test


def label_encode(train: pd.DataFrame, test: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Map each low-cardinality column to integer codes fit on train."""
    for col in cols:
        codes = {v: i for i, v in enumerate(train[col].astype(str).unique())}
        train[f"{col}_enc"] = train[col].astype(str).map(codes)
        test[f"{col}_enc"] = test[col].astype(str).map(codes).fillna(-1)
        train = train.drop(columns=[col])
        test = test.drop(columns=[col])
    return train, test


def main() -> None:
    train = load_sample(TRAIN_SAMPLE)
    test = load_sample(TEST_SAMPLE)
    print(f"Train sample: {train.shape}, Test sample: {test.shape}")

    # 1. Parse hour
    train = parse_hour(train)
    test = parse_hour(test)

    # 2. Frequency encoding for high-cardinality columns
    train, test = frequency_encode(train, test, HIGH_CARDINALITY)

    # 3. Label encoding for low-cardinality columns
    train, test = label_encode(train, test, LOW_CARDINALITY)

    # 4. Drop raw identifiers
    train = train.drop(columns=[c for c in DROP_COLS if c in train.columns])
    test = test.drop(columns=[c for c in DROP_COLS if c in test.columns])

    # 5. Align columns (test may lack the target)
    feature_cols = [c for c in train.columns if c != TARGET]
    test = test.reindex(columns=feature_cols)

    train.to_csv(PROCESSED_TRAIN, index=False)
    test.to_csv(PROCESSED_TEST, index=False)
    print(f"[saved] {PROCESSED_TRAIN} ({train.shape})")
    print(f"[saved] {PROCESSED_TEST} ({test.shape})")

    # Summary
    summary = {
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "high_cardinality_encoded": HIGH_CARDINALITY,
        "low_cardinality_encoded": LOW_CARDINALITY,
        "dropped": DROP_COLS,
    }
    save_json("03_feature_engineering_summary.json", summary)
    save_table(pd.DataFrame({"feature": feature_cols}), "03_features.csv")

    # Figures: frequency distributions of the new encodings
    freq_cols = [c for c in train.columns if c.endswith("_freq")]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, col in zip(axes.ravel(), freq_cols):
        ax.hist(train[col], bins=50, color="#0ea5e9")
        ax.set_title(col)
        ax.set_yscale("log")
    fig.suptitle("Frequency-encoded feature distributions (train)", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "fe_01_frequency_encodings.png")

    fig, ax = plt.subplots()
    ax.hist(train["hour_of_day"], bins=24, color="#10b981", edgecolor="white")
    ax.set_title("Impressions by hour of day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Count")
    save_fig(fig, "fe_02_hour_of_day.png")

    print("\nFeature engineering complete.")


if __name__ == "__main__":
    main()