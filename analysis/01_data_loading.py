"""Stage 01 — Data Loading: load, inspect and preview the Avazu CTR dataset.

Produces:
    analysis/output/01_data_loading_summary.json
    analysis/output/01_head.csv, 01_describe.csv, 01_dtypes.csv, 01_missing.csv

Usage:
    python analysis/01_data_loading.py
"""
import pandas as pd

from common import load_sample, save_json, save_table, TARGET


def main() -> None:
    df = load_sample()
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns")

    # --- 1. Overview -------------------------------------------------------
    overview = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "click_rate": round(float(df[TARGET].mean()), 6),
        "positive_clicks": int(df[TARGET].sum()),
    }

    # --- 2. First rows -----------------------------------------------------
    head = df.head(10)
    save_table(head, "01_head.csv")

    # --- 3. Data types -----------------------------------------------------
    dtypes = df.dtypes.astype(str).rename("dtype").reset_index().rename(columns={"index": "column"})
    save_table(dtypes, "01_dtypes.csv")

    # --- 4. Descriptive statistics -----------------------------------------
    describe = df.describe(include="all").transpose().reset_index().rename(columns={"index": "column"})
    save_table(describe, "01_describe.csv")

    # --- 5. Missing values -------------------------------------------------
    missing = df.isna().sum().rename("missing").reset_index().rename(columns={"index": "column"})
    missing["missing_pct"] = (missing["missing"] / len(df) * 100).round(4)
    save_table(missing, "01_missing.csv")
    overview["missing_total"] = int(df.isna().sum().sum())

    # --- 6. Target balance -------------------------------------------------
    target_counts = df[TARGET].value_counts().to_dict()
    overview["target_counts"] = {str(k): int(v) for k, v in target_counts.items()}

    save_json("01_data_loading_summary.json", overview)

    print("\n--- Overview ---")
    for k, v in overview.items():
        print(f"  {k}: {v}")
    print("\n--- First 5 rows ---")
    print(head.head().to_string())


if __name__ == "__main__":
    main()