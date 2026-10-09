"""Stage 02 — Exploratory Data Analysis.

Produces figures in analysis/output/figures/ and a JSON summary.
All outputs are also mirrored to webpage/public/ for the React frontend.

Analysis covered:
  - Dataset overview (shape, column names, missing values, duplicate rows)
  - Target distribution (uni-variate)
  - Numeric & categorical feature distributions
  - Outlier detection (IQR boxplots)
  - Correlation heatmaps & ranked target correlation
  - Relationship scatter plots & count plots
  - Click rate by context (device type, connection type, site/app category)
  - Temporal patterns (click rate by hour & day of week)
  - Anonymised feature profiles (C14–C21)
  - Pairplot analysis (scatter matrix)

Usage:
    python analysis/02_eda.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import load_sample, save_fig, save_json, save_table, TARGET, SEED

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)


def main() -> None:
    df = load_sample()
    n = len(df)
    summary = {}

    # ------------------------------------------------------------------
    # Task 1 — Dataset overview
    # ------------------------------------------------------------------
    print("TASK 1: Dataset overview")
    summary["rows"] = int(n)
    summary["columns"] = int(df.shape[1])
    summary["column_names"] = list(df.columns)

    # ------------------------------------------------------------------
    # Task 2 — Basic info: dtypes + describe
    # ------------------------------------------------------------------
    print("TASK 2: Basic info / dtypes / describe")
    save_table(pd.concat([df.head(5), df.tail(5)]), "02_head_tail.csv")
    save_table(df.describe().transpose().reset_index(), "02_describe.csv")
    dtype_counts = df.dtypes.astype(str).value_counts().to_dict()
    summary["dtype_counts"] = {str(k): int(v) for k, v in dtype_counts.items()}

    # ------------------------------------------------------------------
    # Task 3 — Missing values
    # ------------------------------------------------------------------
    print("TASK 3: Missing values")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    summary["missing_columns"] = {str(k): int(v) for k, v in missing.items()}
    summary["missing_total"] = int(df.isna().sum().sum())

    # ------------------------------------------------------------------
    # Task 4 — Duplicate rows
    # ------------------------------------------------------------------
    print("TASK 4: Duplicate rows")
    dup = int(df.duplicated().sum())
    summary["duplicate_rows"] = dup
    print(f"  Duplicate rows: {dup}")

    # ------------------------------------------------------------------
    # Task 5 — Target distribution
    # ------------------------------------------------------------------
    print("TASK 5: Target distribution")
    click_rate = float(df[TARGET].mean())
    summary["click_rate"] = round(click_rate, 6)
    summary["positive_clicks"] = int(df[TARGET].sum())
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots()
    bars = ax.bar(["No click (0)", "Click (1)"], counts.values,
                  color=["#94a3b8", "#6366f1"], width=0.5)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + n * 0.005,
                f"{v:,}", ha="center", va="bottom", fontsize=10)
    ax.set_title(f"Target distribution — click rate {click_rate:.4%}")
    ax.set_ylabel("Impressions")
    save_fig(fig, "eda_01_target_distribution.png")

    # ------------------------------------------------------------------
    # Task 6 — Numeric feature distributions (histograms)
    # ------------------------------------------------------------------
    print("TASK 6: Numeric feature distributions")
    num_cols = ["C1", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "banner_pos"]
    fig, axes = plt.subplots(5, 2, figsize=(14, 16))
    for ax, col in zip(axes.ravel(), num_cols):
        vals = df[col].value_counts().sort_index()
        ax.bar(vals.index.astype(str), vals.values, color="#0ea5e9")
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig.suptitle("Numeric categorical feature distributions", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "eda_10_numeric_distributions.png")

    # ------------------------------------------------------------------
    # Task 7 — Outlier detection (boxplots)  [NEW]
    # ------------------------------------------------------------------
    print("TASK 7: Outlier detection — boxplots")
    box_cols = ["banner_pos", "C14", "C15", "C17", "C21"]
    fig, axes = plt.subplots(1, len(box_cols), figsize=(16, 5))
    for ax, col in zip(axes, box_cols):
        ax.boxplot(df[col].dropna(), patch_artist=True,
                   boxprops=dict(facecolor="#6366f120", color="#6366f1"),
                   medianprops=dict(color="#ef4444", linewidth=2),
                   whiskerprops=dict(color="#64748b"),
                   capprops=dict(color="#64748b"),
                   flierprops=dict(marker=".", markersize=2, alpha=0.3,
                                  markerfacecolor="#94a3b8"))
        ax.set_title(col)
        ax.set_ylabel("Value")
    fig.suptitle("Outlier detection — boxplots for key numeric features", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "eda_13_outlier_boxplots.png")

    # ------------------------------------------------------------------
    # Task 8 — Correlation analysis: heatmap + ranked bar chart  [EXPANDED]
    # ------------------------------------------------------------------
    print("TASK 8: Correlation analysis")
    # Parse hour for use in correlations
    df["hour_of_day"] = df["hour"] % 100
    corr_cols = num_cols + ["hour_of_day", TARGET]
    corr = df[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Correlation heatmap of numeric features")
    save_fig(fig, "eda_11_correlation_heatmap.png")

    # Ranked bar: |correlation| with click target
    target_corr = corr[TARGET].drop(TARGET).abs().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#6366f1" if v > 0.05 else "#94a3b8" for v in target_corr.values]
    ax.barh(target_corr.index, target_corr.values, color=colors)
    ax.set_xlabel("|Pearson r| with click")
    ax.set_title("Feature correlation with target (click)")
    ax.axvline(0.05, color="#ef4444", ls="--", alpha=0.7, label="r = 0.05")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "eda_15_correlation_with_target.png")
    summary["corr_with_target"] = target_corr.round(4).to_dict()

    # ------------------------------------------------------------------
    # Task 9 — Relationship plots (scatter / regression)  [NEW]
    # ------------------------------------------------------------------
    print("TASK 9: Relationship scatter plots")
    # Sample 5k rows for scatter plots (speed + readability)
    scatter_df = df.sample(n=5000, random_state=SEED)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: banner_pos vs click (jittered because banner_pos is discrete)
    ax = axes[0]
    jitter = np.random.default_rng(SEED).uniform(-0.2, 0.2, len(scatter_df))
    ax.scatter(scatter_df["banner_pos"] + jitter,
               scatter_df[TARGET] + np.random.default_rng(SEED + 1).uniform(-0.05, 0.05, len(scatter_df)),
               alpha=0.15, s=10, color="#6366f1")
    # Overlay mean click rate per position
    cr_bp = df.groupby("banner_pos")[TARGET].mean()
    ax.plot(cr_bp.index, cr_bp.values, "o-", color="#ef4444", lw=2,
            label="Mean click rate")
    ax.set_xlabel("banner_pos")
    ax.set_ylabel("click")
    ax.set_title("Banner position vs Click (with mean CTR)")
    ax.legend()

    # Plot 2: C14 click rate scatter (top 30 values)
    ax = axes[1]
    cr_c14 = df.groupby("C14")[TARGET].agg(["mean", "count"]).reset_index()
    cr_c14 = cr_c14[cr_c14["count"] > 100].sort_values("C14")
    ax.scatter(cr_c14["C14"], cr_c14["mean"] * 100,
               s=cr_c14["count"] / cr_c14["count"].max() * 100 + 5,
               alpha=0.6, color="#10b981", edgecolors="#064e3b", linewidths=0.5)
    ax.set_xlabel("C14 value")
    ax.set_ylabel("Click rate (%)")
    ax.set_title("C14 vs Click Rate (bubble size = impressions)")

    fig.suptitle("Bi-variate: feature vs click", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "eda_16_scatter_feature_click.png")

    # ------------------------------------------------------------------
    # Task 10 — Categorical count plots  [NEW]
    # ------------------------------------------------------------------
    print("TASK 10: Categorical count plots")
    cat_count_cols = ["device_type", "banner_pos", "device_conn_type"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, cat_count_cols):
        counts_col = df[col].value_counts().sort_index()
        ax.bar(counts_col.index.astype(str), counts_col.values,
               color="#8b5cf6", alpha=0.85)
        ax.set_title(f"Count: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        for i, v in enumerate(counts_col.values):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Categorical feature counts (uni-variate)", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "eda_14_feature_count_plots.png")

    # Top categories
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col in zip(axes2, ["site_category", "app_category"]):
        top = df[col].value_counts().head(10)
        ax.barh(top.index.astype(str), top.values, color="#0ea5e9")
        ax.set_title(f"Top 10: {col}")
        ax.set_xlabel("Count")
    fig2.suptitle("Top site & app categories (count)", fontsize=14)
    fig2.tight_layout()
    save_fig(fig2, "eda_14b_top_categories.png")

    # ------------------------------------------------------------------
    # Task 11 — Click rate by banner position  (was step 9)
    # ------------------------------------------------------------------
    print("TASK 11: Click rate by banner position")
    cr_banner = df.groupby("banner_pos")[TARGET].agg(["mean", "count"]).reset_index()
    cr_banner.columns = ["banner_pos", "click_rate", "impressions"]
    save_table(cr_banner, "02_click_rate_banner_pos.csv")
    summary["click_rate_by_banner_pos"] = cr_banner.to_dict(orient="records")
    fig, ax = plt.subplots()
    ax.bar(cr_banner["banner_pos"].astype(str), cr_banner["click_rate"] * 100,
           color="#6366f1")
    ax.set_title("Click rate by banner position")
    ax.set_xlabel("banner_pos")
    ax.set_ylabel("Click rate (%)")
    save_fig(fig, "eda_04_click_rate_banner_pos.png")

    # ------------------------------------------------------------------
    # Task 12 — Click rate by site / app category  (was step 12)
    # ------------------------------------------------------------------
    print("TASK 12: Click rate by site & app category")
    for col, fname in [("site_category", "eda_08_click_rate_site_category.png"),
                       ("app_category", "eda_09_click_rate_app_category.png")]:
        cr = df.groupby(col)[TARGET].agg(["mean", "count"]).reset_index()
        cr.columns = [col, "click_rate", "impressions"]
        cr = cr.sort_values("click_rate", ascending=False)
        save_table(cr, f"02_click_rate_{col}.csv")
        summary[f"click_rate_by_{col}"] = cr.head(10).to_dict(orient="records")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(cr[col].astype(str).head(15), cr["click_rate"].head(15) * 100,
                color="#10b981")
        ax.set_title(f"Click rate by {col} (top 15)")
        ax.set_xlabel("Click rate (%)")
        save_fig(fig, fname)

    # ------------------------------------------------------------------
    # Task 13 — Click rate by hour + day of week  [EXPANDED]
    # ------------------------------------------------------------------
    print("TASK 13: Click rate by hour of day + day of week")
    df["hour_of_day"] = df["hour"] % 100
    cr_hour = df.groupby("hour_of_day")[TARGET].agg(["mean", "count"]).reset_index()
    cr_hour.columns = ["hour_of_day", "click_rate", "impressions"]
    save_table(cr_hour, "02_click_rate_hour.csv")
    summary["click_rate_by_hour"] = cr_hour.to_dict(orient="records")

    # Parse full datetime for day_of_week
    df["dt"] = pd.to_datetime(df["hour"].astype(str).str.zfill(8), format="%y%m%d%H")
    df["day_of_week"] = df["dt"].dt.dayofweek
    cr_dow = df.groupby("day_of_week")[TARGET].agg(["mean", "count"]).reset_index()
    cr_dow.columns = ["day_of_week", "click_rate", "impressions"]
    summary["click_rate_by_dayofweek"] = cr_dow.to_dict(orient="records")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(cr_hour["hour_of_day"], cr_hour["click_rate"] * 100,
                 marker="o", color="#6366f1", lw=2)
    axes[0].set_title("Click rate by hour of day")
    axes[0].set_xlabel("Hour (0–23)")
    axes[0].set_ylabel("Click rate (%)")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].fill_between(cr_hour["hour_of_day"], cr_hour["click_rate"] * 100,
                         alpha=0.15, color="#6366f1")

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    axes[1].bar(cr_dow["day_of_week"], cr_dow["click_rate"] * 100,
                color="#8b5cf6", alpha=0.85)
    axes[1].set_xticks(cr_dow["day_of_week"])
    axes[1].set_xticklabels(day_labels)
    axes[1].set_title("Click rate by day of week")
    axes[1].set_ylabel("Click rate (%)")

    fig.suptitle("Temporal click-rate patterns (multi-variate)", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "eda_05_click_rate_hour.png")

    # Day of week only (standalone figure)
    fig, ax = plt.subplots()
    ax.bar(cr_dow["day_of_week"], cr_dow["click_rate"] * 100,
           color="#8b5cf6", alpha=0.85)
    ax.set_xticks(cr_dow["day_of_week"])
    ax.set_xticklabels(day_labels)
    ax.set_title("Click rate by day of week")
    ax.set_ylabel("Click rate (%)")
    save_fig(fig, "eda_17_click_rate_dayofweek.png")

    # ------------------------------------------------------------------
    # Task 14 — Click rate by device type & connection type
    # ------------------------------------------------------------------
    print("TASK 14: Click rate by device type & connection type")
    for col, fname in [("device_type", "eda_06_click_rate_device_type.png"),
                       ("device_conn_type", "eda_07_click_rate_conn_type.png")]:
        cr = df.groupby(col)[TARGET].agg(["mean", "count"]).reset_index()
        cr.columns = [col, "click_rate", "impressions"]
        save_table(cr, f"02_click_rate_{col}.csv")
        summary[f"click_rate_by_{col}"] = cr.to_dict(orient="records")
        fig, ax = plt.subplots()
        ax.bar(cr[col].astype(str), cr["click_rate"] * 100, color="#8b5cf6")
        ax.set_title(f"Click rate by {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Click rate (%)")
        save_fig(fig, fname)

    # Cardinality chart
    cat_cols = df.select_dtypes(include="object").columns
    cardinality = {c: int(df[c].nunique()) for c in cat_cols}
    summary["cardinality"] = cardinality
    card_df = pd.DataFrame({"column": list(cardinality),
                             "unique_values": list(cardinality.values())})
    save_table(card_df.sort_values("unique_values", ascending=False),
               "02_cardinality.csv")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(list(cardinality.keys()), list(cardinality.values()), color="#10b981")
    ax.set_xscale("log")
    ax.set_title("Cardinality of categorical columns (log scale)")
    ax.set_xlabel("Unique values")
    save_fig(fig, "eda_03_cardinality.png")

    # Click rate across C14–C21
    for col in ["C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]:
        cr = df.groupby(col)[TARGET].mean().reset_index()
        cr.columns = [col, "click_rate"]
        summary[f"click_rate_by_{col}"] = cr.to_dict(orient="records")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, col in zip(axes.ravel(),
                       ["C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]):
        cr = df.groupby(col)[TARGET].mean()
        ax.plot(cr.index.astype(str), cr.values * 100,
                marker=".", color="#ef4444", markersize=3)
        ax.set_title(f"Click rate by {col}")
        ax.tick_params(axis="x", rotation=90, labelsize=6)
    fig.suptitle("Click rate across anonymised numeric features", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "eda_12_click_rate_c14_c21.png")

    # ------------------------------------------------------------------
    # Task 15 — Pairplot (scatter_matrix, 3k sample for speed)  [NEW]
    # ------------------------------------------------------------------
    print("TASK 15: Pairplot — scatter_matrix on 3k-row sample")
    pair_cols = ["banner_pos", "C14", "C17", "C21", TARGET]
    pair_sample = df[pair_cols].sample(n=min(3000, n), random_state=SEED)

    # Use pandas scatter_matrix (matplotlib-native, 5-10x faster than seaborn pairplot)
    from pandas.plotting import scatter_matrix
    fig, axes = plt.subplots(len(pair_cols), len(pair_cols),
                             figsize=(14, 12))
    sm = scatter_matrix(
        pair_sample,
        ax=axes,
        alpha=0.25,
        figsize=(14, 12),
        diagonal="hist",
        color=pair_sample[TARGET].map({0: "#94a3b8", 1: "#6366f1"}),
        hist_kwds={"bins": 20, "edgecolor": "white"},
        s=8,
    )
    plt.suptitle("Pairplot — banner_pos, C14, C17, C21 (colored by click)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    save_fig(plt.gcf(), "eda_18_pairplot.png")

    save_json("02_eda_summary.json", summary)
    print("\nEDA complete — 15 tasks, all figures saved.")


if __name__ == "__main__":
    main()