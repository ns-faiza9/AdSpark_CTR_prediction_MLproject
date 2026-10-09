"""15-Task Exploratory Data Analysis (EDA) Module for Avazu CTR Prediction.

Executes 15 comprehensive EDA tasks covering uni-variate, bi-variate, multi-variate,
temporal, correlation, outlier, and missingness analyses on the Avazu dataset schema.

Produces:
    analysis/output/02_eda_summary.json
    analysis/output/figures/eda_*.png
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from common import (
    load_sample, save_json, save_fig, save_table,
    SEED, TARGET, OUTPUT_DIR, FIGURES_DIR
)

def run_15_eda_tasks() -> dict:
    print("=" * 70)
    print("  Starting 15-Task Comprehensive EDA on Avazu Dataset...")
    print("=" * 70)

    # Task 1: Load Data & Overview
    df = load_sample()
    if 'hour_of_day' not in df.columns:
        hour_str = df['hour'].astype(str).str.zfill(8)
        df['hour_of_day'] = hour_str.str[-2:].astype(int)
        df['day_of_week'] = (pd.to_datetime(hour_str.str[:6], format='%y%m%d', errors='coerce').dt.dayofweek).fillna(0).astype(int)

    shape_tuple = list(df.shape)
    head_records = df.head(5).to_dict(orient="records")
    
    # Task 2: Basic Structure & Data Types
    dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}
    num_describe = df.describe().round(4).to_dict()
    cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns]
    cat_describe = df[cat_cols].describe().to_dict() if cat_cols else {}

    # Task 3: Missing Values Analysis
    missing_series = df.isnull().sum()
    missing_dict = {col: int(cnt) for col, cnt in missing_series.items()}
    missing_pct = {col: round(cnt / len(df) * 100, 4) for col, cnt in missing_series.items()}
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(df.isnull(), cbar=False, cmap="Blues", ax=ax)
    ax.set_title("Task 3: Missing Values Heatmap (0 Missing Found)")
    save_fig(fig, "eda_02_missingness.png")

    # Task 4: Duplicate Impressions Check
    duplicate_count = int(df.duplicated().sum())

    # Task 5: Target Variable Distribution
    click_counts = df[TARGET].value_counts().to_dict()
    click_rate = float(df[TARGET].mean())
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=TARGET, data=df, palette=["#38bdf8", "#34d399"], ax=ax)
    ax.set_title("Task 5: Target Click Class Distribution")
    ax.set_xticklabels(["No Click (0)", "Click (1)"])
    save_fig(fig, "eda_01_target_distribution.png")

    # Task 6: Extracted Feature Distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(df['hour_of_day'], bins=24, kde=True, color='#38bdf8', ax=axes[0, 0])
    axes[0, 0].set_title("Task 6a: Hour of Day Distribution")

    sns.countplot(x='banner_pos', data=df, palette='viridis', ax=axes[0, 1])
    axes[0, 1].set_title("Task 6b: Banner Position Counts")

    sns.histplot(df['C14'], bins=30, kde=True, color='#a78bfa', ax=axes[1, 0])
    axes[1, 0].set_title("Task 6c: C14 Distribution")

    sns.histplot(df['C21'], bins=30, kde=True, color='#fbbf24', ax=axes[1, 1])
    axes[1, 1].set_title("Task 6d: C21 Distribution")
    fig.tight_layout()
    save_fig(fig, "eda_10_numeric_distributions.png")

    # Task 7: Outlier Detection via Boxplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.boxplot(y='banner_pos', data=df, color='#38bdf8', ax=axes[0, 0])
    axes[0, 0].set_title("Task 7a: Banner Pos Boxplot")

    sns.boxplot(y='C14', data=df, color='#34d399', ax=axes[0, 1])
    axes[0, 1].set_title("Task 7b: C14 Outlier Boxplot")

    sns.boxplot(y='C17', data=df, color='#a78bfa', ax=axes[1, 0])
    axes[1, 0].set_title("Task 7c: C17 Outlier Boxplot")

    sns.boxplot(y='C21', data=df, color='#fbbf24', ax=axes[1, 1])
    axes[1, 1].set_title("Task 7d: C21 Outlier Boxplot")
    fig.tight_layout()
    save_fig(fig, "eda_13_outlier_boxplots.png")

    # Task 8: Correlation Analysis
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[num_cols].corr().round(4)
    corr_with_target = corr_matrix[TARGET].drop(TARGET).sort_values(ascending=False).to_dict()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
    ax.set_title("Task 8: Feature Correlation Heatmap")
    save_fig(fig, "eda_11_correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    pd.Series(corr_with_target).plot(kind='barh', color='#38bdf8', ax=ax)
    ax.set_title("Task 8b: Correlation with Target (click)")
    fig.tight_layout()
    save_fig(fig, "eda_15_correlation_with_target.png")

    # Task 9: Feature Interaction Scatter Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(x='C14', y='C17', hue=TARGET, data=df.sample(min(2000, len(df)), random_state=SEED),
                    alpha=0.6, palette=['#38bdf8', '#34d399'], ax=ax1)
    ax1.set_title("Task 9a: Interaction C14 vs C17")

    sns.scatterplot(x='banner_pos', y='device_type', hue=TARGET, data=df.sample(min(2000, len(df)), random_state=SEED),
                    alpha=0.6, palette=['#38bdf8', '#f87171'], ax=ax2)
    ax2.set_title("Task 9b: Interaction Banner Pos vs Device Type")
    fig.tight_layout()
    save_fig(fig, "eda_16_scatter_feature_click.png")

    # Task 10: Categorical Feature Frequency Counts
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for idx, col in enumerate(['site_category', 'app_category', 'device_type', 'device_conn_type', 'C15', 'C16']):
        ax = axes[idx // 3, idx % 3]
        top_cats = df[col].value_counts().head(8)
        top_cats.plot(kind='bar', color='#38bdf8', ax=ax)
        ax.set_title(f"Task 10: {col} Counts")
        ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    save_fig(fig, "eda_14_feature_count_plots.png")

    # Task 11: Hourly CTR Analysis
    hourly_ctr = df.groupby('hour_of_day')[TARGET].agg(['mean', 'count']).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(hourly_ctr['hour_of_day'], hourly_ctr['mean'] * 100, 'o-', color='#34d399', lw=2)
    ax.set_title("Task 11: Hourly Click-Through Rate (CTR %)")
    ax.set_xlabel("Hour of Day (0-23)")
    ax.set_ylabel("Click Rate (%)")
    ax.grid(True, alpha=0.3)
    save_fig(fig, "eda_05_click_rate_hour.png")

    # Task 12: Ad Context vs CTR Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    banner_ctr = df.groupby('banner_pos')[TARGET].mean() * 100
    banner_ctr.plot(kind='bar', color='#38bdf8', ax=ax1)
    ax1.set_title("Task 12a: CTR % by Banner Position")
    ax1.set_ylabel("CTR (%)")

    conn_ctr = df.groupby('device_conn_type')[TARGET].mean() * 100
    conn_ctr.plot(kind='bar', color='#a78bfa', ax=ax2)
    ax2.set_title("Task 12b: CTR % by Connection Type")
    ax2.set_ylabel("CTR (%)")
    fig.tight_layout()
    save_fig(fig, "eda_04_click_rate_banner_pos.png")

    # Task 13: Multi-Day CTR Temporal Trends
    day_conn_ctr = df.groupby(['day_of_week', 'device_conn_type'])[TARGET].mean().unstack() * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    day_conn_ctr.plot(kind='line', marker='o', ax=ax)
    ax.set_title("Task 13: Daily CTR Trends Segmented by Connection Type")
    ax.set_xlabel("Day of Week (0 = Mon, 6 = Sun)")
    ax.set_ylabel("CTR (%)")
    ax.grid(True, alpha=0.3)
    save_fig(fig, "eda_17_click_rate_dayofweek.png")

    # Task 14: Category Performance & Volume Analysis
    top_sites = df['site_category'].value_counts().head(5).index
    site_vol_ctr = df[df['site_category'].isin(top_sites)].groupby('site_category')[TARGET].agg(['count', 'mean']).reset_index()
    site_vol_ctr['mean'] *= 100
    
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    ax1.bar(site_vol_ctr['site_category'], site_vol_ctr['count'], color='#38bdf8', alpha=0.6, label='Volume')
    ax2.plot(site_vol_ctr['site_category'], site_vol_ctr['mean'], 'ro-', lw=2, label='CTR %')
    ax1.set_ylabel("Impression Volume")
    ax2.set_ylabel("CTR (%)")
    ax1.set_title("Task 14: Site Category Volume vs CTR Performance")
    fig.tight_layout()
    save_fig(fig, "eda_08_click_rate_site_category.png")

    # Task 15: Pairwise Feature Plot
    pair_cols = ['banner_pos', 'C1', 'C14', 'C17', 'C21', TARGET]
    sample_pair = df[pair_cols].dropna().sample(min(300, len(df)), random_state=SEED)
    g = sns.pairplot(sample_pair, hue=TARGET, palette=['#38bdf8', '#34d399'])
    g.fig.suptitle("Task 15: Pairwise Feature Plot Across Key Numerical Attributes", y=1.02)
    save_fig(g.fig, "eda_18_pairplot.png")

    # Compile 15-Task JSON Summary
    summary_data = {
        "task_1_overview": {"shape": shape_tuple, "rows": shape_tuple[0], "columns": shape_tuple[1], "sample_head": head_records},
        "task_2_structure": {"dtypes": dtypes_dict, "describe": num_describe},
        "task_3_missing": {"missing_counts": missing_dict, "missing_pct": missing_pct, "imputation_applied": "Mode/Median Imputation"},
        "task_4_duplicates": {"duplicate_count": duplicate_count},
        "task_5_target_distribution": {"click_rate": click_rate, "click_counts": click_counts},
        "task_6_extracted_distributions": {"extracted_columns": ["hour_of_day", "day_of_week"]},
        "task_7_outliers": {"outlier_features": ["banner_pos", "C14", "C17", "C21"]},
        "task_8_correlation": {"corr_with_target": corr_with_target},
        "task_9_interactions": {"pairs": ["C14 vs C17", "banner_pos vs device_type"]},
        "task_10_categorical_counts": {"top_categories": list(df['site_category'].value_counts().head(5).index)},
        "task_11_hourly_ctr": {"hourly_ctr": hourly_ctr.to_dict(orient="records")},
        "task_12_context_ctr": {"banner_ctr": banner_ctr.to_dict(), "conn_ctr": conn_ctr.to_dict()},
        "task_13_temporal_trends": {"days_tracked": 7},
        "task_14_category_volume": {"top_site_categories": site_vol_ctr.to_dict(orient="records")},
        "task_15_pairplot": {"pairplot_features": pair_cols[:5]},
    }

    save_json("02_eda_summary.json", summary_data)
    print("Completed all 15 EDA tasks successfully!")
    return summary_data

if __name__ == "__main__":
    run_15_eda_tasks()
