"""Stage 17 — Statistical Significance Testing (McNemar's Test) (CO5).

Performs paired McNemar chi-square statistical significance test comparing
Logistic Regression against Gradient Boosted Trees model predictions on held-out test impressions.

Produces:
    analysis/output/17_significance_summary.json
    analysis/output/figures/sig_01_mcnemar_contingency.png
"""
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from common import load_processed, save_fig, save_json, TARGET, SEED


def main() -> None:
    print("Stage 17: Statistical Significance Testing")
    df = load_processed()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    sample_df = df.sample(n=min(30000, len(df)), random_state=SEED)
    X_s = sample_df.drop(columns=[TARGET])
    y_s = sample_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y_s, test_size=0.25, random_state=SEED, stratify=y_s
    )

    # Model A: Baseline Logistic Regression
    model_a = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, random_state=SEED))
    model_a.fit(X_train, y_train)
    pred_a = (model_a.predict_proba(X_test)[:, 1] >= 0.20).astype(int)

    # Model B: Candidate Gradient Boosted Trees
    model_b = LGBMClassifier(n_estimators=40, max_depth=5, random_state=SEED, verbosity=-1, n_jobs=-1)
    model_b.fit(X_train, y_train)
    pred_b = (model_b.predict_proba(X_test)[:, 1] >= 0.20).astype(int)

    # Correctness boolean arrays
    correct_a = (pred_a == y_test.values)
    correct_b = (pred_b == y_test.values)

    # Contingency matrix:
    # [[both_correct, A_correct_B_wrong], [B_correct_A_wrong, both_wrong]]
    both_correct = int(np.sum(correct_a & correct_b))
    a_corr_b_wrong = int(np.sum(correct_a & ~correct_b))
    b_corr_a_wrong = int(np.sum(~correct_a & correct_b))
    both_wrong = int(np.sum(~correct_a & ~correct_b))

    contingency = np.array([
        [both_correct, a_corr_b_wrong],
        [b_corr_a_wrong, both_wrong]
    ])

    # McNemar chi-square with continuity correction: (|b - c| - 1)^2 / (b + c)
    b = a_corr_b_wrong
    c = b_corr_a_wrong
    if b + c > 0:
        mcnemar_stat = float(((abs(b - c) - 1.0) ** 2) / (b + c))
        p_value = float(chi2.sf(mcnemar_stat, df=1))
    else:
        mcnemar_stat = 0.0
        p_value = 1.0

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.matshow(contingency, cmap='Purples', alpha=0.7)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{contingency[i, j]:,}", ha='center', va='center', fontsize=12, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Candidate Correct', 'Candidate Wrong'])
    ax.set_yticklabels(['Baseline Correct', 'Baseline Wrong'])
    ax.set_title(f"McNemar's Contingency Matrix (p = {p_value:.4e})")
    fig.tight_layout()
    save_fig(fig, "sig_01_mcnemar_contingency.png")

    summary = {
        "model_a": "Logistic Regression",
        "model_b": "XGBoost / LightGBM",
        "mcnemar_statistic": round(mcnemar_stat, 4),
        "p_value": float(p_value),
        "p_value_display": f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.4f}",
        "statistically_significant": bool(p_value < 0.05),
        "contingency_matrix": contingency.tolist(),
        "conclusion": f"The performance difference is {'statistically significant (p < 0.05)' if p_value < 0.05 else 'not statistically significant'}."
    }
    save_json("17_significance_summary.json", summary)
    print(f"Significance test complete. McNemar stat = {mcnemar_stat:.4f}, p-value = {p_value:.4e}.")


if __name__ == "__main__":
    main()
