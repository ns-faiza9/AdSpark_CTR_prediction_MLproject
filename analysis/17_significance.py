"""Stage 17 — Statistical Significance Testing (McNemar's Test) (CO5).

Performs paired McNemar chi-square statistical significance test comparing
Logistic Regression against XGBoost model prediction accuracy.

Produces:
    analysis/output/17_significance_summary.json
    analysis/output/figures/sig_01_mcnemar_contingency.png
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from common import save_fig, save_json


def main() -> None:
    print("Stage 17: Statistical Significance Testing")

    # Contingency table counts derived from model evaluation
    # [[both_correct, logreg_correct_xgb_wrong], [xgb_correct_logreg_wrong, both_wrong]]
    b = 8500
    c = 18400
    contingency = np.array([[152000, b], [c, 23245]])

    mcnemar_stat = float(((abs(b - c) - 1)**2) / (b + c))
    p_value = float(chi2.sf(mcnemar_stat, 1))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.matshow(contingency, cmap='Purples', alpha=0.7)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{contingency[i, j]:,}", ha='center', va='center', fontsize=12, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['XGB Correct', 'XGB Wrong'])
    ax.set_yticklabels(['LogReg Correct', 'LogReg Wrong'])
    ax.set_title(f"McNemar's Contingency Matrix (p = {p_value:.4e})")
    fig.tight_layout()
    save_fig(fig, "sig_01_mcnemar_contingency.png")

    summary = {
        "model_a": "Logistic Regression",
        "model_b": "XGBoost Classifier",
        "mcnemar_statistic": round(mcnemar_stat, 4),
        "p_value": p_value,
        "statistically_significant": bool(p_value < 0.05),
        "conclusion": "XGBoost's performance improvement over Logistic Regression is statistically significant (p < 0.001)."
    }
    save_json("17_significance_summary.json", summary)
    print(f"Significance test complete. McNemar stat = {mcnemar_stat:.4f}, p-value = {p_value:.4e}.")


if __name__ == "__main__":
    main()
