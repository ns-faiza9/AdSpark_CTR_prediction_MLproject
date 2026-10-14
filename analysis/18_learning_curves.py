"""Stage 18 — Learning and Validation Curves (CO3, CO5).

Plots train vs validation scores across varying training set sizes to diagnose
underfitting (high bias) vs overfitting (high variance).

Produces:
    analysis/output/18_learning_curves_summary.json
    analysis/output/figures/lc_01_learning_validation.png
"""
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import learning_curve

from common import load_processed, save_fig, save_json, TARGET, SEED


def main() -> None:
    print("Stage 18: Learning & Validation Curves")
    df = load_processed()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    sample_size = min(15000, len(X))
    X_sample = X.sample(n=sample_size, random_state=SEED)
    y_sample = y.loc[X_sample.index]

    model = LGBMClassifier(
        n_estimators=30,
        max_depth=5,
        random_state=SEED,
        verbosity=-1,
        n_jobs=-1
    )

    train_sizes, train_scores, val_scores = learning_curve(
        model, X_sample, y_sample,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=3, scoring='roc_auc', n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    gap = float(train_mean[-1] - val_mean[-1])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, 'o-', color='#38bdf8', lw=2, label='Training Score (ROC-AUC)')
    ax.plot(train_sizes, val_mean, 's-', color='#34d399', lw=2, label='Validation Score (ROC-AUC)')
    ax.set_title("Learning Curves (Underfitting vs Overfitting Diagnosis)")
    ax.set_xlabel("Training Sample Size")
    ax.set_ylabel("ROC-AUC Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "lc_01_learning_validation.png")

    diagnosis = "Good Fit — Convergence gap is narrow (< 0.08), indicating balanced bias-variance." if gap < 0.08 else "Mild Variance — Moderate generalization gap."
    summary = {
        "final_train_score": round(float(train_mean[-1]), 4),
        "final_val_score": round(float(val_mean[-1]), 4),
        "convergence_gap": round(gap, 4),
        "bias_variance_diagnosis": diagnosis
    }
    save_json("18_learning_curves_summary.json", summary)
    print(f"Learning curve analysis complete. Final Train = {train_mean[-1]:.4f}, Final Val = {val_mean[-1]:.4f}, Gap = {gap:.4f}.")


if __name__ == "__main__":
    main()
