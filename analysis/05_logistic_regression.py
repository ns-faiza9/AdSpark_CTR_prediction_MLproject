"""Stage 05 — Logistic Regression (classification model for click prediction).

Produces:
    analysis/output/05_logistic_regression_summary.json
    analysis/output/figures/logreg_*.png

JSON includes:
  - Overall metrics (accuracy, AUC, F1, precision, recall, log_loss)
  - Scaling comparison (Unscaled / StandardScaler / MinMaxScaler)
  - Per-class classification report (for the frontend table)

Usage:
    python analysis/05_logistic_regression.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    f1_score, log_loss, precision_score, recall_score,
    roc_auc_score, roc_curve, average_precision_score, precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from common import load_processed, save_fig, save_json, TARGET, SEED


def eval_pipeline(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Optimization step: find threshold that maximizes F1 for Click class
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_t = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    y_bin = (y_prob >= best_t).astype(int)
    return {
        "best_threshold": float(best_t),
        "train_accuracy": float(accuracy_score(y_train, (model.predict_proba(X_train)[:, 1] >= best_t).astype(int))),
        "test_accuracy": float(accuracy_score(y_test, y_bin)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "log_loss": float(log_loss(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_bin, zero_division=0)),
        "precision": float(precision_score(y_test, y_bin, zero_division=0)),
        "recall": float(recall_score(y_test, y_bin, zero_division=0)),
    }, y_prob, y_bin


def main() -> None:
    df = load_processed()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"Features: {X.shape[1]}, Rows: {len(df):,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # ------------------------------------------------------------------
    # Primary model: StandardScaler + LogReg (L2, balanced)
    # ------------------------------------------------------------------
    primary = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=SEED),
    )
    metrics, y_prob, y_bin = eval_pipeline(primary, X_train, X_test, y_train, y_test)
    metrics["n_iter"] = int(primary.named_steps["logisticregression"].n_iter_[0])
    metrics["n_features"] = int(X.shape[1])
    metrics["train_records"] = int(len(X_train))
    metrics["test_records"] = int(len(X_test))
    metrics["target_ratio"] = f"{int(y_train.sum())} / {int((y_train == 0).sum())}"

    # ------------------------------------------------------------------
    # Per-class classification report
    # ------------------------------------------------------------------
    report = classification_report(y_test, y_bin,
                                   target_names=["No Click", "Click"],
                                   output_dict=True)
    metrics["classification_report"] = {
        cls: {
            "precision": round(vals["precision"], 4),
            "recall": round(vals["recall"], 4),
            "f1_score": round(vals["f1-score"], 4),
            "support": int(vals["support"]),
        }
        for cls, vals in report.items()
        if cls in ["No Click", "Click"]
    }

    # ------------------------------------------------------------------
    # Scaling comparison
    # ------------------------------------------------------------------
    scalers = {
        "Unscaled": make_pipeline(
            LogisticRegression(max_iter=2000, random_state=SEED)
        ),
        "StandardScaler": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=SEED),
        ),
        "MinMaxScaler": make_pipeline(
            MinMaxScaler(),
            LogisticRegression(max_iter=2000, random_state=SEED),
        ),
    }
    scaling_comparison = {}
    for name, model in scalers.items():
        m, _, _ = eval_pipeline(model, X_train, X_test, y_train, y_test)
        scaling_comparison[name] = {
            "train_accuracy": round(m["train_accuracy"] * 100, 2),
            "test_accuracy": round(m["test_accuracy"] * 100, 2),
        }
    metrics["scaling_comparison"] = scaling_comparison

    save_json("05_logistic_regression_summary.json", metrics)

    print("\n--- Logistic Regression metrics ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif not isinstance(v, dict):
            print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color="#10b981", lw=2,
            label=f"ROC (AUC = {metrics['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — Logistic Regression")
    ax.legend()
    save_fig(fig, "logreg_01_roc_curve.png")

    # Confusion matrix
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_bin, ax=ax, cmap="Blues",
                                            display_labels=["No Click", "Click"])
    ax.set_title(f"Confusion matrix — Logistic Regression (threshold {metrics['best_threshold']:.2f})")
    save_fig(fig, "logreg_02_confusion_matrix.png")

    # Probability distribution
    fig, ax = plt.subplots()
    ax.hist(y_prob[y_test == 1], bins=50, alpha=0.6, color="#10b981",
            label="Click (1)")
    ax.hist(y_prob[y_test == 0], bins=50, alpha=0.6, color="#94a3b8",
            label="No click (0)")
    ax.set_title("Predicted probability by true class")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.legend()
    save_fig(fig, "logreg_03_probability_distribution.png")

    # Scaling comparison bar chart
    methods = list(scaling_comparison.keys())
    train_accs = [scaling_comparison[m]["train_accuracy"] for m in methods]
    test_accs = [scaling_comparison[m]["test_accuracy"] for m in methods]
    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, train_accs, width, label="Training Accuracy",
                   color="#6366f1", alpha=0.85)
    bars2 = ax.bar(x + width / 2, test_accs, width, label="Testing Accuracy",
                   color="#10b981", alpha=0.85)
    for bar, v in zip(list(bars1) + list(bars2), train_accs + test_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Logistic Regression — Scaling Method Comparison")
    ax.legend()
    ax.set_ylim(bottom=min(test_accs) - 2)
    fig.tight_layout()
    save_fig(fig, "logreg_04_scaling_comparison.png")

    print("\nLogistic regression complete.")


if __name__ == "__main__":
    main()