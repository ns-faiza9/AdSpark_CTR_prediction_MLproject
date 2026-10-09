"""Stage 15 — Imbalanced Classification Metrics Suite (CO5).

Evaluates models under heavy class imbalance (~17% CTR positive rate),
plotting Precision-Recall curves and Confusion Matrices at tuned thresholds.

Produces:
    analysis/output/15_imbalanced_summary.json
    analysis/output/figures/imb_01_roc_pr_curves.png
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, roc_auc_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

from common import load_processed, save_fig, save_json, TARGET, SEED


def main() -> None:
    print("Stage 15: Imbalanced Classification Metrics")
    df = load_processed()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=30, max_depth=6, class_weight='balanced', random_state=SEED)
    rf.fit(X_train, y_train)

    y_prob = rf.predict_proba(X_test)[:, 1]

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = float(auc(rec, prec))
    roc_auc = float(roc_auc_score(y_test, y_prob))

    # Evaluate at threshold = 0.20
    thresh = 0.20
    y_pred = (y_prob >= thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    ll = float(log_loss(y_test, y_prob))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(rec, prec, color='#fbbf24', lw=2, label=f'PR Curve (PR-AUC = {pr_auc:.4f})')
    ax1.set_title("Precision-Recall Curve (Imbalanced Benchmark)")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.matshow(cm, cmap='Blues', alpha=0.7)
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f"{cm[i, j]:,}", ha='center', va='center', fontsize=12, fontweight='bold')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['No Click', 'Click'])
    ax2.set_yticklabels(['No Click', 'Click'])
    ax2.set_title(f"Confusion Matrix (Threshold = {thresh})")
    fig.tight_layout()
    save_fig(fig, "imb_01_roc_pr_curves.png")

    summary = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "log_loss": round(ll, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm.tolist()
    }
    save_json("15_imbalanced_summary.json", summary)
    print(f"Imbalanced metrics evaluation complete. PR-AUC = {pr_auc:.4f}, ROC-AUC = {roc_auc:.4f}.")


if __name__ == "__main__":
    main()
