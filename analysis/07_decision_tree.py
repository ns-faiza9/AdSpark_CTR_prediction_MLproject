"""Stage 07 — Decision Tree Classifier.

Trains a DecisionTreeClassifier with balanced class weights on the engineered
features. Also sweeps max_depth 1–10 to show the depth vs AUC trade-off.

Produces:
    analysis/output/07_decision_tree_summary.json
    analysis/output/figures/dt_*.png

Usage:
    python analysis/07_decision_tree.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

from common import load_processed, save_fig, save_json, TARGET, SEED


def main() -> None:
    df = load_processed()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    feature_names = list(X.columns)
    print(f"Features: {X.shape[1]}, Rows: {len(df):,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # ------------------------------------------------------------------
    # Primary model (depth = 5)
    # ------------------------------------------------------------------
    DEFAULT_DEPTH = 5
    primary = DecisionTreeClassifier(
        max_depth=DEFAULT_DEPTH,
        class_weight="balanced",
        random_state=SEED,
    )
    primary.fit(X_train, y_train)
    y_prob = primary.predict_proba(X_test)[:, 1]
    y_bin = primary.predict(X_test)
    y_train_bin = primary.predict(X_train)

    report = classification_report(y_test, y_bin,
                                   target_names=["No Click", "Click"],
                                   output_dict=True)
    metrics = {
        "max_depth": DEFAULT_DEPTH,
        "n_features": int(X.shape[1]),
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "train_accuracy": round(float(accuracy_score(y_train, y_train_bin)) * 100, 2),
        "test_accuracy": round(float(accuracy_score(y_test, y_bin)) * 100, 2),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "f1": round(float(f1_score(y_test, y_bin, zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, y_bin, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_bin, zero_division=0)), 4),
        "n_leaves": int(primary.get_n_leaves()),
        "n_nodes": int(primary.tree_.node_count),
        "classification_report": {
            cls: {
                "precision": round(vals["precision"], 4),
                "recall": round(vals["recall"], 4),
                "f1_score": round(vals["f1-score"], 4),
                "support": int(vals["support"]),
            }
            for cls, vals in report.items()
            if cls in ["No Click", "Click"]
        },
    }

    # ------------------------------------------------------------------
    # Depth sweep: max_depth 1 to 10
    # ------------------------------------------------------------------
    depth_results = []
    for d in range(1, 11):
        dt = DecisionTreeClassifier(
            max_depth=d, class_weight="balanced", random_state=SEED
        )
        dt.fit(X_train, y_train)
        yp = dt.predict_proba(X_test)[:, 1]
        yb = dt.predict(X_test)
        depth_results.append({
            "depth": d,
            "train_accuracy": round(float(accuracy_score(y_train, dt.predict(X_train))) * 100, 2),
            "test_accuracy": round(float(accuracy_score(y_test, yb)) * 100, 2),
            "roc_auc": round(float(roc_auc_score(y_test, yp)), 4),
            "f1": round(float(f1_score(y_test, yb, zero_division=0)), 4),
        })
    metrics["depth_sweep"] = depth_results

    # Feature importances (top 20)
    importances = pd.Series(primary.feature_importances_,
                            index=feature_names).sort_values(ascending=False)
    metrics["top_features"] = importances.head(20).round(4).to_dict()

    save_json("07_decision_tree_summary.json", metrics)

    print("\n--- Decision Tree (depth=5) metrics ---")
    for k, v in metrics.items():
        if not isinstance(v, (dict, list)):
            print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    # 1. Tree visualisation (depth limited to 3 for readability)
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(
        primary,
        max_depth=3,
        feature_names=feature_names,
        class_names=["No Click", "Click"],
        filled=True,
        impurity=True,
        rounded=True,
        fontsize=7,
        ax=ax,
    )
    ax.set_title("Decision Tree — top 3 levels (max_depth=5 model)", fontsize=13)
    save_fig(fig, "dt_01_tree_visualization.png")

    # 2. Feature importances
    top_imp = importances.head(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_imp.index[::-1], top_imp.values[::-1], color="#6366f1", alpha=0.85)
    ax.set_xlabel("Feature importance (Gini)")
    ax.set_title("Decision Tree — top 15 feature importances")
    fig.tight_layout()
    save_fig(fig, "dt_02_feature_importance.png")

    # 3. Confusion matrix
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_bin, ax=ax, cmap="Blues",
        display_labels=["No Click", "Click"]
    )
    ax.set_title("Confusion matrix — Decision Tree (depth=5)")
    save_fig(fig, "dt_03_confusion_matrix.png")

    # 4. Depth vs AUC curve
    depths = [r["depth"] for r in depth_results]
    aucs = [r["roc_auc"] for r in depth_results]
    train_accs = [r["train_accuracy"] for r in depth_results]
    test_accs = [r["test_accuracy"] for r in depth_results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(depths, aucs, "o-", color="#6366f1", lw=2)
    axes[0].axvline(DEFAULT_DEPTH, color="#ef4444", ls="--", alpha=0.7,
                    label=f"Selected depth={DEFAULT_DEPTH}")
    axes[0].set_xlabel("max_depth")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("Depth sweep — ROC-AUC")
    axes[0].set_xticks(depths)
    axes[0].legend()

    axes[1].plot(depths, train_accs, "o-", color="#6366f1", lw=2,
                 label="Training accuracy")
    axes[1].plot(depths, test_accs, "s-", color="#10b981", lw=2,
                 label="Testing accuracy")
    axes[1].axvline(DEFAULT_DEPTH, color="#ef4444", ls="--", alpha=0.7)
    axes[1].set_xlabel("max_depth")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Depth sweep — Accuracy")
    axes[1].set_xticks(depths)
    axes[1].legend()

    fig.suptitle("Decision Tree: depth sweep (max_depth 1–10)", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "dt_04_depth_vs_auc.png")

    print("\nDecision tree complete.")


if __name__ == "__main__":
    main()
