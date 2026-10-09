"""Stage 08 — Ensemble Learning.

Trains three ensemble classifiers and compares their performance:
  - Random Forest   (Bagging family)
  - Gradient Boosting (Boosting family, sklearn)
  - AdaBoost        (Boosting family)

Produces:
    analysis/output/08_ensemble_summary.json
    analysis/output/figures/ens_*.png

Usage:
    python analysis/08_ensemble.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from common import load_processed, save_fig, save_json, TARGET, SEED


MODELS = {
    "Random Forest": RandomForestClassifier(
        n_estimators=40,
        max_depth=8,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    ),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, class_weight="balanced"),
        n_estimators=25,
        learning_rate=0.5,
        random_state=SEED,
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=50,
        max_depth=5,
        learning_rate=0.1,
        random_state=SEED,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=50,
        max_depth=5,
        learning_rate=0.1,
        random_state=SEED,
        verbosity=-1,
        n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=50,
        max_depth=5,
        learning_rate=0.1,
        random_state=SEED,
        n_jobs=-1,
        eval_metric="logloss",
    ),
}

COLORS = {
    "Random Forest": "#6366f1",
    "AdaBoost": "#f59e0b",
    "Gradient Boosting": "#10b981",
    "LightGBM": "#06b6d4",
    "XGBoost": "#ec4899",
}

ALGO_FAMILY = {
    "Random Forest": "Bagging",
    "AdaBoost": "Boosting",
    "Gradient Boosting": "Boosting",
    "LightGBM": "Boosting",
    "XGBoost": "Boosting",
}


def main() -> None:
    df = load_processed()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    feature_names = list(X.columns)
    print(f"Features: {X.shape[1]}, Rows: {len(df):,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    results = {}
    probs = {}
    preds = {}

    for name, model in MODELS.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_bin = (y_prob >= 0.5).astype(int)
        y_train_bin = (model.predict_proba(X_train)[:, 1] >= 0.5).astype(int)

        probs[name] = y_prob
        preds[name] = y_bin

        report = classification_report(y_test, y_bin,
                                       target_names=["No Click", "Click"],
                                       output_dict=True)
        results[name] = {
            "algorithm_family": ALGO_FAMILY[name],
            "n_features": int(X.shape[1]),
            "train_records": int(len(X_train)),
            "test_records": int(len(X_test)),
            "train_accuracy": round(float(accuracy_score(y_train, y_train_bin)) * 100, 2),
            "test_accuracy": round(float(accuracy_score(y_test, y_bin)) * 100, 2),
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "f1": round(float(f1_score(y_test, y_bin, zero_division=0)), 4),
            "precision": round(float(precision_score(y_test, y_bin, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_bin, zero_division=0)), 4),
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

        # Feature importances (RF and GB have .feature_importances_)
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_,
                            index=feature_names).sort_values(ascending=False)
            results[name]["top_features"] = imp.head(20).round(4).to_dict()

    # Identify best model by AUC
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    summary = {"models": results, "best_model": best_name}
    save_json("08_ensemble_summary.json", summary)

    print("\n--- Ensemble Results ---")
    for name, m in results.items():
        print(f"  {name}: test_acc={m['test_accuracy']:.2f}%, "
              f"AUC={m['roc_auc']:.4f}, F1={m['f1']:.4f}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    # 1. ROC curves — all models on one chart
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_prob in probs.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2, color=COLORS[name],
                label=f"{name} (AUC = {results[name]['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — Ensemble models")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, "ens_01_roc_comparison.png")

    # 2. AUC comparison bar
    names = list(results.keys())
    aucs = [results[n]["roc_auc"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, aucs, color=[COLORS[n] for n in names], alpha=0.85)
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001, f"{v:.4f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Ensemble models — AUC comparison")
    ax.set_ylim(bottom=min(aucs) - 0.02)
    fig.tight_layout()
    save_fig(fig, "ens_02_auc_comparison.png")

    # 3. Accuracy comparison (train vs test)
    train_accs = [results[n]["train_accuracy"] for n in names]
    test_accs = [results[n]["test_accuracy"] for n in names]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width / 2, train_accs, width, label="Training Acc",
                color="#6366f1", alpha=0.85)
    b2 = ax.bar(x + width / 2, test_accs, width, label="Testing Acc",
                color="#10b981", alpha=0.85)
    for bar, v in zip(list(b1) + list(b2), train_accs + test_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Ensemble models — Accuracy comparison")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "ens_03_accuracy_comparison.png")

    # 4. Feature importance — Random Forest
    if "top_features" in results.get("Random Forest", {}):
        rf_imp = pd.Series(results["Random Forest"]["top_features"]).sort_values()
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(rf_imp.index, rf_imp.values, color="#6366f1", alpha=0.85)
        ax.set_xlabel("Feature importance")
        ax.set_title("Random Forest — top 20 feature importances")
        fig.tight_layout()
        save_fig(fig, "ens_04_feature_importance_rf.png")

    # 5. Confusion matrix — best model
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds[best_name], ax=ax, cmap="Blues",
        display_labels=["No Click", "Click"]
    )
    ax.set_title(f"Confusion matrix — {best_name} (best model)")
    save_fig(fig, "ens_05_confusion_matrix_best.png")

    print("\nEnsemble learning complete.")


if __name__ == "__main__":
    main()
