"""Stage 06 — Regularization: Lasso, Ridge & Elastic Net.

Compares regularised linear models (Lasso, Ridge, ElasticNet) on the regression
view of the task, plus regularised logistic regression (L1 / L2 / ElasticNet)
for the classification view used in CTR prediction.

Produces:
    analysis/output/06_regularization_summary.json
    analysis/output/figures/reg_*.png

Usage:
    python analysis/06_regularization.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import (
    ElasticNet, Lasso, LogisticRegression, Ridge,
)
from sklearn.metrics import (
    f1_score, log_loss, mean_squared_error, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import load_processed, save_fig, save_json, TARGET, SEED


def main() -> None:
    df = load_processed()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # --- 1. Regularised linear regression (regression view) ----------------
    reg_models = {
        "Lasso (L1)": make_pipeline(StandardScaler(), Lasso(alpha=0.001, max_iter=5000, random_state=SEED)),
        "Ridge (L2)": make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED)),
        "Elastic Net": make_pipeline(StandardScaler(), ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=SEED)),
    }
    reg_results = {}
    for name, model in reg_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        reg_results[name] = {
            "mse": float(mean_squared_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "n_nonzero_coefs": int(np.sum(model.named_steps["lasso"].coef_ != 0))
            if "lasso" in model.named_steps
            else int(np.sum(model.named_steps["ridge"].coef_ != 0))
            if "ridge" in model.named_steps
            else int(np.sum(model.named_steps["elasticnet"].coef_ != 0)),
        }

    # --- 2. Regularised logistic regression (classification view) ----------
    logreg_models = {
        "LogReg L1": make_pipeline(StandardScaler(), LogisticRegression(
            solver="liblinear", l1_ratio=1.0, C=1.0, class_weight="balanced", random_state=SEED)),
        "LogReg L2": make_pipeline(StandardScaler(), LogisticRegression(
            solver="lbfgs", l1_ratio=0.0, C=1.0, max_iter=2000, class_weight="balanced", random_state=SEED)),
        "LogReg ElasticNet": make_pipeline(StandardScaler(), LogisticRegression(
            solver="saga", l1_ratio=0.5, C=1.0, max_iter=3000,
            class_weight="balanced", random_state=SEED)),
    }
    logreg_results = {}
    for name, model in logreg_models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_bin = (y_prob >= 0.5).astype(int)
        logreg_results[name] = {
            "log_loss": float(log_loss(y_test, y_prob)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "f1": float(f1_score(y_test, y_bin, zero_division=0)),
            "n_nonzero_coefs": int(np.sum(model.named_steps["logisticregression"].coef_[0] != 0)),
        }

    summary = {"regression_view": reg_results, "classification_view": logreg_results}
    save_json("06_regularization_summary.json", summary)
    print("\n--- Regularised linear regression (regression view) ---")
    for name, m in reg_results.items():
        print(f"  {name}: RMSE={m['rmse']:.4f}, nonzero coefs={m['n_nonzero_coefs']}")
    print("\n--- Regularised logistic regression (classification view) ---")
    for name, m in logreg_results.items():
        print(f"  {name}: log_loss={m['log_loss']:.4f}, AUC={m['roc_auc']:.4f}, "
              f"F1={m['f1']:.4f}, nonzero coefs={m['n_nonzero_coefs']}")

    # --- Figures -----------------------------------------------------------
    # Coefficient paths for Lasso (computed on a subsample for speed — the
    # path shape is what matters for the visualisation, not the exact values)
    path_sample = X_train.sample(n=200_000, random_state=SEED)
    y_path = y_train.loc[path_sample.index]
    alphas = np.logspace(-4, 0, 15)
    coef_paths = []
    for a in alphas:
        lasso = Lasso(alpha=a, max_iter=5000, random_state=SEED)
        lasso.fit(path_sample, y_path)
        coef_paths.append(lasso.coef_)
    coef_paths = np.array(coef_paths)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(coef_paths.shape[1]):
        ax.plot(alphas, coef_paths[:, i], lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Alpha (regularisation strength)")
    ax.set_ylabel("Coefficient value")
    ax.set_title("Lasso coefficient paths")
    save_fig(fig, "reg_01_lasso_paths.png")

    # RMSE comparison bar chart
    fig, ax = plt.subplots()
    names = list(reg_results.keys())
    rmses = [reg_results[n]["rmse"] for n in names]
    bars = ax.bar(names, rmses, color=["#ef4444", "#3b82f6", "#10b981"])
    for b, v in zip(bars, rmses):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom")
    ax.set_title("Regularised linear regression — RMSE comparison")
    ax.set_ylabel("RMSE")
    save_fig(fig, "reg_02_rmse_comparison.png")

    # AUC comparison bar chart
    fig, ax = plt.subplots()
    names = list(logreg_results.keys())
    aucs = [logreg_results[n]["roc_auc"] for n in names]
    bars = ax.bar(names, aucs, color=["#ef4444", "#3b82f6", "#10b981"])
    for b, v in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom")
    ax.set_title("Regularised logistic regression — ROC-AUC comparison")
    ax.set_ylabel("ROC-AUC")
    save_fig(fig, "reg_03_auc_comparison.png")

    print("\nRegularization complete.")


if __name__ == "__main__":
    main()