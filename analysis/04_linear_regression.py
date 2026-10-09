"""Stage 04 — Linear Regression Analysis (Simple & Regularized fits).

Fits linear models (OLS, Ridge L2, Lasso L1) on top predictors and multivariate data.
Evaluates metrics (Train/Test MSE, Train/Test R2, ROC-AUC, Log Loss) and plots the
Fitted Regression Line scatter plot for each penalty option.

Produces:
    analysis/output/04_linear_regression_summary.json
    analysis/output/figures/lr_fitted_ols.png
    analysis/output/figures/lr_fitted_ridge.png
    analysis/output/figures/lr_fitted_lasso.png
    analysis/output/figures/lr_01_coefficients.png
    analysis/output/figures/lr_02_prediction_distribution.png

Usage:
    python analysis/04_linear_regression.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, 
)
from sklearn.model_selection import train_test_split

from common import load_processed, save_fig, save_json, TARGET, SEED


def main() -> None:
    df = load_processed()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"Features: {X.shape[1]}, Rows: {len(df):,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # 1. Multivariate OLS Baseline
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_clip = np.clip(y_pred, 1e-6, 1 - 1e-6)
    y_bin = (y_pred >= 0.5).astype(int)

    summary = {
        "mse": float(mean_squared_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "log_loss": float(log_loss(y_test, y_pred_clip)),
        "accuracy": float(accuracy_score(y_test, y_bin)),
        "precision": float(precision_score(y_test, y_bin, zero_division=0)),
        "recall": float(recall_score(y_test, y_bin, zero_division=0)),
        "f1": float(f1_score(y_test, y_bin, zero_division=0)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "penalties": {},
    }

    # Select primary predictor feature for Simple Linear Regression fit (e.g. C18_enc)
    feature = "C18_enc" if "C18_enc" in X.columns else X.columns[0]
    X_tr_feat = X_train[[feature]]
    X_te_feat = X_test[[feature]]

    penalty_specs = [
        ("None (Ordinary Least Squares)", LinearRegression(), "lr_fitted_ols.png", "None (OLS)"),
        ("L2 (Ridge Regression)", Ridge(alpha=1.0, random_state=SEED), "lr_fitted_ridge.png", "L2 (Ridge)"),
        ("L1 (Lasso Regression)", Lasso(alpha=0.001, random_state=SEED), "lr_fitted_lasso.png", "L1 (Lasso)"),
    ]

    # Sample points for scatter plot visualization
    scatter_df = df.sample(n=1200, random_state=SEED)
    x_scat = scatter_df[feature].values
    y_scat = scatter_df[TARGET].values

    for name, reg_model, fig_name, label_short in penalty_specs:
        reg_model.fit(X_tr_feat, y_train)

        # Get intercept and slope
        if hasattr(reg_model, "coef_"):
            slope = float(reg_model.coef_[0] if reg_model.coef_.ndim == 1 else reg_model.coef_[0][0])
        else:
            slope = 0.0
        intercept = float(reg_model.intercept_)

        tr_pred = reg_model.predict(X_tr_feat)
        te_pred = reg_model.predict(X_te_feat)

        tr_mse = float(mean_squared_error(y_train, tr_pred))
        te_mse = float(mean_squared_error(y_test, te_pred))
        tr_r2 = float(r2_score(y_train, tr_pred))
        te_r2 = float(r2_score(y_test, te_pred))

        eq_str = f"Click Rate = {intercept:.4f} + {slope:.4f} * {feature}"

        summary["penalties"][name] = {
            "label": name,
            "feature": feature,
            "intercept": round(intercept, 4),
            "slope": round(slope, 4),
            "equation": eq_str,
            "train_mse": round(tr_mse, 5),
            "test_mse": round(te_mse, 5),
            "train_r2": round(tr_r2, 4),
            "test_r2": round(te_r2, 4),
            "sample_size": int(len(X_train)),
            "figure": fig_name,
        }

        # Generate Fitted Regression Line Scatter Plot
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.scatter(x_scat, y_scat, alpha=0.35, color="#3b82f6", s=18, label="Actual Students / Impressions")

        x_grid = np.linspace(X[feature].min(), X[feature].max(), 200)
        y_grid = intercept + slope * x_grid
        ax.plot(x_grid, y_grid, color="#dc2626", lw=2.5, label=f"Fit ({label_short}): y = {intercept:.2f} + {slope:.2f}*x")

        ax.set_title(f"Simple Linear Regression: Click Rate vs {feature} ({label_short})", fontsize=11, fontweight="bold")
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel("Click Rate / Probability", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
        fig.tight_layout()
        save_fig(fig, fig_name)

    # 2. Save JSON summary
    save_json("04_linear_regression_summary.json", summary)

    # 3. Coefficient magnitude plot
    coefs = pd.Series(model.coef_, index=X.columns).sort_values(key=np.abs, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(coefs.head(20).index, coefs.head(20).values, color="#6366f1")
    ax.set_title("Top 20 linear regression coefficients (by |value|)")
    ax.set_xlabel("Coefficient")
    fig.tight_layout()
    save_fig(fig, "lr_01_coefficients.png")

    # 4. Predicted vs actual distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(y_pred, bins=80, color="#6366f1", alpha=0.7, label="Predicted")
    ax.axvline(0.5, color="red", ls="--", label="Threshold 0.5")
    ax.set_title("Distribution of predicted click probabilities")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "lr_02_prediction_distribution.png")

    print("\nLinear regression analysis complete.")


if __name__ == "__main__":
    main()