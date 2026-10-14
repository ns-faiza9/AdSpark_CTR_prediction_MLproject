"""Stage 14 — Validation Strategies (CO5).

Evaluates cross-validation partitioning methods (K-Fold, Stratified K-Fold,
and Nested CV) to ensure unbiased evaluation without data leakage.

Produces:
    analysis/output/14_validation_summary.json
"""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, cross_val_score

from common import load_processed, save_json, TARGET, SEED


def main() -> None:
    print("Stage 14: Validation Strategies")
    df = load_processed()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Sample 20,000 records for robust, fast CV without memory thrashing
    X_sample = X.sample(n=min(20000, len(X)), random_state=SEED)
    y_sample = y.loc[X_sample.index]

    base_model = LGBMClassifier(
        n_estimators=30,
        max_depth=5,
        random_state=SEED,
        verbosity=-1,
        n_jobs=-1,
    )

    # 1. Standard K-Fold
    cv_kfold = cross_val_score(
        base_model, X_sample, y_sample,
        cv=KFold(5, shuffle=True, random_state=SEED),
        scoring='roc_auc'
    )

    # 2. Stratified K-Fold
    cv_skfold = cross_val_score(
        base_model, X_sample, y_sample,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        scoring='roc_auc'
    )

    # 3. Genuine Nested Cross-Validation (5 Outer Folds / 3 Inner Folds)
    inner_cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    param_grid = {"max_depth": [4, 6]}
    grid_search = GridSearchCV(
        base_model,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='roc_auc',
        n_jobs=-1
    )
    nested_cv_scores = cross_val_score(
        grid_search, X_sample, y_sample,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        scoring='roc_auc'
    )

    kfold_auc = float(np.mean(cv_kfold))
    skfold_auc = float(np.mean(cv_skfold))
    nested_auc = float(np.mean(nested_cv_scores))

    print(f"  Standard 5-Fold AUC: {kfold_auc:.4f}")
    print(f"  Stratified 5-Fold AUC: {skfold_auc:.4f}")
    print(f"  Nested CV (5 Outer / 3 Inner) AUC: {nested_auc:.4f}")

    summary = {
        "split_ratio": {"train": 70, "validation": 15, "test": 15},
        "kfold_mean_auc": round(kfold_auc, 4),
        "stratified_kfold_mean_auc": round(skfold_auc, 4),
        "nested_cv_auc": round(nested_auc, 4),
        "optimism_bias_reduction": "Nested CV separates inner tuning folds from outer evaluation folds to completely prevent hyperparameter overfitting."
    }
    save_json("14_validation_summary.json", summary)
    print("Validation strategies benchmark complete.")


if __name__ == "__main__":
    main()
