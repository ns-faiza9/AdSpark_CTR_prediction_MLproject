"""Stage 16 — Probability Calibration (CO5).

Evaluates model confidence calibration via Reliability Diagrams and compares
Uncalibrated vs Platt Scaling vs Isotonic Regression calibration.

Produces:
    analysis/output/16_calibration_summary.json
    analysis/output/figures/cal_01_reliability_diagrams.png
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from common import load_processed, save_fig, save_json, TARGET, SEED


def compute_ece(y_true, y_prob, n_bins=10) -> float:
    """Calculate true empirical Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return float(ece)


def main() -> None:
    print("Stage 16: Probability Calibration")
    df = load_processed()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Subsample for fast, stable isotonic fitting
    sample_df = df.sample(n=min(50000, len(df)), random_state=SEED)
    X_s = sample_df.drop(columns=[TARGET])
    y_s = sample_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y_s, test_size=0.2, random_state=SEED, stratify=y_s
    )

    base_model = LogisticRegression(max_iter=300, random_state=SEED)
    base_model.fit(X_train, y_train)
    y_prob_uncal = base_model.predict_proba(X_test)[:, 1]

    # Platt scaling (Sigmoid)
    platt_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
    platt_model.fit(X_train, y_train)
    y_prob_platt = platt_model.predict_proba(X_test)[:, 1]

    # Isotonic calibration
    iso_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
    iso_model.fit(X_train, y_train)
    y_prob_iso = iso_model.predict_proba(X_test)[:, 1]

    prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, y_prob_uncal, n_bins=8)
    prob_true_iso, prob_pred_iso = calibration_curve(y_test, y_prob_iso, n_bins=8)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k:', label='Perfect Calibration')
    ax.plot(prob_pred_uncal, prob_true_uncal, 's-', color='#f87171', label='Uncalibrated Model')
    ax.plot(prob_pred_iso, prob_true_iso, 'o-', color='#34d399', label='Isotonic Calibrated')
    ax.set_title("Reliability Diagram (Probability Calibration)")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Clicks (Observed CTR)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "cal_01_reliability_diagrams.png")

    brier_before = float(brier_score_loss(y_test, y_prob_uncal))
    brier_platt = float(brier_score_loss(y_test, y_prob_platt))
    brier_iso = float(brier_score_loss(y_test, y_prob_iso))

    ece_before = compute_ece(y_test.values, y_prob_uncal)
    ece_after_iso = compute_ece(y_test.values, y_prob_iso)
    ece_red_pct = round((1.0 - (ece_after_iso / max(ece_before, 1e-6))) * 100, 1)

    summary = {
        "brier_score_before": round(brier_before, 4),
        "brier_score_platt": round(brier_platt, 4),
        "brier_score_isotonic": round(brier_iso, 4),
        "ece_before": round(ece_before, 4),
        "ece_after_isotonic": round(ece_after_iso, 4),
        "ece_reduction_pct": ece_red_pct,
        "conclusion": f"Isotonic Regression reduces Expected Calibration Error (ECE) by {ece_red_pct}%."
    }
    save_json("16_calibration_summary.json", summary)
    print(f"Probability calibration complete. ECE reduced from {ece_before:.4f} to {ece_after_iso:.4f} ({ece_red_pct}% drop).")


if __name__ == "__main__":
    main()
