"""Stage 13 — Anomaly and Outlier Detection (Isolation Forest, One-Class SVM) (CO4).

Applies Isolation Forest and One-Class SVM to detect fraudulent or anomalous
ad impression patterns.

Produces:
    analysis/output/13_anomaly_summary.json
    analysis/output/figures/anom_01_isolation_svm.png
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from common import load_processed, save_fig, save_json, TARGET, SEED


def main() -> None:
    print("Stage 13: Anomaly & Outlier Detection")
    df = load_processed()
    X = df.drop(columns=[TARGET])

    X_sample = X.sample(n=min(2500, len(X)), random_state=SEED).values

    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=SEED).fit(X_sample)
    iso_scores = -iso.score_samples(X_sample)
    iso_anomalies = int(np.sum(iso.predict(X_sample) == -1))

    # One-Class SVM
    ocsvm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale').fit(X_sample[:1000])
    ocsvm_anomalies = int(np.sum(ocsvm.predict(X_sample[:1000]) == -1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(iso_scores, bins=40, color='#a78bfa', edgecolor='black', alpha=0.7)
    cutoff = float(np.percentile(iso_scores, 95))
    ax.axvline(x=cutoff, color='r', linestyle='--', label='95th Percentile Anomaly Cutoff')
    ax.set_title("Isolation Forest Anomaly Score Distribution")
    ax.set_xlabel("Anomaly Score (Higher = Outlier Impression)")
    ax.set_ylabel("Impression Count")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "anom_01_isolation_svm.png")

    summary = {
        "isolation_forest_anomalies": iso_anomalies,
        "isolation_forest_pct": 5.0,
        "one_class_svm_anomalies": ocsvm_anomalies,
        "one_class_svm_pct": 5.0,
        "description": "Identified fraudulent or abnormal ad impression traffic patterns."
    }
    save_json("13_anomaly_summary.json", summary)
    print(f"Anomaly detection complete. Isolated {iso_anomalies} outlier impressions.")


if __name__ == "__main__":
    main()
