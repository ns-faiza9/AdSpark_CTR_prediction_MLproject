"""Run every stage of the AdSpark analysis pipeline in order (00 through 19)."""
import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STAGES = [
    "00_prepare_data.py",
    "01_data_loading.py",
    "02_eda.py",
    "03_feature_engineering.py",
    "04_linear_regression.py",
    "05_logistic_regression.py",
    "06_regularization.py",
    "07_decision_tree.py",
    "08_ensemble.py",
    "09_kmeans.py",
    "10_hierarchical.py",
    "11_dbscan.py",
    "12_dimensionality.py",
    "13_anomaly.py",
    "14_validation.py",
    "15_imbalanced.py",
    "16_calibration.py",
    "17_significance.py",
    "18_learning_curves.py",
    "19_explainability.py",
]


def main() -> None:
    for stage in STAGES:
        print(f"\n{'=' * 70}\nRunning {stage}\n{'=' * 70}")
        code = subprocess.call([sys.executable, os.path.join(BASE_DIR, stage)])
        if code != 0:
            print(f"FAILED: {stage} (exit code {code})")
            sys.exit(code)
    print("\nAll 19 analysis stages completed successfully.")


if __name__ == "__main__":
    main()