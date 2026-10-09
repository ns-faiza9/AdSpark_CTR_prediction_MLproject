"""AdSpark CTR Prediction — Flask Web Application.

Covers Machine Learning Course Outcomes CO1 through CO5 and Advanced CTR Mathematics
across 24 modular tabs and deep theoretical foundations:
Supervised Learning, Unsupervised Clustering, Dimensionality Reduction, Anomaly Detection,
Validation Strategies, Imbalanced Metrics, Probability Calibration, McNemar Significance,
Learning Curves, SHAP Explainability, Google FTRL-Proximal, Factorization Machines (FM),
Empirical Bayes Smoothing, and Negative Downsampling Calibration.
"""
import json
import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

from ctr_predictor import predict_ctr
import ml_engine

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "analysis" / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"

app = Flask(__name__, template_folder="templates", static_folder="static")


def load_summary(filename: str) -> dict:
    """Helper to safely load summary JSON files."""
    return ml_engine.load_json(filename)


@app.context_processor
def inject_global_data():
    """Inject global variables into all templates."""
    ensemble_summary = load_summary("08_ensemble_summary.json")
    data_summary = load_summary("01_data_loading_summary.json")
    return {
        "global_dataset_rows": data_summary.get("rows", 1010724),
        "global_click_rate": round(data_summary.get("click_rate", 0.1694) * 100, 2),
        "global_best_model": ensemble_summary.get("best_model", "XGBoost"),
        "global_best_auc": ensemble_summary.get("models", {}).get("XGBoost", {}).get("roc_auc", 0.7397),
    }


# =============================================================================
# 00. Overview Dashboard (4-Card Grid + 12-Model Leaderboard)
# =============================================================================
@app.route("/")
def dashboard():
    cards = ml_engine.get_executive_summary_cards()
    ensemble_summary = load_summary("08_ensemble_summary.json")
    models_data = ensemble_summary.get("models", {})
    leaderboard = ml_engine.get_full_leaderboard()

    return render_template(
        "dashboard.html",
        active_page="dashboard",
        card_1=cards["card_1"],
        card_2=cards["card_2"],
        card_3=cards["card_3"],
        card_4=cards.get("card_4", {}),
        models_data=models_data,
        leaderboard=leaderboard,
        ensemble_summary=ensemble_summary,
    )


# =============================================================================
# SUPERVISED LEARNING MODULES (CO1, CO2, CO3)
# =============================================================================
@app.route("/data-loading")
def data_loading():
    return render_template("data_loader.html", active_page="data_loading", summary=load_summary("01_data_loading_summary.json"))

@app.route("/api/sample-data")
def api_sample_data():
    import pandas as pd
    import sys
    sys.path.append("analysis")
    from common import TRAIN_SAMPLE
    try:
        df = pd.read_csv(TRAIN_SAMPLE, nrows=50)
        import json
        return app.response_class(
            response=json.dumps(df.to_dict(orient="records"), sort_keys=False),
            status=200,
            mimetype="application/json"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/eda")
def eda():
    return render_template("eda.html", active_page="eda", summary=load_summary("02_eda_summary.json"))

@app.route("/preprocessing")
@app.route("/feature-engineering")
def feature_engineering():
    return render_template("feature_engineering.html", active_page="feature_engineering", summary=load_summary("03_feature_engineering_summary.json"))

@app.route("/linear-regression")
def linear_regression():
    return render_template("linear_regression.html", active_page="linear_regression", summary=load_summary("04_linear_regression_summary.json"))

@app.route("/logistic-regression")
def logistic_regression():
    return render_template("logistic_regression.html", active_page="logistic_regression", summary=load_summary("05_logistic_regression_summary.json"))

@app.route("/regularization")
def regularization():
    return render_template("regularization.html", active_page="regularization", summary=load_summary("06_regularization_summary.json"))

@app.route("/decision-tree")
def decision_tree():
    return render_template("decision_tree.html", active_page="decision_tree", summary=load_summary("07_decision_tree_summary.json"))

@app.route("/ensemble")
def ensemble():
    return render_template("ensemble.html", active_page="ensemble", summary=load_summary("08_ensemble_summary.json"))


# =============================================================================
# UNSUPERVISED, DIMENSIONALITY & ANOMALY MODULES (CO4)
# =============================================================================
@app.route("/kmeans")
def kmeans():
    return render_template("kmeans.html", active_page="kmeans", summary=load_summary("09_kmeans_summary.json"))

@app.route("/hierarchical")
def hierarchical():
    return render_template("hierarchical.html", active_page="hierarchical", summary=load_summary("10_hierarchical_summary.json"))

@app.route("/dbscan")
def dbscan():
    return render_template("dbscan.html", active_page="dbscan", summary=load_summary("11_dbscan_summary.json"))

@app.route("/dimensionality")
def dimensionality():
    return render_template("dimensionality.html", active_page="dimensionality", summary=load_summary("12_dimensionality_summary.json"))

@app.route("/anomaly")
def anomaly():
    return render_template("anomaly.html", active_page="anomaly", summary=load_summary("13_anomaly_summary.json"))


# =============================================================================
# VALIDATION, METRICS & CALIBRATION (CO5)
# =============================================================================
@app.route("/validation")
def validation():
    return render_template("validation.html", active_page="validation", summary=load_summary("14_validation_summary.json"))

@app.route("/imbalanced-metrics")
def imbalanced_metrics():
    return render_template("imbalanced_metrics.html", active_page="imbalanced_metrics", summary=load_summary("15_imbalanced_summary.json"))

@app.route("/calibration")
def calibration():
    return render_template("calibration.html", active_page="calibration", summary=load_summary("16_calibration_summary.json"))

@app.route("/significance")
def significance():
    return render_template("significance.html", active_page="significance", summary=load_summary("17_significance_summary.json"))


# =============================================================================
# DIAGNOSTICS & EXPLAINABILITY (CO3, CO5)
# =============================================================================
@app.route("/learning-curves")
def learning_curves():
    return render_template("learning_curves.html", active_page="learning_curves", summary=load_summary("18_learning_curves_summary.json"))

@app.route("/explainability")
def explainability():
    return render_template("explainability.html", active_page="explainability", summary=load_summary("19_explainability_summary.json"))


# =============================================================================
# INTERACTIVE TOOL & API ENDPOINTS
# =============================================================================
@app.route("/predict")
def predict_page():
    return render_template("predict.html", active_page="predict")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json() or request.form.to_dict()
        result = predict_ctr(data)
        return jsonify({"success": True, "prediction": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/modules/eda")
def api_modules_eda():
    """API endpoint for 15-Task EDA module data."""
    summary = load_summary("02_eda_summary.json")
    return jsonify({"success": True, "data": summary})

@app.route("/api/summary/<filename>")
def api_summary(filename: str):
    if not filename.endswith(".json"):
        filename += ".json"
    summary = load_summary(filename)
    return jsonify(summary)

@app.route("/figures/<path:filename>")
def serve_figures(filename: str):
    return send_from_directory(FIGURES_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting AdSpark Flask Web App on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
