# ⚡ AdSpark — Ad Click-Through Rate (CTR) Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-green.svg?logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AdSpark** is an end-to-end Machine Learning web application and data analysis project built to predict whether an online ad impression will be clicked or not. Trained on over **1 million mobile ad events** from the canonical **Avazu CTR benchmark dataset**, the project covers the full data science lifecycle—from exploratory analysis and feature engineering to supervised model comparison, unsupervised customer segmentation, dimensionality reduction, and a live web application with real-time predictions.

---

## 🌟 Key Project Highlights

* **Real-World Scale:** Trained on a representative 1,010,724-row sample extracted from the 40.4-million-event Kaggle Avazu CTR dataset.
* **10 Machine Learning Models:** Evaluated and compared algorithms from baseline linear models to high-performance tree ensembles (XGBoost, LightGBM, Random Forest, AdaBoost).
* **Consistent 83%+ Accuracy:** All classification algorithms achieve realistic, production-aligned testing accuracy (~83.1% to 83.3%) matching the natural ad impression distribution.
* **Unsupervised Clustering:** Uncovered natural audience groups using **K-Means (3 clusters)**, **DBSCAN (14 density clusters)**, and **Hierarchical Clustering (Ward linkage)**.
* **Dimensionality Reduction:** Compressed 23 raw features into 14 principal components while retaining **90% of total information**, visualized through PCA and t-SNE manifold projections.
* **Interactive Web Dashboard:** A responsive Flask web application featuring 22 dedicated analytical pages and an interactive **Live CTR Predictor Sandbox**.

---

## 📊 Dataset Overview

The project uses mobile ad impression data from the **Avazu CTR Prediction Challenge**:

| Characteristic | Detail |
| :--- | :--- |
| **Total Raw Dataset** | ~40.4 Million impressions (6.0 GB) across 10 days of mobile ad activity |
| **Active Project Sample** | **1,010,724 rows** (statistically stratified 2.5% sample) |
| **Features & Columns** | **24 columns** (device info, banner positions, site/app IDs, timestamp) |
| **Overall Click Rate (CTR)** | **16.94%** (171,209 clicks vs. 839,515 non-clicks) |
| **Train / Test Split** | **80% Training** (808,579 rows) / **20% Testing** (202,145 rows) |
| **Validation Strategy** | Compared against a 3-way 70/15/15 holdout split and 5-fold Stratified / Nested Cross-Validation |

---

## 🏆 Model Performance Leaderboard

All models were evaluated on the exact same held-out 20% test partition (202,145 impressions). Tree-based ensembles deliver the highest discrimination power while maintaining consistent accuracy:

| Rank | Model Name | Model Family | Testing Accuracy | ROC-AUC | Log-Loss | Primary Advantage |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **🥇 #1** | **XGBoost Classifier** | Boosted Trees | **83.29%** | **0.7202** | **0.3951** | Best overall discrimination on non-linear ad interactions |
| **🥈 #2** | **Gradient Boosting (GBM)** | Sequential Ensemble | **83.32%** | **0.7200** | **0.3991** | High accuracy and robust variance reduction |
| **🥉 #3** | **LightGBM Classifier** | Histogram GBDT | **83.29%** | **0.7197** | **0.3960** | Sub-millisecond inference latency; ideal for live bidding |
| **#4** | **Random Forest** | Bagging Ensemble | **83.27%** | **0.7111** | **0.4018** | High stability and robustness against noisy features |
| **#5** | **Decision Tree (Pruned)** | Single CART Tree | **83.10%** | **0.6705** | **0.4450** | Clear, interpretable if-else decision rules (depth = 5) |
| **#6** | **AdaBoost Classifier** | Adaptive Boosting | **82.88%** | **0.6650** | **0.4105** | Sequential re-weighting of challenging impressions |
| **#7** | **Logistic Regression (L2)** | Generalized Linear | **83.05%** | **0.6468** | **0.4336** | Fast, convex baseline at standard decision threshold |
| **#8** | **Logistic Regression (L1)** | Sparse Lasso Logit | **83.05%** | **0.6468** | **0.6566** | Automatic feature selection via L1 weight shrinkage |
| **#9** | **Ridge Regression** | Continuous Baseline | **83.05%** | **0.6460** | **0.4355** | Closed-form regularized baseline |
| **#10** | **Linear Regression (OLS)** | Unregularized Linear | **83.05%** | **0.6460** | **0.4355** | Baseline continuous benchmark |

> **Evaluation Note:** In ad click prediction, ~83.06% of impressions are non-clicks. Evaluating models at the standard decision threshold (0.50) yields authentic ~83.1% to 83.3% accuracy. If the threshold is adjusted to 0.178 to prioritize catching rare clicks, the model catches 58.8% of clicks with a balanced accuracy of 61.5%.

---

## 🔄 End-to-End Machine Learning Pipeline

```text
[ Raw Avazu Data: 1.01M Rows ]
               │
               ▼
[ 1. Data Cleaning & Feature Engineering ]
   • Extracted hour of day and cyclical time signals (sin/cos encoding)
   • Target frequency encodings for high-cardinality IDs (device IP, site ID, app ID)
   • Standardized numeric features with StandardScaler
               │
               ▼
[ 2. Supervised Learning & Model Comparison ]
   • Linear baselines: OLS, Ridge, Logistic Regression (L1 & L2)
   • Decision Trees: CART pruning & max-depth hyperparameter sweeps (1 to 10)
   • Ensembles: Random Forest, AdaBoost, Gradient Boosting, LightGBM, XGBoost
               │
               ▼
[ 3. Unsupervised Clustering & Dimensionality Reduction ]
   • K-Means: Optimal k=3 clusters identified via WCSS elbow & silhouette curves
   • DBSCAN: 14 density-based interaction clusters discovered; 18.3% noise isolated
   • Hierarchical: Bottom-up agglomerative dendrograms comparing 4 linkage types
   • PCA: 23 features compressed to 14 components retaining 90% information
   • t-SNE: Multi-perplexity non-linear manifold projections (10, 30, 50)
               │
               ▼
[ 4. Validation, Probability Calibration & Explainability ]
   • Validation strategies: 70/15/15 3-way split vs. Stratified & Nested 5-fold CV
   • Probability calibration: Isotonic regression dropped error (ECE) by 85.2%
   • Statistical significance: McNemar test confirmed ensemble superiority (p < 0.05)
   • Feature attribution: Identified key drivers (device model, site domain, hour)
               │
               ▼
[ 5. Interactive Flask Web Application & Real-Time Prediction Sandbox ]
```

---

## 🌐 Interactive Web Application

The project includes a complete Flask web dashboard with **22 responsive pages**:

* **Overview Dashboard (`/`):** Executive summary cards, dataset metrics, and the full 10-model comparative leaderboard.
* **Data Ingestion & Inspection (`/data-loading`):** Memory usage analysis, column data types, missing value checks, and an interactive 50-row data viewer.
* **Exploratory Data Analysis (`/eda`):** 15 visual statistical steps analyzing hourly click patterns, banner locations, device categories, and site domains.
* **Preprocessing & Features (`/feature-engineering`):** Step-by-step breakdown of frequency encoding and cyclical time transformations.
* **Linear & Logistic Models (`/linear-regression`, `/logistic-regression`, `/regularization`):** ROC curves, confusion matrices, and L1 vs. L2 weight shrinkage.
* **Decision Trees & Ensembles (`/decision-tree`, `/ensemble`):** Tree visualization diagrams, depth sweeps, and multi-model ROC comparisons.
* **Unsupervised Clustering (`/kmeans`, `/hierarchical`, `/dbscan`):** 2D spatial cluster scatter plots, centroids, and market segmentation breakdown tables.
* **Dimensionality Reduction (`/dimensionality`):** PCA Scree plots, 2D feature biplots, factor loadings heatmap, and t-SNE manifold projections.
* **Validation & Calibration (`/validation`, `/calibration`, `/significance`, `/learning-curves`):** 70/15/15 vs. K-Fold CV, reliability diagrams, and learning curves.
* **Live Sandbox (`/predict`):** Interactive form where users input ad details (banner position, site category, app, device type, hour) and receive an instant click probability with key decision drivers.

---

## 💻 Tech Stack

* **Programming Language:** Python 3.10+
* **Web Framework:** Flask, Jinja2, HTML5, CSS3, JavaScript
* **Machine Learning & Modeling:** Scikit-Learn, LightGBM, XGBoost
* **Data Processing & Analysis:** Pandas, NumPy, SciPy
* **Data Visualization:** Matplotlib, Seaborn
* **Model Serialization:** Joblib

---

## 📁 Project Directory Structure

```text
AdSpark_CTR Prediction/
├── app.py                         # Flask web application entry point
├── ctr_predictor.py               # Live real-time prediction engine
├── ml_engine.py                   # Backend data loader & dynamic leaderboard
├── templates/                     # 22 HTML templates (dashboard, models, sandbox)
├── static/                        # CSS styles and JavaScript assets
├── analysis/                      # 19 standalone Python ML pipeline scripts
│   ├── common.py                  # Shared data loaders, paths, and plot helpers
│   ├── 01_data_loading.py         # Stage 01: Ingestion & inspection
│   ├── 02_eda.py                  # Stage 02: 15-task exploratory analysis
│   ├── 03_feature_engineering.py  # Stage 03: Feature transformations
│   ├── 04_linear_regression.py    # Stage 04: Linear regression baseline
│   ├── 05_logistic_regression.py  # Stage 05: Logistic regression classification
│   ├── 06_regularization.py       # Stage 06: L1/L2 regularization
│   ├── 07_decision_tree.py        # Stage 07: Pruned CART decision trees
│   ├── 08_ensemble.py             # Stage 08: Tree ensembles (RF, Ada, GBM, LGBM, XGB)
│   ├── 09_kmeans.py               # Stage 09: K-Means clustering (k=3)
│   ├── 10_hierarchical.py         # Stage 10: Hierarchical clustering dendrograms
│   ├── 11_dbscan.py               # Stage 11: DBSCAN density clustering
│   ├── 12_dimensionality.py       # Stage 12: PCA scree, biplot, and t-SNE
│   ├── 13_anomaly.py              # Stage 13: Isolation Forest & One-Class SVM
│   ├── 14_validation.py           # Stage 14: 70/15/15 & Cross-validation strategies
│   ├── 15_imbalanced.py           # Stage 15: Imbalanced metrics & PR curves
│   ├── 16_calibration.py          # Stage 16: Probability calibration & ECE
│   ├── 17_significance.py         # Stage 17: McNemar statistical testing
│   ├── 18_learning_curves.py      # Stage 18: Sample complexity & bias-variance
│   ├── 19_explainability.py       # Stage 19: SHAP feature importances
│   ├── output/                    # Generated JSON metrics & trained champion model
│   └── output/figures/            # High-resolution generated charts and plots
└── README.md                      # Project documentation
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/ns-faiza9/AdSpark_CTR_prediction_MLproject.git
cd "AdSpark_CTR Prediction"
```

### 2. Install Required Packages
```bash
pip install -r analysis/requirements.txt
```

### 3. Start the Web App
```bash
python app.py
```

### 4. View in Browser
Open your browser and visit:
```text
http://127.0.0.1:5000
```
Explore all 22 dashboard tabs and test live ad impressions on the `/predict` page.

---

## 🔮 Live Prediction API Example

The web app includes a REST endpoint at `/api/predict` for instant predictions:

**Sample Request (POST `/api/predict`):**
```json
{
  "hour": 14,
  "banner_pos": "0",
  "site_category": "28905ebd",
  "app_category": "07d7d19e",
  "device_type": "1",
  "device_conn_type": "0",
  "C1": "1005",
  "C15": "320",
  "C16": "50",
  "C18": "0",
  "site_id": "1fbe01fe",
  "app_id": "ecad2386",
  "device_model": "8a487514"
}
```

**Sample Response:**
```json
{
  "success": true,
  "prediction": {
    "model_selected": "Trained Champion Ensemble (LightGBM / XGBoost)",
    "predicted_click": 1,
    "prediction_label": "Will Click (1)",
    "ctr_probability": 0.2706,
    "ctr_percentage": 27.06,
    "confidence_tier": "High Click Probability",
    "tier_color": "green",
    "wilson_ci_95": {
      "lower": 24.39,
      "upper": 29.89
    }
  }
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.