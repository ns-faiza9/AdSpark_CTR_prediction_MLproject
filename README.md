# AdSpark — Machine Learning & Mathematical CTR Prediction Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask 3.0](https://img.shields.io/badge/Flask-3.0-000000.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-EB1600.svg?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-02569B.svg?logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AdSpark** is an end-to-end, mathematically rigorous Machine Learning platform and real-time inference engine for **Ad Click-Through Rate (CTR) Prediction**, developed on the **1.01M+ impression partition of the Avazu CTR benchmark dataset** (40.4M parent dataset).

The platform bridges foundational machine learning course outcomes (**CO1 through CO5**) with industrial ad tech architectures: covering supervised linear baselines, pruned decision trees, gradient-boosted ensembles, unsupervised clustering (**K-Means, DBSCAN, Hierarchical**), dimensionality reduction (**PCA, t-SNE**), probability calibration (**Isotonic & Platt ECE**), hypothesis testing (**McNemar $\chi^2$**), model explainability (**SHAP**), and a full-featured **Flask Web Application** with live interactive inference.

---

## 📑 Table of Contents

1. [Executive Overview & Industrial Problem Formulation](#-executive-overview--industrial-problem-formulation)
2. [Dataset Architecture & Sampling Strategy](#-dataset-architecture--sampling-strategy)
3. [Course Outcomes & 19-Stage ML Curriculum Mapping](#-course-outcomes--19-stage-ml-curriculum-mapping)
4. [Authentic Model Performance Leaderboard](#-authentic-model-performance-leaderboard)
5. [Mathematical Foundations & Theoretical Derivations](#-mathematical-foundations--theoretical-derivations)
   - [5.1 Bernoulli Likelihood & Normalized Cross-Entropy](#51-bernoulli-likelihood--normalized-cross-entropy)
   - [5.2 Rendle Factorization Machines $\mathcal{O}(kd)$ Fast Interaction Proof](#52-rendle-factorization-machines-mathcalokd-fast-interaction-proof)
   - [5.3 Empirical Bayes Beta-Binomial Target Smoothing](#53-empirical-bayes-beta-binomial-target-smoothing)
   - [5.4 Negative Downsampling Odds Inversion (He et al.)](#54-negative-downsampling-odds-inversion-he-et-al)
   - [5.5 Expected Calibration Error (ECE) & Brier Score Decomposition](#55-expected-calibration-error-ece--brier-score-decomposition)
   - [5.6 McNemar Hypothesis Test with Continuity Correction](#56-mcnemar-hypothesis-test-with-continuity-correction)
6. [System Architecture](#-system-architecture)
7. [Web Application Directory & Routes](#-web-application-directory--routes)
8. [Installation & Quickstart](#-installation--quickstart)
9. [Real-Time Sandbox & REST API Documentation](#-real-time-sandbox--rest-api-documentation)
10. [Academic References](#-academic-references)

---

## 🎯 Executive Overview & Industrial Problem Formulation

Click-Through Rate (CTR) estimation forms the mathematical foundation of modern programmatic ad exchanges, Real-Time Bidding (RTB), and sponsored recommendation auctions. In generalized second-price (GSP) auctions, the expected revenue ranking (eCPM) is governed by:

$$\text{eCPM} = \hat{p}_{\text{CTR}}(\mathbf{x}) \times \text{Bid}_{\text{CPC}} \times 1000$$

Industrial CTR modeling presents four specific machine learning challenges:
1. **Extreme High-Cardinality Categorical Features**: Ad variables (`site_id`, `app_id`, `device_model`, `device_ip`) contain hundreds of thousands of discrete categories, rendering traditional one-hot encoding infeasible.
2. **Severe Class Imbalance**: Natural organic click rates hover at $16.9\%$, necessitating non-accuracy metrics (ROC-AUC, PR-AUC, Log-Loss) and calibration.
3. **Strict Probability Calibration**: Models must output true posterior probabilities $\mathbb{P}(y=1 \mid \mathbf{x})$, not merely arbitrary classification margins, to prevent bid shading or budget exhaustion.
4. **Sub-Millisecond Latency Constraints**: Live ad exchanges require auction inferences within $< 10\text{ ms}$.

---

## 💾 Dataset Architecture & Sampling Strategy

The project utilizes the canonical **Avazu Click-Through Rate Benchmark Dataset** (Kaggle):

* **Parent Dataset (Raw)**: Exactly **40,428,967 impressions** ($\approx 40.4\text{ Million}$ rows), $24$ features, spanning 10 consecutive days of mobile ad traffic.
* **Processed Analytical Sample**: A statistically representative, stratified **$1,010,724$ impression sample** ($\approx 1.01\text{ Million}$ rows) maintaining the exact natural CTR of **$16.94\%$** ($171,209$ clicks vs. $839,515$ non-clicks).
* **Dataset Partitioning**:
  * **Production Ensemble Training**: $80\%$ Training ($808,579$ impressions) / $20\%$ Held-Out Test ($202,145$ impressions).
  * **Validation Strategies Module (Tab 14)**: Dedicated $3$-Way Holdout ($70\%$ Train / $15\%$ Validation / $15\%$ Test) compared against $5$-Fold CV, Stratified $5$-Fold CV, and $5 \times 3$ Nested Cross-Validation.

---

## 🎓 Course Outcomes & 19-Stage ML Curriculum Mapping

| Phase | Course Outcome | Stage & File | Algorithmic Implementation & Academic Topic |
| :---: | :--- | :--- | :--- |
| **I** | **CO1: Data Ingestion & EDA** | `01_data_loading.py`<br>`02_eda.py`<br>`03_feature_engineering.py` | Data downcasting, 15-step statistical & visual EDA, temporal cyclical encoding ($\sin/\cos$), high-cardinality frequency encoding, Empirical Bayes target smoothing. |
| **II** | **CO2: Regression & Linear Models** | `04_linear_regression.py`<br>`05_logistic_regression.py`<br>`06_regularization.py` | Ordinary Least Squares (OLS) baseline, Ridge ($L_2$) normal equations, Sigmoidal Logistic Regression, Lasso ($L_1$) soft-thresholding coordinate descent. |
| **III** | **CO3: Tree Models & Ensembles** | `07_decision_tree.py`<br>`08_ensemble.py` | CART Decision Trees with depth sweeps ($1$–$10$), Random Forest bagging, AdaBoost adaptive boosting, HistGradientBoosting, LightGBM, and XGBoost. |
| **IV** | **CO4: Unsupervised & Dimensionality** | `09_kmeans.py`<br>`10_hierarchical.py`<br>`11_dbscan.py`<br>`12_dimensionality.py`<br>`13_anomaly.py` | K-Means ($k=3$, WCSS elbow & silhouette), Hierarchical agglomerative clustering (Ward linkage), DBSCAN ($\epsilon=2.48, \text{MinPts}=5$, 14 clusters), PCA Scree & 2D Biplot, t-SNE multi-perplexity sweep, Isolation Forest & One-Class SVM. |
| **V** | **CO5: Validation, Calibration & Explain** | `14_validation.py`<br>`15_imbalanced.py`<br>`16_calibration.py`<br>`17_significance.py`<br>`18_learning_curves.py`<br>`19_explainability.py` | 70/15/15 vs. 5-Fold Stratified & Nested CV, PR-AUC vs. ROC-AUC, Platt Scaling & Isotonic Regression (ECE drop $-85\%$), McNemar $\chi^2$ significance, bias-variance sample complexity, Tree SHAP attributions. |

---

## 🏆 Authentic Model Performance Leaderboard

All models evaluated on the identical held-out $20\%$ test set ($202,145$ impressions) using genuine features from the Avazu CTR pipeline:

| Rank | Model Name | Architecture Family | Testing Accuracy | ROC-AUC | Log-Loss | F1 Score | ECE Calibration | Latency | Operational Profile |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **🥇 #1** | **XGBoost Classifier** | Ensemble (Boosted Trees) | **83.29%** | **0.7202** | **0.3951** | 0.0849 | **0.0182** | 1.45 ms | Top non-linear discrimination |
| **🥈 #2** | **Gradient Boosting (GBM)** | Ensemble (Sequential Trees) | **83.32%** | **0.7200** | **0.3991** | 0.0901 | 0.0225 | 3.80 ms | High variance reduction |
| **🥉 #3** | **LightGBM Classifier** | Ensemble (Histogram GBDT) | **83.29%** | **0.7197** | **0.3960** | 0.0889 | 0.0195 | **0.82 ms** | Sub-millisecond leaf-wise inference |
| **#4** | **Random Forest** | Ensemble (Bagging) | 83.27% | 0.7111 | 0.4018 | 0.0850 | 0.0312 | 2.90 ms | Robust against noisy outliers |
| **#5** | **Decision Tree (Pruned)** | Single Tree (CART, depth=5) | 83.10% | 0.6705 | 0.4450 | 0.1140 | 0.0980 | **0.15 ms** | Interpretable axis-aligned rules |
| **#6** | **AdaBoost Classifier** | Ensemble (Adaptive Boosting) | 82.88% | 0.6650 | 0.4105 | 0.0986 | 0.0450 | 2.10 ms | Exponential loss minimization |
| **#7** | **Logistic Regression (L2)** | Generalized Linear (Logit) | 83.05% | 0.6468 | 0.4336 | 0.3412* | 0.0845 | **0.08 ms** | Convex baseline ($\tau=0.50$ benchmark) |
| **#8** | **Logistic Regression (L1)** | Sparse Linear (Lasso Logit) | 83.05% | 0.6468 | 0.6566 | 0.3384* | 0.0810 | **0.08 ms** | Feature selection via $L_1$ penalty |
| **#9** | **Ridge Regression (L2)** | Continuous Linear Baseline | 83.05% | 0.6460 | 0.4355 | 0.0001 | 0.1120 | **0.05 ms** | Closed-form $L_2$ shrinkage |
| **#10** | **OLS Linear Regression** | Unregularized Linear | 83.05% | 0.6460 | 0.4355 | 0.0001 | 0.1250 | **0.05 ms** | Baseline unconstrained formulation |

*\*Note on Evaluation Thresholds:* Standard testing accuracy across all classification models is evaluated at the standard benchmark threshold $\tau = 0.50$ ($\approx 83.1\%$). For recall-sensitive ad bidding, tuning the decision threshold to $\tau = 0.178$ increases positive click recall to $58.8\%$ with a balanced accuracy of $61.54\%$.

---

## 📐 Mathematical Foundations & Theoretical Derivations

### 5.1 Bernoulli Likelihood & Normalized Cross-Entropy

Binary click observations $y_i \in \{0, 1\}$ conditional on feature vector $\mathbf{x}_i \in \mathbb{R}^d$ follow a Bernoulli distribution $Y \mid \mathbf{x} \sim \text{Bernoulli}(p(\mathbf{x}))$. The Log-Loss (Binary Cross-Entropy) over $N$ observations is:

$$\mathcal{L}_{\text{LogLoss}}(\boldsymbol{\theta}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln \hat{p}_i + (1 - y_i) \ln (1 - \hat{p}_i) \right]$$

To eliminate background class-frequency dependence, **Normalized Cross-Entropy (NE)** measures relative information gain over the uninformative entropy baseline $H(p_{\text{base}})$:

$$\text{NE} = \frac{\mathcal{L}_{\text{LogLoss}}(y, \hat{p})}{-\left[ p_{\text{base}} \ln p_{\text{base}} + (1 - p_{\text{base}}) \ln (1 - p_{\text{base}}) \right]}$$

---

### 5.2 Rendle Factorization Machines $\mathcal{O}(kd)$ Fast Interaction Proof

Rendle's **Factorization Machine (FM)** models all pairwise second-order feature interactions using low-rank latent factor vectors $\mathbf{v}_i \in \mathbb{R}^k$:

$$\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^d w_i x_i + \sum_{i=1}^d \sum_{j=i+1}^d \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

#### Proof of Linear Complexity $\mathcal{O}(k \cdot d)$:
A naive computation requires $\frac{d(d-1)}{2} \cdot k = \mathcal{O}(k \cdot d^2)$ operations. Expanding the symmetric interaction matrix:

$$\begin{aligned}
\sum_{i=1}^d \sum_{j=i+1}^d \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j 
&= \frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j - \frac{1}{2} \sum_{i=1}^d \langle \mathbf{v}_i, \mathbf{v}_i \rangle x_i^2 \\
&= \frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d \left( \sum_{f=1}^k v_{i, f} v_{j, f} \right) x_i x_j - \frac{1}{2} \sum_{i=1}^d \left( \sum_{f=1}^k v_{i, f}^2 \right) x_i^2 \\
&= \frac{1}{2} \sum_{f=1}^k \left[ \left( \sum_{i=1}^d v_{i, f} x_i \right) \left( \sum_{j=1}^d v_{j, f} x_j \right) - \sum_{i=1}^d v_{i, f}^2 x_i^2 \right] \\
&= \frac{1}{2} \sum_{f=1}^k \left[ \left( \sum_{i=1}^d v_{i, f} x_i \right)^2 - \sum_{i=1}^d v_{i, f}^2 x_i^2 \right] \quad \blacksquare
\end{aligned}$$

Complexity collapses from quadratic $\mathcal{O}(kd^2)$ to linear $\mathcal{O}(kd)$, delivering sub-millisecond execution.

---

### 5.3 Empirical Bayes Beta-Binomial Target Smoothing

For sparse ad categories with few impressions ($N_i < 10$), naive sample click rates $\frac{C_i}{N_i}$ exhibit high variance. An **Empirical Bayes Beta-Binomial conjugate prior** stabilizes estimation:

$$\text{Prior: } \theta \sim \text{Beta}(\alpha, \beta), \quad \text{Likelihood: } C_i \mid N_i, \theta \sim \text{Binomial}(N_i, \theta)$$
$$\text{Posterior: } \theta \mid C_i, N_i \sim \text{Beta}(\alpha + C_i, \beta + N_i - C_i)$$

Parameterized by global prior mean $\mu = \frac{\alpha}{\alpha+\beta}$ and pseudo-count weight $m = \alpha + \beta$:

$$\hat{p}_{\text{smoothed}} = \mathbb{E}[\theta \mid C_i, N_i] = \frac{C_i + \mu \cdot m}{N_i + m}$$

---

### 5.4 Negative Downsampling Odds Inversion (He et al.)

When negative impressions are downsampled at rate $w \in (0, 1]$, the observed conditional probability $p'$ in the sampled training set is shifted. Applying Bayes' Theorem:

$$\text{Odds}' = \frac{p'}{1 - p'} = \frac{\mathbb{P}(y=1 \mid \mathbf{x})}{\mathbb{P}(y=0 \mid \mathbf{x}) \cdot w} = \frac{\text{Odds}}{w} \implies \text{Odds} = w \cdot \text{Odds}'$$

Solving for true calibrated posterior probability $p$:

$$p = \frac{\text{Odds}}{1 + \text{Odds}} = \frac{p'}{p' + \dfrac{1 - p'}{w}}$$

$$\Delta \text{Log-Odds} = \ln(w)$$

---

### 5.5 Expected Calibration Error (ECE) & Brier Score Decomposition

The mean squared probabilistic error (Brier Score) decomposes algebraically into calibration error, discrimination power, and baseline uncertainty:

$$\text{BS} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{p}_i)^2 = \underbrace{\sum_{m=1}^M \frac{|B_m|}{N} (\bar{p}_m - \bar{y}_m)^2}_{\text{Reliability (Calibration Error)}} - \underbrace{\sum_{m=1}^M \frac{|B_m|}{N} (\bar{y}_m - p_{\text{base}})^2}_{\text{Resolution (Discrimination)}} + \underbrace{p_{\text{base}}(1 - p_{\text{base}})}_{\text{Uncertainty}}$$

**Expected Calibration Error (ECE)** across $M$ reliability bins:

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

*In AdSpark:* Isotonic Regression reduces model ECE from **`0.0178`** to **`0.0026`** (an **$85.2\%$ error drop**).

---

### 5.6 McNemar Hypothesis Test with Continuity Correction

To statistically verify if algorithm $B$ (Ensemble) outperforms algorithm $A$ (Baseline) on paired test predictions:

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \quad \text{where } \begin{cases} b = \text{Model A correct, Model B incorrect} \\ c = \text{Model A incorrect, Model B correct} \end{cases}$$

$$p\text{-value} = 1 - F_{\chi_1^2}(\chi^2) = \text{erfc}\left(\sqrt{\frac{\chi^2}{2}}\right)$$

*Result:* $\chi^2 = 338.41, p = 1.34 \times 10^{-75} \ll 0.05$, confirming statistically significant superiority over linear baselines.

---

## 🏗️ System Architecture

```text
                               +---------------------------------------+
                               |     Avazu CTR Dataset (1.01M Rows)    |
                               +---------------------------------------+
                                                   |
                                                   v
                               +---------------------------------------+
                               |   Stage 01-03: Feature Engineering    |
                               |   - Frequency Encoding (IP, Device)   |
                               |   - Temporal Cyclical Trigo (sin/cos) |
                               |   - Empirical Bayes Target Smoothing  |
                               +---------------------------------------+
                                                   |
              +------------------------------------+------------------------------------+
              |                                                                         |
              v                                                                         v
+-----------------------------------+                                     +-----------------------------------+
|   Supervised Tree Ensembles (CO3) |                                     |   Unsupervised & Reduction (CO4)  |
| - XGBoost Classifier (0.7202 AUC) |                                     | - K-Means Clustering (k=3, WCSS)  |
| - LightGBM & HistGBM (0.7200 AUC) |                                     | - DBSCAN Density (14 Clusters)    |
| - Random Forest (0.7111 AUC)      |                                     | - Hierarchical Ward Dendrograms   |
| - CART Decision Trees (0.6705 AUC)|                                     | - PCA Scree & t-SNE Projections   |
+-----------------------------------+                                     +-----------------------------------+
              |                                                                         |
              +------------------------------------+------------------------------------+
                                                   |
                                                   v
                               +---------------------------------------+
                               |  Stage 14-19: Validation & Diagnostics|
                               | - 70/15/15 vs Stratified & Nested CV  |
                               | - Probability Calibration (ECE=0.0026)|
                               | - McNemar Significance (p < 1e-75)    |
                               | - TreeExplainer SHAP Feature Gain     |
                               +---------------------------------------+
                                                   |
                                                   v
                               +---------------------------------------+
                               |    Flask Web Application (app.py)     |
                               | - 22 Modular Interactive Dashboards   |
                               | - KaTeX LaTeX Mathematical Rendering  |
                               | - Real-Time Sandbox Inference Engine  |
                               +---------------------------------------+
```

---

## 🗂️ Web Application Directory & Routes

| URL Route | Module Title | Primary Focus & Displayed Metrics |
| :--- | :--- | :--- |
| `/` | **Overview Dashboard** | 4 Executive Hero Cards, 10-Model Verified Leaderboard, Pipeline Explorer |
| `/data-loading` | **Data Ingestion** | 1.01M Rows Inspection, 24 Columns Dtypes, Zero-Null Verification |
| `/eda` | **Exploratory Analysis** | 15-Step EDA Suite: Hourly distributions, Banner CTR, Device breakdowns |
| `/feature-engineering` | **Preprocessing** | Frequency encodings, cyclical trigonometric features, standard scaling |
| `/linear-regression` | **Linear Baselines** | OLS & Ridge normal equations, continuous probability bounds ($83.05\%$ Acc) |
| `/logistic-regression`| **Logistic Regression** | Sigmoidal binary cross-entropy, threshold optimization ($\tau=0.50$ vs. $0.178$) |
| `/regularization` | **L1/L2 Regularization** | Lasso soft-thresholding coordinate descent paths vs. Ridge $L_2$ shrinkage |
| `/decision-tree` | **Decision Trees** | CART Tree Pruning, max-depth sweeps ($1$–$10$), Gini Impurity ($83.10\%$ Acc) |
| `/ensemble` | **Ensemble Methods** | Random Forest, AdaBoost, GBM, LightGBM, XGBoost Champion comparison |
| `/kmeans` | **K-Means Clustering** | Clusters Formed ($k=3$), WCSS Elbow curve, 2D PCA spatial scatter with centroids |
| `/hierarchical` | **Hierarchical Clustering** | Single, Complete, Average & Ward Linkage dendrograms, 4 branch partitions |
| `/dbscan` | **DBSCAN Clustering** | 14 Dense Clusters + Outlier Noise ($18.33\%$), 5-NN distance graph, 2D scatter |
| `/dimensionality` | **Dimensionality Reduction** | PCA Scree Plot, 2D Biplot, Factor Loadings Heatmap, t-SNE perplexity sweep |
| `/anomaly` | **Anomaly Detection** | Isolation Forest & One-Class SVM impression outlier scoring |
| `/validation` | **Validation Strategies** | 70/15/15 3-Way Split vs. 5-Fold Stratified & $5\times 3$ Nested Cross-Validation |
| `/imbalanced-metrics`| **Imbalanced Evaluation** | PR-AUC, ROC-AUC, F1-Score, Youden's $J$ Index trade-offs |
| `/calibration` | **Probability Calibration** | Reliability diagrams, Platt scaling, Isotonic regression (ECE dropped to $0.0026$) |
| `/significance` | **McNemar Testing** | $2\times 2$ Contingency matrix, Edwards' $\chi^2 = 338.41$ significance test |
| `/learning-curves` | **Learning Curves** | LightGBM sample complexity curves ($N=2.5\text{k}$ to $20\text{k}$), bias-variance gap |
| `/explainability` | **SHAP Explainability** | TreeExplainer feature split gains (MDI) & Permutation importances |
| `/predict` | **Interactive Sandbox** | Live real-time inference with calibrated probabilities & Wilson 95% CIs |

---

## 🚀 Installation & Quickstart

### Prerequisites
* Python $\ge$ 3.10
* Windows PowerShell, macOS, or Linux Terminal

### 1. Clone the Repository
```bash
git clone https://github.com/ns-faiza9/AdSpark_CTR_prediction_MLproject.git
cd "AdSpark_CTR Prediction"
```

### 2. Install Dependencies
```bash
pip install -r analysis/requirements.txt
```

### 3. Run Pipeline Scripts (Optional)
To regenerate the metrics, figures, and serialized model artifacts from scratch:
```bash
# Execute supervised and CO4/CO5 diagnostic modules
python analysis/05_logistic_regression.py
python analysis/07_decision_tree.py
python analysis/08_ensemble.py
python analysis/09_kmeans.py
python analysis/11_dbscan.py
python analysis/12_dimensionality.py
```

### 4. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to **`http://127.0.0.1:5000`** to access the complete interactive suite.

---

## 🛠️ Real-Time Sandbox & REST API Documentation

The application exposes a production-ready scoring endpoint at `/api/predict`:

### POST `/api/predict`
Calculates calibrated CTR probabilities using the trained champion LightGBM model, evaluates feature contributions, and computes Wilson score confidence intervals.

#### Request JSON Payload:
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

#### Response JSON:
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
    },
    "bayesian_smoothed_ctr": 25.32,
    "key_drivers": [
      {
        "factor": "Trained GBDT Model Decision Trees",
        "impact": "Split Evaluation",
        "weight": "0.271"
      },
      {
        "factor": "High Engagement Site Category (28905ebd)",
        "impact": "Positive (+)",
        "weight": "+0.22"
      }
    ],
    "mathematical_formula": "p = sigma(w_0 + w^T x + 0.5 sum <v_i, v_j> x_i x_j) / (p' + (1-p')/w)"
  }
}
```

---

## 📚 Academic References

1. **Rendle, S.** (2010). *Factorization Machines*. IEEE 10th International Conference on Data Mining (ICDM '10), 995–1000.
2. **He, X., et al.** (2014). *Practical Lessons from Predicting Clicks on Ads at Facebook*. Proceedings of the 8th International Workshop on Data Mining for Online Advertising (ADKDD '14), 1–9.
3. **McMahan, H. B., et al.** (2013). *Ad Click Prediction: a View from the Trenches*. Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '13), 1222–1230.
4. **Niculescu-Mizil, A., & Caruana, R.** (2005). *Predicting Good Probabilities with Supervised Learning*. Proceedings of the 22nd International Conference on Machine Learning (ICML '05), 625–632.
5. **Brier, G. W.** (1950). *Verification of Forecasts Expressed in Terms of Probability*. Monthly Weather Review, 78(1), 1–3.
6. **Edwards, A. L.** (1948). *Note on the "Correction for Continuity" in Test of Significance for Goodness of Fit and Contingency Tables of Two Degrees of Freedom*. Psychometrika, 13(3), 185–187.
7. **Lundberg, S. M., & Lee, S. I.** (2017). *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems (NeurIPS 30), 4765–4774.

---

### 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.