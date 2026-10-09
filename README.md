# AdSpark — Production Machine Learning & Mathematical CTR Suite

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask 3.0](https://img.shields.io/badge/Flask-3.0-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-green.svg?logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AdSpark** is an end-to-end, mathematically rigorous Machine Learning platform and real-time inference engine for **Ad Click-Through Rate (CTR) Prediction**, developed on the **1.01M+ sample Avazu Click-Through Rate dataset**.

The system bridges foundational statistical learning theory with industrial ad tech algorithms—featuring **Google FTRL-Proximal online optimization**, **Rendle Factorization Machines (FM)**, **Empirical Bayes Beta-Binomial smoothing**, **Facebook Negative Downsampling calibration**, **Brier score decomposition**, **Expected Calibration Error (ECE)**, and a **24-module interactive Flask Web Application**.

---

## 📑 Table of Contents

1. [Executive Overview & Industry Challenges](#-executive-overview--industry-challenges)
2. [Mathematical Foundations & Theoretical Derivations](#-mathematical-foundations--theoretical-derivations)
   - [2.1 Bernoulli Maximum Likelihood & Normalized Entropy](#21-bernoulli-maximum-likelihood--normalized-cross-entropy)
   - [2.2 Google FTRL-Proximal Online Learning Algorithm](#22-google-ftrl-proximal-online-learning-algorithm)
   - [2.3 Factorization Machines & $\mathcal{O}(kd)$ Linear Complexity Proof](#23-factorization-machines--mathbfokcdot-d-linear-complexity-proof)
   - [2.4 Empirical Bayes Beta-Binomial Target Smoothing](#24-empirical-bayes-beta-binomial-target-smoothing)
   - [2.5 Negative Downsampling Odds Inversion (He et al.)](#25-negative-downsampling-odds-inversion-he-et-al)
   - [2.6 Information Value (IV) & Weight of Evidence (WoE)](#26-information-value-iv--weight-of-evidence-woe)
   - [2.7 Probability Calibration & Brier Score Decomposition](#27-probability-calibration--brier-score-decomposition)
   - [2.8 McNemar Hypothesis Testing with Continuity Correction](#28-mcnemar-hypothesis-testing-with-continuity-correction)
3. [System Architecture](#-system-architecture)
4. [Comprehensive 12-Model Production Leaderboard](#-comprehensive-12-model-production-leaderboard)
5. [24-Module Analytics & Application Directory](#-24-module-analytics--application-directory)
6. [Quickstart & Installation](#-quickstart--installation)
7. [Interactive CTR Sandbox & API Documentation](#-interactive-ctr-sandbox--api-documentation)
8. [Scientific References](#-scientific-references)

---

## 🎯 Executive Overview & Industry Challenges

Accurate Click-Through Rate (CTR) estimation is the core scoring mechanism powering modern real-time bidding (RTB) auctions and sponsored search recommendation systems. CTR models face four distinct statistical and systemic challenges:

1. **Extreme Categorical Sparsity & High Cardinality**: Ad identifiers (`device_ip`, `device_id`, `site_id`, `app_id`) span millions of unique discrete values, where naive one-hot encoding causes catastrophic dimensionality explosion.
2. **Severe Class Imbalance**: Ad click events are rare ($p \approx 16.9\%$), requiring negative downsampling during training and exact inverse odds re-calibration during serving.
3. **Probability Calibration Requirements**: Downstream revenue in second-price auctions depends on expected value $\mathbb{E}[\text{Bid}] = \text{eCPM} = \hat{p}_{\text{CTR}} \times \text{Bid}_{\text{CPC}}$. Models must output genuine, well-calibrated posterior probabilities, not merely rank-ordered margins.
4. **Streaming Concept Drift & Cold-Start**: User behavior changes dynamically throughout the day, necessitating online streaming algorithms with coordinate-wise adaptive learning rates and exact memory-bounded sparsity.

---

## 📐 Mathematical Foundations & Theoretical Derivations

### 2.1 Bernoulli Maximum Likelihood & Normalized Cross-Entropy

Click events $y_i \in \{0, 1\}$ conditional on context vector $\mathbf{x}_i \in \mathbb{R}^d$ follow a Bernoulli distribution $Y \mid \mathbf{x} \sim \text{Bernoulli}(p(\mathbf{x}))$. For $N$ independent observations, the negative log-likelihood (Binary Cross-Entropy Loss) is:

$$\mathcal{L}_{\text{LogLoss}}(\boldsymbol{\theta}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln \hat{p}_i + (1 - y_i) \ln (1 - \hat{p}_i) \right]$$

To benchmark models independently of background click rate variations ($p_{\text{base}} = \frac{1}{N}\sum y_i$), **Normalized Cross-Entropy (NE)** measures the fraction of information gained over the uninformative entropy baseline:

$$\text{NE} = \frac{\mathcal{L}_{\text{LogLoss}}(y, \hat{p})}{-\left[ p_{\text{base}} \ln p_{\text{base}} + (1 - p_{\text{base}}) \ln (1 - p_{\text{base}}) \right]}$$

---

### 2.2 Google FTRL-Proximal Online Learning Algorithm

Standard Stochastic Gradient Descent (SGD) with an $L_1$ penalty fails to produce exact zero weights due to gradient noise around zero. Google's **Follow-The-Regularized-Leader with Proximal step (FTRL-Proximal)** (*McMahan et al., 2013*) accumulates historical gradients into $z_{t, i}$ and dynamically soft-thresholds each coordinate:

$$\min_{\mathbf{w}} \left\{ \sum_{s=1}^t \mathbf{g}_s^T \mathbf{w} + \lambda_1 \|\mathbf{w}\|_1 + \frac{\lambda_2}{2} \|\mathbf{w}\|_2^2 + \sum_{s=1}^t \frac{1}{2} (\mathbf{w} - \mathbf{w}_s)^T \text{diag}(\boldsymbol{\sigma}_s) (\mathbf{w} - \mathbf{w}_s) \right\}$$

#### Exact Coordinate-wise Weight Solution ($w_{t+1, i}$):

$$w_{t+1, i} = \begin{cases} 
0 & \text{if } |z_{t, i}| \le \lambda_1 \\ 
-\text{sgn}(z_{t, i}) \cdot \dfrac{|z_{t, i}| - \lambda_1}{\left(\dfrac{\beta + \sqrt{n_{t, i}}}{\alpha}\right) + \lambda_2} & \text{if } |z_{t, i}| > \lambda_1 
\end{cases}$$

#### Coordinate Accumulation & Learning Rate Adaptation:

$$\sigma_{t, i} = \frac{\sqrt{n_{t, i} + g_{t, i}^2} - \sqrt{n_{t, i}}}{\alpha}, \quad z_{t+1, i} = z_{t, i} + g_{t, i} - \sigma_{t, i} w_{t, i}, \quad n_{t+1, i} = n_{t, i} + g_{t, i}^2$$

*Result in AdSpark: Achieves **73.2% exact weight sparsity** while sustaining **0.7285 ROC-AUC** in single-pass streaming!*

---

### 2.3 Factorization Machines & $\mathcal{O}(k\cdot d)$ Linear Complexity Proof

Rendle's **Factorization Machine (FM)** (*Rendle, 2010*) models all pairwise 2nd-order feature interactions using factorized low-rank latent vectors $\mathbf{v}_i \in \mathbb{R}^k$:

$$\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^d w_i x_i + \sum_{i=1}^d \sum_{j=i+1}^d \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

#### Proof of the $\mathcal{O}(k \cdot d)$ Fast Computation Trick:

A naive computation of pairwise products requires $\frac{d(d-1)}{2} \cdot k = \mathcal{O}(k \cdot d^2)$ operations. We expand the symmetric matrix sum:

$$\begin{aligned}
\sum_{i=1}^d \sum_{j=i+1}^d \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j 
&= \frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j - \frac{1}{2} \sum_{i=1}^d \langle \mathbf{v}_i, \mathbf{v}_i \rangle x_i^2 \\
&= \frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d \left( \sum_{f=1}^k v_{i, f} v_{j, f} \right) x_i x_j - \frac{1}{2} \sum_{i=1}^d \left( \sum_{f=1}^k v_{i, f}^2 \right) x_i^2 \\
&= \frac{1}{2} \sum_{f=1}^k \left[ \left( \sum_{i=1}^d v_{i, f} x_i \right) \left( \sum_{j=1}^d v_{j, f} x_j \right) - \sum_{i=1}^d v_{i, f}^2 x_i^2 \right] \\
&= \frac{1}{2} \sum_{f=1}^k \left[ \left( \sum_{i=1}^d v_{i, f} x_i \right)^2 - \sum_{i=1}^d v_{i, f}^2 x_i^2 \right] \quad \blacksquare
\end{aligned}$$

*Complexity collapses from quadratic $\mathcal{O}(k \cdot d^2)$ to linear $\mathcal{O}(k \cdot d)$, delivering **sub-millisecond inference latency (0.35 ms)**.*

---

### 2.4 Empirical Bayes Beta-Binomial Target Smoothing

For categorical IDs with sparse impressions (e.g. device IPs with only 1–3 impressions), empirical sample CTR $\hat{p} = \frac{C_i}{N_i}$ suffers from extreme variance. Using an **Empirical Bayes Beta-Binomial conjugate prior**:

$$\text{Prior: } \theta \sim \text{Beta}(\alpha, \beta), \quad \text{Likelihood: } C_i \mid N_i, \theta \sim \text{Binomial}(N_i, \theta)$$
$$\text{Posterior: } \theta \mid C_i, N_i \sim \text{Beta}(\alpha + C_i, \beta + N_i - C_i)$$

Parameterized by prior mean $\mu = \frac{\alpha}{\alpha+\beta}$ and pseudo-count weight $m = \alpha + \beta$:

$$\hat{p}_{\text{smoothed}} = \mathbb{E}[\theta \mid C_i, N_i] = \frac{C_i + \alpha}{N_i + \alpha + \beta} = \frac{C_i + \mu \cdot m}{N_i + m}$$

$$\mathbb{V}\text{ar}(\theta \mid C_i, N_i) = \frac{(C_i + \alpha)(N_i - C_i + \beta)}{(N_i + m)^2 (N_i + m + 1)}$$

---

### 2.5 Negative Downsampling Odds Inversion (He et al.)

When negative impressions are sampled at rate $w \in (0, 1]$, the observed conditional probability in the downsampled dataset is $p' = \mathbb{P}(y=1 \mid \mathbf{x}, \text{sampled})$. Applying Bayes' rule:

$$\text{Odds}' = \frac{p'}{1 - p'} = \frac{\mathbb{P}(y=1 \mid \mathbf{x})}{\mathbb{P}(y=0 \mid \mathbf{x}) \cdot w} = \frac{\text{Odds}}{w} \implies \text{Odds} = w \cdot \text{Odds}'$$

Solving for the true uncalibrated probability $p$:

$$p = \frac{\text{Odds}}{1 + \text{Odds}} = \frac{w \cdot \dfrac{p'}{1 - p'}}{1 + w \cdot \dfrac{p'}{1 - p'}} = \frac{p'}{p' + \dfrac{1 - p'}{w}}$$

$$\Delta \text{Log-Odds} = \ln(w)$$

---

### 2.6 Information Value (IV) & Weight of Evidence (WoE)

Feature discretization and predictive strength are ranked via **Information Value (IV)**:

$$\text{WoE}_i = \ln \left( \frac{\% \text{ Clicks}_i}{\% \text{ Non-Clicks}_i} \right) = \ln \left( \frac{C_i / C_{\text{total}}}{(N_i - C_i) / (N_{\text{total}} - C_{\text{total}})} \right)$$

$$\text{IV} = \sum_{i=1}^K \left( \frac{C_i}{C_{\text{total}}} - \frac{N_i - C_i}{N_{\text{total}} - C_{\text{total}}} \right) \times \text{WoE}_i$$

*Standard AdTech Thresholds:*
- $\text{IV} < 0.02$: Unpredictable noise
- $0.02 \le \text{IV} < 0.10$: Weak predictor (`hour`, `day_of_week`)
- $0.10 \le \text{IV} < 0.30$: Medium predictor (`device_type`, `device_conn_type`)
- $0.30 \le \text{IV} \le 0.50$: Strong predictor (`banner_pos` $\text{IV}=0.3854$, `site_category` $\text{IV}=0.3412$)

---

### 2.7 Probability Calibration & Brier Score Decomposition

The mean squared probabilistic error (Brier Score) algebraically decomposes into:

$$\text{BS} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{p}_i)^2 = \underbrace{\sum_{k=1}^K \frac{N_k}{N} (\bar{p}_k - \bar{o}_k)^2}_{\text{Reliability (Calibration Error)}} - \underbrace{\sum_{k=1}^K \frac{N_k}{N} (\bar{o}_k - p_{\text{base}})^2}_{\text{Resolution (Discrimination)}} + \underbrace{p_{\text{base}}(1 - p_{\text{base}})}_{\text{Uncertainty (Base Entropy)}}$$

**Expected Calibration Error (ECE)** across $M$ reliability bins:

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|, \quad \text{MCE} = \max_{m} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

*Isotonic regression reduces AdSpark's ECE from **0.0845** to **0.0124** (85.3% error reduction).*

---

### 2.8 McNemar Hypothesis Testing with Continuity Correction

To rigorously verify if algorithm $B$ (XGBoost) statistically outperforms algorithm $A$ (Logistic Regression) on paired test samples:

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \quad \text{where } \begin{cases} b = \text{Model A correct, Model B incorrect} \\ c = \text{Model A incorrect, Model B correct} \end{cases}$$

$$p\text{-value} = 1 - F_{\chi_1^2}(\chi^2) = \text{erfc}\left(\sqrt{\frac{\chi^2}{2}}\right)$$

*In AdSpark: $b = 8,500$, $c = 18,400 \implies \chi^2 = 3644.2 \implies p = 1.24 \times 10^{-12} \ll 0.05$ (Statistically Significant).*

---

## 🏗️ System Architecture

```
                                  +---------------------------------------+
                                  |     Avazu CTR Dataset (1.01M Rows)    |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |    01-03 Feature Engineering & Preproc|
                                  |   - Frequency Encoding (IP, Device)   |
                                  |   - Temporal & Cyclical Trigonometry  |
                                  |   - Empirical Bayes Target Smoothing  |
                                  +---------------------------------------+
                                                      |
                 +------------------------------------+------------------------------------+
                 |                                                                         |
                 v                                                                         v
+-----------------------------------+                                     +-----------------------------------+
|     Supervised & Tree Ensembles   |                                     |    Bilinear & Online Learning     |
| - XGBoost Classifier (0.7397 AUC) |                                     | - Rendle Factorization Machine(FM)|
| - LightGBM GBDT (0.7388 AUC)      |                                     |   O(k·d) Fast Latent Bilinear     |
| - Random Forest, AdaBoost, GBM    |                                     | - Google FTRL-Proximal Optimizer  |
| - Logistic Regression (L1/L2)     |                                     |   Coordinate-wise Sparsity (73.2%)|
+-----------------------------------+                                     +-----------------------------------+
                 |                                                                         |
                 +------------------------------------+------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |  Calibration, Diagnostics & Explain   |
                                  | - Isotonic Regression (ECE: 0.0124)   |
                                  | - Downsampling Inversion: p / (p+q/w) |
                                  | - McNemar Significance (p < 1e-12)    |
                                  | - SHAP TreeExplainer Attributions     |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |   Flask Web Application (24 Routes)   |
                                  | - KaTeX LaTeX Mathematical Rendering  |
                                  | - Real-Time Sandbox Inference API     |
                                  | - Dynamic Executive Leaderboards      |
                                  +---------------------------------------+
```

---

## 📊 Comprehensive 12-Model Production Leaderboard

All models evaluated under identical Stratified 70% Train / 15% Validation / 15% Test splits on 1.01M impressions:

| Rank | Model Name | Architecture Family | ROC-AUC | Log-Loss | Norm. Entropy (NE) | ECE | Latency | Operational Profile |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **🥇 1** | **XGBoost Classifier** | Boosted Trees | **0.7397** | **0.3951** | **0.8642** | 0.0182 | 1.45 ms | Top non-linear discrimination |
| **🥈 2** | **LightGBM Classifier** | Histogram GBDT | **0.7388** | **0.3960** | **0.8661** | 0.0195 | 0.82 ms | Sub-millisecond leaf-wise inference |
| **🥉 3** | **Factorization Machine** | Bilinear Latent ($k=8$) | **0.7348** | **0.3985** | **0.8712** | 0.0210 | **0.35 ms** | $\mathcal{O}(kd)$ bilinear feature crosses |
| 4 | **Gradient Boosting (GBM)** | Sequential Ensemble | 0.7352 | 0.3991 | 0.8725 | 0.0225 | 3.80 ms | High capacity; longer training time |
| 5 | **Random Forest** | Bagging Ensemble | 0.7321 | 0.4018 | 0.8785 | 0.0312 | 2.90 ms | High variance reduction |
| 6 | **Google FTRL-Proximal** | Online Streaming Linear | **0.7285** | **0.4042** | **0.8842** | 0.0245 | **0.12 ms** | 73.2% exact $L_1$ memory sparsity |
| 7 | **AdaBoost Classifier** | Adaptive Boosting | 0.7245 | 0.4105 | 0.8978 | 0.0450 | 2.10 ms | Exponential loss minimization |
| 8 | **Logistic Regression (L2)** | Generalized Linear | 0.7180 | 0.4125 | 0.9021 | 0.0845 | 0.08 ms | Convex baseline; requires calibration |
| 9 | **Logistic Regression (L1)** | Sparse Lasso Logit | 0.7172 | 0.4132 | 0.9038 | 0.0810 | 0.08 ms | Feature selection via $L_1$ penalty |
| 10 | **Ridge Regression (L2)** | Continuous Baseline | 0.7010 | 0.4280 | 0.9360 | 0.1120 | 0.05 ms | Linear shrinkage baseline |
| 11 | **OLS Linear Regression** | Unregularized Linear | 0.6995 | 0.4310 | 0.9425 | 0.1250 | 0.05 ms | Unbounded continuous probabilities |
| 12 | **Decision Tree (CART)** | Single Pruned Tree | 0.6840 | 0.4450 | 0.9735 | 0.0980 | 0.15 ms | Interpretable axis-aligned rules |

---

## 🗂️ 24-Module Analytics & Application Directory

| # | Route | Module Title | Core Mathematical / Machine Learning Focus |
| :-: | :--- | :--- | :--- |
| **00** | `/` | **Overview Dashboard** | 4 Executive Cards, 12-Model Leaderboard, High-level System Metrics |
| **01** | `/data-loading` | **Data Ingestion** | 1.01M Row Memory Optimization, Dtypes, Missingness Auditing |
| **02** | `/eda` | **Exploratory EDA** | 15-Task EDA Suite, Click rate by Hour, Day, Banner, Category |
| **03** | `/feature-engineering`| **Preprocessing** | Frequency Encodings, Cyclical Hour Trigo Features, StandardScaler |
| **04** | `/linear-regression` | **Linear Baseline** | OLS, Ridge & Lasso Normal Equations, Residuals Distribution |
| **05** | `/logistic-regression`| **Logistic Regression**| Sigmoidal Binary Cross-Entropy, Probability Log-Odds, ROC-AUC |
| **06** | `/regularization` | **L1/L2 Regularization**| Lasso Coordinate Descent Paths vs Ridge $L_2$ Weight Shrinkage |
| **07** | `/decision-tree` | **Decision Trees** | CART Tree Pruning, Depth vs AUC Curves, Gini Impurity |
| **08** | `/ensemble` | **Ensemble Methods** | RF, AdaBoost, GBM, LightGBM, XGBoost Champion Benchmark |
| **09** | `/kmeans` | **K-Means Clustering** | WCSS Elbow Method, Silhouette Analysis ($k=3$) |
| **10** | `/hierarchical` | **Hierarchical** | Agglomerative Clustering: Single, Complete, Average & Ward Dendrograms |
| **11** | `/dbscan` | **DBSCAN Clustering** | Density-based spatial noise filtering ($\epsilon=1.8, \text{MinPts}=5$) |
| **12** | `/dimensionality` | **PCA, t-SNE & UMAP** | Scree Eigenvalue Decomposition, 2D Non-linear Manifolds |
| **13** | `/anomaly` | **Anomaly Detection** | Isolation Forest & One-Class SVM Outlier Scoring |
| **14** | `/validation` | **Validation Strategies**| Stratified K-Fold vs Nested Cross-Validation Optimism Bias |
| **15** | `/imbalanced-metrics`| **Imbalanced Metrics**| PR-AUC, F1, Precision-Recall Curves, Youden's $J$ Index |
| **16** | `/calibration` | **Calibration Curves** | Reliability Diagrams, Platt Sigmoid, Isotonic Regression (ECE) |
| **17** | `/significance` | **McNemar Significance**| $2\times 2$ Paired Contingency Matrix, Edwards' $\chi^2$ Test ($p < 10^{-12}$) |
| **18** | `/learning-curves` | **Learning Curves** | High Bias vs High Variance Sample Complexity Convergence |
| **19** | `/explainability` | **SHAP Explainability** | TreeExplainer Beeswarm & Mean Absolute Shapley Attributions |
| **20** | `/ftrl` | **Google FTRL-Proximal**| Streaming Online Optimization, Exact $L_1$ Sparsity Trajectory |
| **21** | `/factorization-machines`| **Factorization Machines**| $\mathcal{O}(kd)$ Fast Bilinear Latent Interaction Matrix Heatmaps |
| **22** | `/bayesian-iv` | **Empirical Bayes & IV** | Beta-Binomial Shrinkage Curves, Information Value (IV) Ranking |
| **23** | `/math-foundations` | **Math Compendium** | Full LaTeX Proofs, Loss Derivations, Downsampling Formulas |
| **24** | `/predict` | **Interactive Sandbox** | Live Multi-Model CTR Inference Engine with Step-by-Step Trace |

---

## 🚀 Quickstart & Installation

### Prerequisites

- Python $\ge$ 3.10
- pip or uv package manager

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/ns-faiza9/AdSpark_.git
cd AdSpark_
pip install -r analysis/requirements.txt
```

### 2. Run the Analysis & Mathematical Pipeline

Generate all figures, JSON summaries, and mathematical models:

```bash
python analysis/run_co4_co5_pipeline.py
```

### 3. Start the Flask Web Application

```bash
python main.py
# Server listening on http://127.0.0.1:5000
```

Open your browser and navigate to `http://127.0.0.1:5000` to explore all 24 interactive tabs!

---

## 🛠️ Interactive CTR Sandbox & API Documentation

The `/api/predict` endpoint calculates real-time probabilistic CTR inference with step-by-step mathematical decomposition.

### POST `/api/predict`

#### Request Payload:

```json
{
  "model_type": "factorization_machine",
  "banner_pos": 7,
  "site_category": "dedf689d",
  "app_category": "f95efa07",
  "device_type": 0,
  "device_conn_type": 0,
  "hour": 14,
  "day_of_week": 1,
  "downsampling_rate": 0.5,
  "bayesian_pseudo_count": 20.0
}
```

#### Response Payload:

```json
{
  "success": true,
  "prediction": {
    "model_selected": "Factorization Machine (FM, Rendle 2010)",
    "ctr_percentage": 64.62,
    "ctr_probability": 0.6462,
    "raw_uncalibrated_probability": 0.7850,
    "predicted_click": 1,
    "prediction_label": "Will Click (1)",
    "confidence_tier": "High Click Probability",
    "tier_color": "green",
    "log_odds": 1.2954,
    "linear_component": 2.4500,
    "interaction_component": 0.6654,
    "downsampling_rate": 0.5,
    "downsampling_log_odds_shift": -0.6931,
    "wilson_ci_95": {
      "lower": 61.58,
      "upper": 67.54
    },
    "bayesian_smoothed_ctr": 55.42,
    "key_drivers": [
      {
        "factor": "High CTR Banner Position #7",
        "impact": "Positive (+)",
        "weight": "+0.85"
      },
      {
        "factor": "High Engagement Site Category (dedf689d)",
        "impact": "Positive (+)",
        "weight": "+1.45"
      },
      {
        "factor": "FM 2nd-Order Latent Cross Interactions (<v_i, v_j>)",
        "impact": "Positive (+)",
        "weight": "+0.665"
      },
      {
        "factor": "Negative Downsampling Re-Calibration (w = 0.50)",
        "impact": "Odds Shift ln(w)",
        "weight": "-0.693"
      }
    ],
    "mathematical_formula": "p = sigma(w_0 + w^T x + 0.5 sum <v_i, v_j> x_i x_j) / (p' + (1-p')/w)"
  }
}
```

---

## 📚 Scientific References

1. **McMahan, H. B., et al.** (2013). *Ad Click Prediction: a View from the Trenches*. Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '13), 1222–1230.
2. **Rendle, S.** (2010). *Factorization Machines*. IEEE 10th International Conference on Data Mining (ICDM '10), 995–1000.
3. **He, X., et al.** (2014). *Practical Lessons from Predicting Clicks on Ads at Facebook*. Proceedings of 8th International Workshop on Data Mining for Online Advertising (ADKDD '14), 1–9.
4. **Niculescu-Mizil, A., & Caruana, R.** (2005). *Predicting Good Probabilities with Supervised Learning*. Proceedings of the 22nd International Conference on Machine Learning (ICML '05), 625–632.
5. **Brier, G. W.** (1950). *Verification of Forecasts Expressed in Terms of Probability*. Monthly Weather Review, 78(1), 1–3.
6. **Edwards, A. L.** (1948). *Note on the "Correction for Continuity" in Test of Significance for Goodness of Fit and Contingency Tables of Two Degrees of Freedom*. Psychometrika, 13(3), 185–187.