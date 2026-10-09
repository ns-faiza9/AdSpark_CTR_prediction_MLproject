# AdSpark — Analysis Pipeline

The **analysis** side of the AdSpark CTR Prediction project. This is kept
**separate from the frontend** (`webpage/`): the Python scripts here do all the
data science work and save their results (figures + JSON summaries) into
`analysis/output/`, which the React frontend then displays.

## Stages (one script per sidebar component)

| Script | Sidebar component | What it does |
| ------ | ----------------- | ------------ |
| `00_prepare_data.py` | — (setup) | Samples ~1M rows from the 1.1 GB raw `train.gz` |
| `01_data_loading.py` | Data Loading | Loads, inspects and previews the dataset |
| `02_eda.py` | EDA | 15-step statistical & visual analysis |
| `03_feature_engineering.py` | Preprocessing | Missing-value strategy, frequency/label encoding, hour parsing |
| `04_linear_regression.py` | Linear Regression | OLS baseline model & performance evaluation |
| `05_logistic_regression.py` | Logistic Regression | Classification model for click prediction |
| `06_regularization.py` | Regularization | Lasso, Ridge & Elastic Net comparison |
| `07_decision_tree.py` | Decision Tree | Tree-based classifier with depth sweep |
| `08_ensemble.py` | Ensemble Learning | Random Forest, Gradient Boosting & AdaBoost |

## How to run

```bash
pip install -r requirements.txt

python analysis/00_prepare_data.py   # one-time: builds the sample
python analysis/01_data_loading.py
python analysis/02_eda.py
python analysis/03_feature_engineering.py
python analysis/04_linear_regression.py
python analysis/05_logistic_regression.py
python analysis/06_regularization.py
python analysis/07_decision_tree.py
python analysis/08_ensemble.py
```

Or run everything in one go:

```bash
python analysis/run_all.py
```

## How to open the webpage

1. Open terminal and run:
   ```bash
   cd webpage
   npm run dev
   ```
2. Open your browser at [http://localhost:3000](http://localhost:3000)

## Outputs

- **Figures** → `analysis/output/figures/*.png` (displayed in the webpage)
- **JSON summaries** → `analysis/output/*.json` (metrics shown in the webpage)
- **Tables** → `analysis/output/*.csv`
- **Processed datasets** → `Data/processed/*.csv`

## Dataset

Avazu CTR (Kaggle) — ~40.4M ad impressions, 24 columns. The pipeline samples
~2.5% (~1M rows) so it runs quickly on a laptop while remaining statistically
sound. The target is `click` (1 = clicked, 0 = not clicked).