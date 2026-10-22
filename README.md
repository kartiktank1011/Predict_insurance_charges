# 🏥 Insurance Charge Predictor

> **ML project predicting annual medical insurance costs** using demographic and health data from the Kaggle Medical Cost Personal Dataset.

---

## 📋 Table of Contents
1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Methodology](#methodology)
5. [Results](#results)
6. [Run Instructions](#run-instructions)
7. [Web App](#web-app)
8. [Testing](#testing)

---

## Problem Statement

Medical insurance pricing is influenced by many personal factors. The goal is to build a **regression model** that accurately predicts a patient's annual insurance charge based on:

- Demographic features: `age`, `sex`, `region`
- Health indicators: `bmi`, `smoker`, `children`

This helps insurers price fairly and helps individuals estimate coverage costs.

---

## Dataset

| Property       | Value                         |
|----------------|-------------------------------|
| **Source**     | Kaggle — Medical Cost Personal Datasets |
| **Author**     | Miri Choi                     |
| **Rows**       | 1,338                         |
| **Columns**    | 7 (6 features + 1 target)     |
| **Target**     | `charges` — annual USD cost   |

### Columns

| Column     | Type    | Description                               |
|------------|---------|-------------------------------------------|
| `age`      | int     | Age of beneficiary (18–64)                |
| `sex`      | string  | Biological sex (`male`/`female`)          |
| `bmi`      | float   | Body Mass Index                           |
| `children` | int     | Number of dependents (0–5)               |
| `smoker`   | string  | Smoking status (`yes`/`no`)              |
| `region`   | string  | US region (`northeast/northwest/southeast/southwest`) |
| `charges`  | float   | Annual insurance charge in USD (target)  |

### Key EDA Findings

1. **Charges are right-skewed** (skew=1.51) — log-transform normalises the target.
2. **Smoking is the dominant predictor** — smokers pay **~3.8× more** ($32,050 vs $8,434 average).
3. **Age has a positive linear relationship** with charges, especially for smokers.
4. **High BMI + smoking creates a high-risk cluster** with extreme charges.
5. **Region and sex** have minimal marginal impact (correlations < 0.1).

---

## Project Structure

```
Predict_insurance_charges/
├── data/
│   └── insurance.csv              # Raw dataset (1,338 rows)
├── notebooks/
│   └── 01_eda.py                  # Exploratory Data Analysis script
├── src/
│   ├── __init__.py
│   ├── preprocess.py              # ColumnTransformer pipeline
│   ├── train.py                   # 6-model cross-validation comparison
│   └── tune_evaluate.py           # RandomizedSearchCV + plots + model save
├── models/
│   └── best_model.pkl             # Saved joblib bundle (model + preprocessor)
├── app/
│   └── streamlit_app.py           # Interactive web application
├── reports/
│   ├── model_comparison.csv       # Phase 4 CV results
│   ├── tuning_results.csv         # Phase 5 tuning results
│   ├── eda_charges_distribution.png
│   ├── eda_charges_vs_age.png
│   ├── eda_bmi_vs_charges.png
│   ├── eda_categorical_boxplots.png
│   ├── eda_correlation_heatmap.png
│   ├── actual_vs_predicted.png
│   ├── residuals.png
│   └── feature_importance.png
├── tests/
│   ├── __init__.py
│   ├── test_preprocess.py         # 22 preprocessing unit tests
│   └── test_prediction.py         # 18 prediction unit tests
├── requirements.txt
└── README.md
```

---

## Methodology

### 1. Preprocessing (`src/preprocess.py`)

- **Numeric features** (`age`, `bmi`, `children`): `StandardScaler`
- **Categorical features** (`sex`, `smoker`, `region`): `OneHotEncoder` (drop first)
- **Target** (`charges`): `log1p` transformation to reduce skew
- **Split**: 80/20 train/test with `random_state=42` (no data leakage — preprocessor fit on train only)

### 2. Baseline Models (`src/train.py`)

Six models trained with **5-fold cross-validation**:

| Model              | CV R²  | CV MAE  | CV RMSE |
|--------------------|--------|---------|---------|
| Gradient Boosting  | 0.8055 | 0.2158  | 0.3989  |
| Random Forest      | 0.7955 | 0.2184  | 0.4095  |
| XGBoost            | 0.7733 | 0.2343  | 0.4315  |
| Linear Regression  | 0.7527 | 0.2846  | 0.4514  |
| Ridge              | 0.7527 | 0.2851  | 0.4515  |
| Lasso              | 0.7488 | 0.2874  | 0.4552  |

*Metrics on log-transformed target.*

### 3. Hyperparameter Tuning (`src/tune_evaluate.py`)

`RandomizedSearchCV` (30 iterations, 5-fold CV) on the top 2 models:

| Model             | CV R²  | Test R²   | Test MAE | Test RMSE |
|-------------------|--------|-----------|----------|-----------|
| Random Forest     | 0.8194 | **0.8772**| $1,974   | $4,366    |
| Gradient Boosting | 0.8201 | 0.8759    | $2,022   | $4,389    |

**Winner: Random Forest** — Test R²=0.8772

### 4. Feature Importance

Top drivers identified by the Random Forest:
1. `smoker_yes` — by far the strongest predictor
2. `age` — strong positive linear effect
3. `bmi` — meaningful effect, especially combined with smoking

---

## Results

| Metric    | Value        |
|-----------|--------------|
| **Model** | Random Forest (tuned) |
| **Test R²** | **0.877**  |
| **Test MAE** | **$1,974** |
| **Test RMSE** | **$4,366** |

The model explains **~87.7%** of the variance in insurance charges and predicts within **$1,974** on average.

---

## Run Instructions

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt
```

### Step-by-step

```bash
# 1. Download dataset (or place insurance.csv in data/)
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/bencmbit/datasets/master/insurance.csv', 'data/insurance.csv')"

# 2. Exploratory Data Analysis (saves plots to reports/)
python notebooks/01_eda.py

# 3. Baseline model comparison (saves reports/model_comparison.csv)
python src/train.py

# 4. Hyperparameter tuning, evaluation & model saving (saves models/best_model.pkl)
python src/tune_evaluate.py

# 5. Run the web app
streamlit run app/streamlit_app.py

# 6. Run tests
python -m pytest tests/ -v
```

---

## Web App

The Streamlit app (`app/streamlit_app.py`) provides:

- **Interactive sliders** for age, BMI, children
- **Dropdowns** for sex, smoker status, and region
- **BMI classification** (Underweight / Normal / Overweight / Obese)
- **Risk tier** (Low / Moderate / High) with colour coding
- **Key driver highlights** for the predicted result
- **Monthly equivalent** charge display

```bash
streamlit run app/streamlit_app.py
# -> Opens at http://localhost:8501
```

---

## Testing

**40 unit tests** in two files:

```bash
python -m pytest tests/ -v
# 40 passed in ~3s
```

| Test File              | Tests | Coverage                              |
|------------------------|-------|---------------------------------------|
| `test_preprocess.py`   | 22    | Data loading, ColumnTransformer, splits, log-transform, feature names |
| `test_prediction.py`   | 18    | Bundle structure, business logic, boundary conditions, batch consistency |

Notable tests:
- Smoker charge > non-smoker charge (business logic)
- Age 60 > Age 20 in predicted charges
- BMI 45 > BMI 18 in predicted charges
- Extreme cases (elderly obese smoker) produce realistic high charges
- Batch vs single prediction consistency

---

## License

Dataset: [CC0 Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) via Kaggle.
Code: MIT License.
#
