# PredictAI

## End-to-End Machine Learning Prediction Platform

PredictAI is an automated machine-learning platform built with Python and Streamlit that takes a tabular CSV dataset and guides it through an end-to-end prediction workflow.

The platform automatically profiles the dataset, detects the machine-learning problem type, preprocesses numerical and categorical features, compares multiple models, optionally performs XGBoost hyperparameter optimization with Optuna, selects the final model, evaluates its performance, explains individual predictions using SHAP, and generates predictions for new data.

---

## Features

### 1. Dataset Profiling

Upload a CSV dataset and immediately inspect:

- Number of rows
- Number of columns
- Missing values
- Duplicate rows
- Column data types
- Missing-value percentages
- Number of unique values
- Dataset preview

PredictAI does not assume a fixed dataset schema.

---

### 2. Automatic Problem-Type Detection

The platform automatically determines whether the task is:

- Classification
- Regression

The target column can be selected directly from the uploaded dataset.

---

### 3. Automated Preprocessing

PredictAI builds a preprocessing pipeline based on feature types.

Numerical and categorical features are handled separately before being passed to the machine-learning models.

The preprocessing pipeline is integrated with model training so that the same transformations are consistently applied during evaluation and prediction.

---

### 4. Model Comparison

PredictAI trains and compares multiple candidate models using cross-validation.

For classification, the platform evaluates metrics including:

- Accuracy
- Precision
- Recall
- F1

For regression, metrics include regression-oriented performance measures such as:

- R²
- RMSE
- MAE

The best baseline model is identified automatically using the primary task metric.

---

### 5. Automatic XGBoost Hyperparameter Optimization

PredictAI uses Optuna to optimize XGBoost hyperparameters.

The optimization budget is automatically determined based on dataset size rather than requiring the user to manually choose the number of trials.

The optimization process reports:

- Trial progress
- Current trial score
- Best cross-validation score
- Best hyperparameters

Example parameters optimized include:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`

---

### 6. Intelligent Optimization Skipping

PredictAI avoids unnecessary hyperparameter optimization when it is unlikely to provide value.

If the best baseline model already achieves a perfect cross-validation score for the primary metric, Optuna optimization is automatically skipped.

The platform also warns that perfect validation performance can sometimes indicate:

- A highly deterministic dataset
- Target leakage
- An unusually easy prediction problem

This prevents unnecessary computation while also encouraging users to investigate suspiciously perfect performance.

---

### 7. Final Model Selection

Candidate models are compared using their cross-validation scores.

The final model is selected automatically.

When tuned XGBoost and the baseline model produce effectively identical scores, PredictAI treats them as a tie and prefers the simpler baseline model.

This avoids unnecessary model complexity when tuning provides no meaningful improvement.

---

### 8. Final Model Evaluation

After selecting the final model, PredictAI evaluates it on the held-out test set.

For classification, the interface displays:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

This separates cross-validation-based model selection from final test-set evaluation.

---

### 9. SHAP Explainability

PredictAI provides individual prediction explanations using SHAP.

Users can select a prediction from the test set and inspect:

- The selected input sample
- The predicted class/value
- Prediction confidence for classification
- Feature-level contribution magnitudes
- Top contributing features
- A contribution visualization

This makes the model's individual predictions more interpretable instead of treating the trained model as a black box.

---

### 10. Prediction on New Data

Once a final model has been trained, users can upload a new CSV containing unseen samples.

PredictAI:

1. Reads the uploaded dataset
2. Validates the feature schema
3. Rejects missing required columns
4. Generates predictions
5. Displays the prediction results
6. Allows the results to be downloaded as a CSV

This enables the trained model to be used on new datasets without retraining.

---

### 11. Schema Validation

Prediction data must follow the feature schema expected by the trained model.

For example, if a required feature such as `customer_age` is missing, PredictAI reports the schema validation error rather than silently generating invalid predictions.

This helps prevent accidental misuse of the trained model.

---

## System Workflow

```
                ┌─────────────────────┐
                │     Upload CSV      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Dataset Profiling   │
                │ • Rows              │
                │ • Columns           │
                │ • Missing values    │
                │ • Duplicates        │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Select Target       │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Problem Detection   │
                │ Classification /    │
                │ Regression          │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Preprocessing       │
                │ Numerical +         │
                │ Categorical         │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Model Comparison    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Best Baseline       │
                └──────────┬──────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Perfect CV Score? │
                 └────────┬──┬───────┘
                        Yes  │No
                          │  │
                          ▼  ▼
                  Skip Optuna  Optuna
                          │     │
                          │     ▼
                          │  Tuned XGBoost
                          │     │
                          └──┬──┘
                             ▼
                ┌─────────────────────┐
                │ Final Model         │
                │ Selection           │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Test Evaluation     │
                └──────────┬──────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
             SHAP Explainability  New Data
                                  │
                                  ▼
                            Predictions CSV
```

---

## Project Structure

```
PredictAI/
│
├── app/
│   └── app.py
│
├── src/
│   ├── data/
│   │   ├── datasets.py
│   │   └── profiler.py
│   │
│   ├── preprocessing/
│   │   └── pipeline.py
│   │
│   ├── models/
│   │   ├── registry.py
│   │   ├── trainer.py
│   │   ├── comparator.py
│   │   ├── tuner.py
│   │   └── production.py
│   │
│   ├── evaluation/
│   │   └── metrics.py
│   │
│   ├── explainability/
│   │   └── shap_explainer.py
│   │
│   └── prediction/
│       └── predictor.py
│
├── tests/
│   ├── test_comparison.py
│   ├── test_evaluation.py
│   ├── test_explainability.py
│   ├── test_prediction.py
│   ├── test_preprocessing.py
│   ├── test_production.py
│   ├── test_profiler.py
│   ├── test_training.py
│   ├── test_tuning.py
│   ├── test_real_classification.py
│   ├── test_real_regression.py
│   └── ...
│
├── data/
│   └── .gitkeep
│
├── models/
│   └── .gitkeep
│
├── .gitignore
├── requirements.txt
└── README.md
```

Datasets and generated model artifacts are intentionally excluded from version control.

---

## Tech Stack

| Category | Tools |
|---|---|
| Programming | Python |
| Machine Learning | scikit-learn, XGBoost, Optuna |
| Explainability | SHAP |
| Data Processing | pandas, NumPy |
| Application | Streamlit |
| Visualization | Matplotlib |

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd PredictAI
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

From the project root:

```bash
streamlit run app/app.py
```

The application will open in the browser.

---

## Typical Usage

**Step 1 — Upload a dataset**
Upload a CSV containing the target variable.

**Step 2 — Select the target**
Choose the column the model should predict.

**Step 3 — Inspect the dataset**
Review dataset size, missing values, duplicates, data types, and feature information.

**Step 4 — Train and compare models**
Run the model comparison pipeline.

**Step 5 — Optimize when useful**
PredictAI automatically determines whether XGBoost optimization should be performed.

**Step 6 — Select the final model**
The platform compares the baseline and tuned candidates and selects the appropriate final model.

**Step 7 — Evaluate**
Train the selected model and inspect its held-out test-set performance.

**Step 8 — Explain predictions**
Use the SHAP explainability section to understand individual predictions.

**Step 9 — Predict on new data**
Upload unseen data using the expected feature schema and generate predictions.

**Step 10 — Download results**
Export predictions as a CSV file.

---

## Testing

The project includes tests covering the major components of the ML pipeline.

Run the test suite by executing the test modules:

```bash
python -m tests.test_profiler
python -m tests.test_preprocessing
python -m tests.test_training
python -m tests.test_comparison
python -m tests.test_tuning
python -m tests.test_evaluation
python -m tests.test_explainability
python -m tests.test_prediction
python -m tests.test_production
python -m tests.test_real_classification
python -m tests.test_real_regression
```

The test suite covers areas including:

- Dataset profiling
- Preprocessing
- Model training
- Model comparison
- Hyperparameter tuning
- Evaluation
- Explainability
- Prediction
- Production behavior
- Real classification datasets
- Real regression datasets

---

## Design Principles

PredictAI is designed around several principles:

**Automation**
Users should not need to manually construct an ML pipeline for every tabular dataset.

**Reproducibility**
Preprocessing and modeling steps are encapsulated in reusable pipelines.

**Model Selection**
The platform compares candidate models instead of assuming that one algorithm is always optimal.

**Efficient Optimization**
Hyperparameter optimization is performed when it can provide value and skipped when baseline validation performance is already perfect.

**Explainability**
Predictions should be interpretable through feature-level explanations.

**Safe Prediction**
New prediction data is validated against the expected feature schema before inference.

**Modularity**
Data processing, preprocessing, modeling, evaluation, explainability, and prediction are separated into independent modules.

---

## Limitations

PredictAI is primarily designed for tabular supervised machine-learning problems.

Potential future improvements include:

- Broader model selection
- More advanced automated feature engineering
- Classification probability calibration
- More detailed SHAP visualizations
- Global feature-importance analysis
- Model persistence and versioning
- Experiment tracking
- Production API deployment
- Authentication and multi-user support
- Monitoring for production model drift

---

## Future Direction

PredictAI can evolve from a local Streamlit machine-learning application into a more complete automated ML platform supporting:

```
Dataset
   ↓
Automated ML
   ↓
Model Selection
   ↓
Explainability
   ↓
Prediction API
   ↓
Deployment
   ↓
Monitoring
```

The current implementation establishes the core end-to-end machine-learning workflow required for that progression.

---

## License

Add the project's chosen license here before public release.