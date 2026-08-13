import pandas as pd

from src.data.profiler import detect_problem_type
from src.preprocessing.pipeline import (
    build_preprocessing_pipeline,
)
from src.models.trainer import split_data
from src.models.tuner import tune_xgboost


DATA_PATH = "data/sample_customer_data.csv"


def main():

    print("\n=== PredictAI Hyperparameter Tuning Test ===\n")

    df = pd.read_csv(DATA_PATH)

    target_column = "churn"

    X = df.drop(columns=[target_column])
    y = df[target_column]

    problem_type = detect_problem_type(
        df,
        target_column,
    )

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=True,
    )

    preprocessor = build_preprocessing_pipeline(
        X_train
    )

    print("Starting Optuna optimization...")

    study = tune_xgboost(
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        problem_type=problem_type,
        n_trials=3,
        cv=3,
    )

    print("\nBest score:")
    print(study.best_value)

    print("\nBest parameters:")
    for parameter, value in study.best_params.items():
        print(f"{parameter}: {value}")

    assert study.best_trial is not None
    assert len(study.best_params) > 0

    print("\nHyperparameter tuning test passed!")


if __name__ == "__main__":
    main()