import pandas as pd

from src.data.profiler import detect_problem_type
from src.preprocessing.pipeline import (
    build_preprocessing_pipeline,
)
from src.models.registry import get_models
from src.models.trainer import split_data
from src.models.comparator import compare_models


DATA_PATH = "data/sample_customer_data.csv"


def main():

    print("\n=== PredictAI Model Comparison Test ===\n")

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

    models = get_models(problem_type)

    # Small dataset → fewer CV folds.
    results = compare_models(
        models=models,
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        problem_type=problem_type,
        cv=3,
    )

    print("=== Model Comparison ===\n")
    print(results.to_string(index=False))

    best_model = results.iloc[0]["model"]

    print(f"\nBest model: {best_model}")

    assert len(results) == len(models)
    assert best_model in models

    print("\nAll comparison tests passed!")


if __name__ == "__main__":
    main()