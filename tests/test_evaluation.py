import pandas as pd

from src.data.profiler import detect_problem_type
from src.preprocessing.pipeline import (
    build_preprocessing_pipeline,
)
from src.models.registry import get_models
from src.models.trainer import (
    split_data,
    build_model_pipeline,
    train_model,
    predict,
    predict_proba,
)
from src.evaluation.metrics import evaluate_model


DATA_PATH = "data/sample_customer_data.csv"


def main():
    print("\n=== PredictAI Evaluation Test ===\n")

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

    for model_name, model in models.items():

        print(f"\n--- {model_name} ---")

        pipeline = build_model_pipeline(
            preprocessor,
            model,
        )

        trained_pipeline = train_model(
            pipeline,
            X_train,
            y_train,
        )

        predictions = predict(
            trained_pipeline,
            X_test,
        )

        probabilities = predict_proba(
            trained_pipeline,
            X_test,
        )

        results = evaluate_model(
            problem_type,
            y_test,
            predictions,
            probabilities,
        )

        for metric, value in results.items():
            print(f"{metric}: {value}")

        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results
        assert "confusion_matrix" in results

    print("\nAll evaluation tests passed!")


if __name__ == "__main__":
    main()