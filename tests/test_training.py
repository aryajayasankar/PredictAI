import pandas as pd

from sklearn.metrics import accuracy_score

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
)


DATA_PATH = "data/sample_customer_data.csv"


def main():
    print("\n=== PredictAI Training Engine Test ===\n")

    df = pd.read_csv(DATA_PATH)

    target_column = "churn"

    X = df.drop(columns=[target_column])
    y = df[target_column]

    problem_type = detect_problem_type(
        df,
        target_column,
    )

    print(f"Problem type: {problem_type}")

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=True,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    preprocessor = build_preprocessing_pipeline(
        X_train
    )

    models = get_models(problem_type)

    for model_name, model in models.items():

        print(f"\nTraining: {model_name}")

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

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        print(f"Accuracy: {accuracy:.4f}")

        assert len(predictions) == len(y_test)

    print("\nAll training tests passed!")


if __name__ == "__main__":
    main()