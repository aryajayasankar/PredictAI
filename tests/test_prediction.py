import pandas as pd

from src.data.datasets import (
    load_breast_cancer_dataset,
)

from src.preprocessing.pipeline import (
    build_preprocessing_pipeline,
)

from src.models.registry import (
    get_models,
)

from src.models.trainer import (
    split_data,
)

from src.models.production import (
    train_final_model,
)

from src.prediction.predictor import (
    generate_predictions,
    validate_prediction_data,
)


def main():

    print(
        "\n=== PredictAI Prediction Test ===\n"
    )

    df = load_breast_cancer_dataset()

    target_column = "target"

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    X_train, X_test, y_train, y_test = (
        split_data(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=True,
        )
    )

    preprocessor = (
        build_preprocessing_pipeline(
            X_train
        )
    )

    model = get_models(
        "classification"
    )["Logistic Regression"]

    pipeline = train_final_model(
        preprocessor=preprocessor,
        model=model,
        X_train=X_train,
        y_train=y_train,
    )

    print(
        "Model trained successfully."
    )

    # Test normal prediction
    predictions = generate_predictions(
        pipeline,
        X_test,
    )

    print(
        f"Predictions generated: "
        f"{len(predictions)}"
    )

    assert len(predictions) == len(X_test)

    assert "prediction" in predictions.columns

    print(
        "Normal prediction test passed."
    )

    # Test different column order
    shuffled_columns = list(
        reversed(X_test.columns)
    )

    X_shuffled = X_test[
        shuffled_columns
    ]

    validated = validate_prediction_data(
        pipeline,
        X_shuffled,
    )

    assert list(
        validated.columns
    ) == list(
        X_train.columns
    )

    print(
        "Column-order adaptation passed."
    )

    # Test extra column
    X_extra = X_test.copy()

    X_extra["extra_column"] = 123

    validated_extra = (
        validate_prediction_data(
            pipeline,
            X_extra,
        )
    )

    assert (
        "extra_column"
        not in validated_extra.columns
    )

    print(
        "Extra-column handling passed."
    )

    # Test missing column
    X_missing = X_test.drop(
        columns=[
            X_test.columns[0]
        ]
    )

    try:

        validate_prediction_data(
            pipeline,
            X_missing,
        )

        raise AssertionError(
            "Missing column was not detected."
        )

    except ValueError:

        print(
            "Missing-column validation passed."
        )

    print(
        "\nAll prediction tests passed!"
    )


if __name__ == "__main__":
    main()