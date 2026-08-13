from src.data.datasets import (
    load_breast_cancer_dataset,
)

from src.data.profiler import (
    detect_problem_type,
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

from src.models.comparator import (
    compare_models,
)


def main():

    print(
        "\n=== Real Classification Test ===\n"
    )

    df = load_breast_cancer_dataset()

    print(f"Dataset shape: {df.shape}")

    target_column = "target"

    problem_type = detect_problem_type(
        df,
        target_column,
    )

    print(
        f"Detected problem type: "
        f"{problem_type}"
    )

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

    models = get_models(
        problem_type
    )

    results = compare_models(
        models=models,
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        problem_type=problem_type,
        cv=5,
    )

    print(
        "\n=== Classification Results ===\n"
    )

    print(
        results[
            [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(index=False)
    )

    best_model = results.iloc[0]["model"]

    print(
        f"\nBest model: {best_model}"
    )

    assert problem_type == "classification"
    assert len(results) == 3
    assert results["f1"].notna().all()

    print(
        "\nReal classification test passed!"
    )


if __name__ == "__main__":
    main()