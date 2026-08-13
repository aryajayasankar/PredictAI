from pathlib import Path

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

from src.models.production import (
    train_final_model,
    evaluate_final_model,
    save_model,
    load_model,
)


def main():

    print(
        "\n=== PredictAI Production Pipeline Test ===\n"
    )

    df = load_breast_cancer_dataset()

    target_column = "target"

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    problem_type = detect_problem_type(
        df,
        target_column,
    )

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

    # Use Logistic Regression here because
    # it was the best model in our CV comparison.
    selected_model = models[
        "Logistic Regression"
    ]

    print(
        "Training final model..."
    )

    pipeline = train_final_model(
        preprocessor=preprocessor,
        model=selected_model,
        X_train=X_train,
        y_train=y_train,
    )

    print(
        "Final model trained."
    )

    results = evaluate_final_model(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        problem_type=problem_type,
    )

    print(
        "\n=== Final Test Results ==="
    )

    for metric, value in results.items():
        print(
            f"{metric}: {value}"
        )

    model_path = save_model(
        pipeline=pipeline,
        problem_type=problem_type,
        target_column=target_column,
    )

    print(
        f"\nModel saved to: {model_path}"
    )

    loaded_artifact = load_model(
        model_path
    )

    assert Path(model_path).exists()
    assert "pipeline" in loaded_artifact
    assert (
        loaded_artifact["problem_type"]
        == problem_type
    )
    assert (
        loaded_artifact["target_column"]
        == target_column
    )

    print(
        "\nSaved model loaded successfully."
    )

    print(
        "\nProduction pipeline test passed!"
    )


if __name__ == "__main__":
    main()