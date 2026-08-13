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

from src.explainability.shap_explainer import (
    get_feature_contributions,
)


def main():

    print(
        "\n=== PredictAI Explainability Test ===\n"
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

    sample = X_test.iloc[[0]]

    contributions = get_feature_contributions(
        pipeline=pipeline,
        X_background=X_train.iloc[:100],
        X_sample=sample,
    )

    print(
        "\nTop feature contributions:\n"
    )

    for feature, contribution in contributions[:10]:
        print(
            f"{feature}: {contribution:.6f}"
        )

    assert len(contributions) > 0

    print(
        "\nExplainability test passed!"
    )


if __name__ == "__main__":
    main()