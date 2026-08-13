import pandas as pd

from src.models.trainer import (
    build_model_pipeline,
    cross_validate_model,
)


CLASSIFICATION_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision_weighted",
    "recall": "recall_weighted",
    "f1": "f1_weighted",
}


REGRESSION_SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


def compare_models(
    models,
    preprocessor,
    X_train,
    y_train,
    problem_type,
    cv=5,
):
    """Compare candidate models using cross-validation."""

    if problem_type == "classification":
        scoring = CLASSIFICATION_SCORING
        primary_metric = "f1"

    elif problem_type == "regression":
        scoring = REGRESSION_SCORING
        primary_metric = "r2"

    else:
        raise ValueError(
            f"Unsupported problem type: {problem_type}"
        )

    results = []

    for model_name, model in models.items():

        pipeline = build_model_pipeline(
            preprocessor,
            model,
        )

        scores = cross_validate_model(
            pipeline,
            X_train,
            y_train,
            scoring=scoring,
            cv=cv,
        )

        row = {
            "model": model_name,
        }

        for metric_name, values in scores.items():

            mean_score = values["mean"]

            # sklearn returns negative values
            # for error metrics.
            if metric_name in ["mae", "rmse"]:
                mean_score = -mean_score

            row[metric_name] = mean_score
            row[f"{metric_name}_std"] = values["std"]

        results.append(row)

    results_df = pd.DataFrame(results)

    if problem_type == "classification":
        results_df = results_df.sort_values(
            by=primary_metric,
            ascending=False,
        )

    else:
        results_df = results_df.sort_values(
            by=primary_metric,
            ascending=False,
        )

    results_df = results_df.reset_index(drop=True)

    return results_df