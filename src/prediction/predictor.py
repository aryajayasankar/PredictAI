import pandas as pd


def validate_prediction_data(
    pipeline,
    X_new,
):
    """
    Validate that prediction data matches
    the schema expected by the trained pipeline.
    """

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    expected_columns = list(
        preprocessor.feature_names_in_
    )

    actual_columns = list(
        X_new.columns
    )

    missing_columns = [
        column
        for column in expected_columns
        if column not in actual_columns
    ]

    extra_columns = [
        column
        for column in actual_columns
        if column not in expected_columns
    ]

    if missing_columns:
        raise ValueError(
            "Prediction data is missing required "
            f"columns: {missing_columns}"
        )

    if extra_columns:
        X_new = X_new.drop(
            columns=extra_columns
        )

    X_new = X_new[
        expected_columns
    ]

    return X_new


def generate_predictions(
    pipeline,
    X_new,
):
    """
    Validate input data and generate predictions.
    """

    X_validated = validate_prediction_data(
        pipeline,
        X_new,
    )

    predictions = pipeline.predict(
        X_validated
    )

    result = X_validated.copy()

    result["prediction"] = predictions

    return result