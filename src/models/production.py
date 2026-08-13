from pathlib import Path

import joblib

from src.models.trainer import (
    build_model_pipeline,
    train_model,
    predict,
    predict_proba,
)

from src.evaluation.metrics import evaluate_model


def train_final_model(
    preprocessor,
    model,
    X_train,
    y_train,
):
    """Train the final preprocessing + model pipeline."""

    pipeline = build_model_pipeline(
        preprocessor,
        model,
    )

    return train_model(
        pipeline,
        X_train,
        y_train,
    )


def evaluate_final_model(
    pipeline,
    X_test,
    y_test,
    problem_type,
):
    """Evaluate the final model on untouched test data."""

    predictions = predict(
        pipeline,
        X_test,
    )

    probabilities = predict_proba(
        pipeline,
        X_test,
    )

    results = evaluate_model(
        problem_type,
        y_test,
        predictions,
        probabilities,
    )

    return results


def save_model(
    pipeline,
    problem_type,
    target_column,
    path="models/model.joblib",
):
    """Save the complete trained pipeline."""

    model_path = Path(path)

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "pipeline": pipeline,
        "problem_type": problem_type,
        "target_column": target_column,
    }

    joblib.dump(
        artifact,
        model_path,
    )

    return model_path


def load_model(path="models/model.joblib"):
    """Load a saved model artifact."""

    return joblib.load(path)