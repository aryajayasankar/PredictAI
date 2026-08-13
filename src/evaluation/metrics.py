import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)


def evaluate_classification(
    y_true,
    y_pred,
    y_proba=None,
):
    """Calculate classification metrics."""

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
        ),
    }

    if y_proba is not None:
        try:
            if y_proba.shape[1] == 2:
                results["roc_auc"] = roc_auc_score(
                    y_true,
                    y_proba[:, 1],
                )
            else:
                results["roc_auc"] = roc_auc_score(
                    y_true,
                    y_proba,
                    multi_class="ovr",
                )
        except ValueError:
            results["roc_auc"] = None

    return results


def evaluate_regression(y_true, y_pred):
    """Calculate regression metrics."""

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    return {
        "mae": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "rmse": rmse,
        "r2": r2_score(
            y_true,
            y_pred,
        ),
    }


def evaluate_model(
    problem_type,
    y_true,
    y_pred,
    y_proba=None,
):
    """Automatically select appropriate metrics."""

    if problem_type == "classification":
        return evaluate_classification(
            y_true,
            y_pred,
            y_proba,
        )

    if problem_type == "regression":
        return evaluate_regression(
            y_true,
            y_pred,
        )

    raise ValueError(
        f"Unsupported problem type: {problem_type}"
    )