import numpy as np
import shap

from xgboost import (
    XGBClassifier,
    XGBRegressor,
)


def transform_features(pipeline, X):
    """Transform raw features using the fitted preprocessor."""

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    return preprocessor.transform(X)


def get_feature_names(pipeline):
    """Get feature names after preprocessing."""

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    return preprocessor.get_feature_names_out()


def create_explainer(
    pipeline,
    X_background=None,
):
    """
    Create an appropriate SHAP explainer
    for the trained model.
    """

    model = pipeline.named_steps[
        "model"
    ]

    # --------------------------------------------------
    # XGBoost
    # --------------------------------------------------

    if isinstance(
        model,
        (
            XGBClassifier,
            XGBRegressor,
        ),
    ):

        # XGBoost's native Tree SHAP implementation
        # supports categorical split handling through
        # tree_path_dependent.
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation=(
                "tree_path_dependent"
            ),
        )

        return explainer

    # --------------------------------------------------
    # Other models
    # --------------------------------------------------

    if X_background is None:
        raise ValueError(
            "Background data is required "
            "for this model."
        )

    X_transformed = transform_features(
        pipeline,
        X_background,
    )

    explainer = shap.Explainer(
        model,
        X_transformed,
    )

    return explainer


def calculate_shap_values(
    pipeline,
    X_background,
    X_samples,
):
    """Calculate SHAP values."""

    explainer = create_explainer(
        pipeline,
        X_background,
    )

    X_transformed = transform_features(
        pipeline,
        X_samples,
    )

    shap_values = explainer(
        X_transformed
    )

    return shap_values


def get_feature_contributions(
    pipeline,
    X_background,
    X_sample,
):
    """
    Return feature contributions
    for a single sample.
    """

    shap_values = calculate_shap_values(
        pipeline,
        X_background,
        X_sample,
    )

    feature_names = get_feature_names(
        pipeline
    )

    values = shap_values.values

    # --------------------------------------------------
    # Handle multi-output SHAP values
    # --------------------------------------------------

    if values.ndim == 3:

        # Binary classification:
        # use positive class.
        values = values[:, :, 1]

    contributions = np.abs(
        values[0]
    )

    result = sorted(
        zip(
            feature_names,
            contributions,
        ),
        key=lambda x: x[1],
        reverse=True,
    )

    return result