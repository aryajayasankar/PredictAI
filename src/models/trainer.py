from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline


def split_data(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=True,
):
    """Split features and target."""

    stratify_value = y if stratify else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_value,
    )


def build_model_pipeline(preprocessor, model):
    """Combine preprocessing and model."""

    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", clone(model)),
        ]
    )


def cross_validate_model(
    pipeline,
    X_train,
    y_train,
    scoring,
    cv=5,
):
    """Perform cross-validation."""

    results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    summary = {}

    for metric_name in scoring:
        test_key = f"test_{metric_name}"

        summary[metric_name] = {
            "mean": results[test_key].mean(),
            "std": results[test_key].std(),
        }

    return summary


def train_model(pipeline, X_train, y_train):
    """Fit the complete pipeline."""

    pipeline.fit(X_train, y_train)

    return pipeline


def predict(pipeline, X):
    """Generate predictions."""

    return pipeline.predict(X)


def predict_proba(pipeline, X):
    """Generate probabilities when supported."""

    if not hasattr(pipeline, "predict_proba"):
        return None

    return pipeline.predict_proba(X)