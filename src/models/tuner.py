import optuna

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from xgboost import XGBClassifier, XGBRegressor


def tune_xgboost(
    preprocessor,
    X_train,
    y_train,
    problem_type,
    n_trials=20,
    cv=5,
):
    """
    Tune XGBoost hyperparameters using Optuna.
    """

    def objective(trial):

        n_estimators = trial.suggest_int(
            "n_estimators",
            100,
            500,
        )

        max_depth = trial.suggest_int(
            "max_depth",
            2,
            8,
        )

        learning_rate = trial.suggest_float(
            "learning_rate",
            0.01,
            0.3,
            log=True,
        )

        subsample = trial.suggest_float(
            "subsample",
            0.6,
            1.0,
        )

        colsample_bytree = trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0,
        )

        if problem_type == "classification":

            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )

            scoring = "f1_weighted"

        elif problem_type == "regression":

            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            )

            scoring = "r2"

        else:
            raise ValueError(
                f"Unsupported problem type: {problem_type}"
            )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    clone(preprocessor),
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
    )

    study.optimize(
        objective,
        n_trials=n_trials,
    )

    return study