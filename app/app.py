import sys
from pathlib import Path
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import streamlit as st
import pandas as pd

from src.data.profiler import (
    profile_dataset,
    get_column_summary,
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
    build_model_pipeline,
    train_model,
    predict,
    predict_proba,
)

from src.models.comparator import (
    compare_models,
)

from src.models.tuner import (
    tune_xgboost,
    build_tuned_xgboost,
    get_default_n_trials,
)

from src.evaluation.metrics import (
    evaluate_model,
)

from src.explainability.shap_explainer import (
    get_feature_contributions,
)


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PredictAI",
    page_icon="🤖",
    layout="wide",
)


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def clear_training_state():
    """
    Clear all model/training state when the
    training dataset or target changes.
    """

    keys_to_clear = [
        "results",
        "X_train",
        "X_test",
        "y_train",
        "y_test",
        "preprocessor",
        "models",
        "problem_type",
        "target_column",
        "pipeline",
        "evaluation",
        "prediction_results",
        "prediction_file_signature",
        "optuna_study",
        "tuned_xgboost",
    ]

    for key in keys_to_clear:
        st.session_state.pop(key, None)


def get_file_signature(uploaded_file):
    """
    Create a stable signature for an uploaded file.
    This allows us to detect when the uploaded
    dataset actually changes.
    """

    if uploaded_file is None:
        return None

    file_bytes = uploaded_file.getvalue()

    return hashlib.md5(
        file_bytes
    ).hexdigest()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("PredictAI")

st.subheader(
    "End-to-End Machine Learning Prediction Platform"
)

st.write(
    "Upload a CSV dataset, select a target, "
    "compare machine-learning models, and "
    "generate predictions."
)


# --------------------------------------------------
# Training Dataset Upload
# --------------------------------------------------

st.sidebar.header("Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"],
    key="training_csv",
)


# --------------------------------------------------
# Detect Training Dataset Changes
# --------------------------------------------------

current_training_signature = (
    get_file_signature(uploaded_file)
)

previous_training_signature = (
    st.session_state.get(
        "training_file_signature"
    )
)

if uploaded_file is None:

    if previous_training_signature is not None:

        clear_training_state()

        st.session_state.pop(
            "training_file_signature",
            None,
        )

else:

    if (
        previous_training_signature is not None
        and previous_training_signature
        != current_training_signature
    ):

        clear_training_state()

    st.session_state[
        "training_file_signature"
    ] = current_training_signature


# --------------------------------------------------
# No Dataset
# --------------------------------------------------

if uploaded_file is None:

    st.info(
        "Upload a CSV dataset from the sidebar "
        "to begin."
    )

    st.stop()


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

try:

    df = pd.read_csv(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Could not read the CSV: {e}"
    )

    st.stop()


if df.empty:

    st.error(
        "The uploaded dataset is empty."
    )

    st.stop()


# --------------------------------------------------
# Dataset Overview
# --------------------------------------------------

st.header("Dataset Overview")

profile = profile_dataset(df)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Rows",
    profile["rows"],
)

col2.metric(
    "Columns",
    profile["columns"],
)

col3.metric(
    "Missing Values",
    profile["missing_values"],
)

col4.metric(
    "Duplicate Rows",
    profile["duplicate_rows"],
)


with st.expander("Preview Dataset"):

    st.dataframe(
        df.head(20),
        use_container_width=True,
    )


with st.expander("Column Summary"):

    st.dataframe(
        get_column_summary(df),
        use_container_width=True,
    )


# --------------------------------------------------
# Target Selection
# --------------------------------------------------

st.header("Model Configuration")

target_index = (
    list(df.columns).index("target")
    if "target" in df.columns
    else 0
)

target_column = st.selectbox(
    "Select target column",
    options=df.columns,
    index=target_index,
)


# --------------------------------------------------
# Detect Target Changes
# --------------------------------------------------

previous_target = st.session_state.get(
    "selected_target"
)

if (
    previous_target is not None
    and previous_target != target_column
):

    clear_training_state()

st.session_state[
    "selected_target"
] = target_column


# --------------------------------------------------
# Detect Problem Type
# --------------------------------------------------

try:

    problem_type = detect_problem_type(
        df,
        target_column,
    )

except ValueError as e:

    st.error(str(e))

    st.stop()


st.success(
    f"Detected problem type: "
    f"**{problem_type.title()}**"
)


# --------------------------------------------------
# Dataset Information
# --------------------------------------------------

with st.expander(
    "Dataset Information"
):

    st.write(
        f"Target: `{target_column}`"
    )

    st.write(
        f"Problem type: `{problem_type}`"
    )

    st.write(
        f"Rows: `{len(df)}`"
    )

    st.write(
        f"Columns: `{len(df.columns)}`"
    )


# --------------------------------------------------
# Training
# --------------------------------------------------

if st.button(
    "Train & Compare Models",
    type="primary",
):

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    stratify = (
        problem_type == "classification"
    )

    X_train, X_test, y_train, y_test = (
        split_data(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
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

    with st.spinner(
        "Comparing models..."
    ):

        results = compare_models(
            models=models,
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            problem_type=problem_type,
            cv=5,
        )

    st.session_state[
        "results"
    ] = results

    st.session_state[
        "X_train"
    ] = X_train

    st.session_state[
        "X_test"
    ] = X_test

    st.session_state[
        "y_train"
    ] = y_train

    st.session_state[
        "y_test"
    ] = y_test

    st.session_state[
        "preprocessor"
    ] = preprocessor

    st.session_state[
        "models"
    ] = models

    st.session_state[
        "problem_type"
    ] = problem_type

    st.session_state[
        "target_column"
    ] = target_column

    st.session_state.pop(
        "optuna_study",
        None,
    )

    st.session_state.pop(
        "tuned_xgboost",
        None,
    )

    # Clear anything from a previous model
    st.session_state.pop(
        "pipeline",
        None,
    )

    st.session_state.pop(
        "evaluation",
        None,
    )

    st.session_state.pop(
        "prediction_results",
        None,
    )

    st.success(
        "Model comparison completed."
    )


# --------------------------------------------------
# Model Comparison Results
# --------------------------------------------------

if "results" in st.session_state:

    st.header("Model Comparison")

    results = st.session_state[
        "results"
    ]

    st.dataframe(
        results,
        use_container_width=True,
    )

    best_model_name = results.iloc[0][
        "model"
    ]

    st.success(
        f"Best model: **{best_model_name}**"
    )





# --------------------------------------------------
# Hyperparameter Optimization
# --------------------------------------------------

if "results" in st.session_state:

    st.header("Hyperparameter Optimization")

    st.write(
        "PredictAI automatically optimizes "
        "XGBoost hyperparameters using Optuna."
    )

    st.caption(
        "The number of optimization trials is "
        "automatically selected based on dataset size."
    )

    X_train = st.session_state[
        "X_train"
    ]

    y_train = st.session_state[
        "y_train"
    ]

    preprocessor = st.session_state[
        "preprocessor"
    ]

    problem_type = st.session_state[
        "problem_type"
    ]






    automatic_trials = (
        get_default_n_trials(
            len(X_train)
        )
    )

    # --------------------------------------------------
    # Smart Optimization Decision
    # --------------------------------------------------

    # Determine the primary metric used for model comparison.
    if problem_type == "classification":
        primary_metric = "f1"
    else:
        primary_metric = "r2"

    best_baseline_score = float(
        st.session_state["results"].iloc[0][
            primary_metric
        ]
    )

    PERFECT_SCORE_THRESHOLD = 0.999999

    if best_baseline_score >= PERFECT_SCORE_THRESHOLD:

        st.success(
            f"🏆 Best baseline model already achieved "
            f"a perfect CV {primary_metric.upper()} score "
            f"of **{best_baseline_score:.4f}**."
        )

        st.info(
            "Hyperparameter optimization was "
            "automatically skipped because the "
            "selected CV metric is already perfect."
        )

        st.caption(
            "For real-world datasets, perfect validation "
            "performance may indicate a highly deterministic "
            "dataset or possible target leakage."
        )

        # --------------------------------------------------
        # Skip Optuna
        # --------------------------------------------------

        st.session_state[
            "optimization_skipped"
        ] = True

        st.session_state[
            "optuna_study"
        ] = None

        st.session_state[
            "tuned_xgboost"
        ] = None



    else:

        st.session_state[
            "optimization_skipped"
        ] = False

        st.info(
            f"Automatic optimization budget: "
            f"**{automatic_trials} trials**"
        )

        if st.button(
            "Optimize XGBoost",
            type="primary",
        ):

            progress_bar = st.progress(
                0
            )

            progress_text = st.empty()

            best_score_text = st.empty()

            technical_logs = []

            log_container = st.empty()

            def update_progress(
                trial_number,
                total_trials,
                trial_value,
                best_value,
                best_params,
            ):

                progress = (
                    trial_number
                    / total_trials
                )

                progress_bar.progress(
                    progress
                )

                progress_text.write(
                    f"Trial {trial_number} "
                    f"/ {total_trials}"
                )

                best_score_text.write(
                    f"Best CV score: "
                    f"**{best_value:.4f}**"
                )

                technical_logs.append(
                    f"Trial {trial_number} "
                    f"completed | "
                    f"Score: {trial_value:.4f} | "
                    f"Best: {best_value:.4f}"
                )

                log_container.code(
                    "\n".join(
                        technical_logs[-10:]
                    ),
                    language="text",
                )

            with st.status(
                "Optimizing XGBoost...",
                expanded=True,
            ) as status:

                st.write(
                    "Running cross-validated "
                    "hyperparameter optimization."
                )

                try:

                    study = tune_xgboost(
                        preprocessor=preprocessor,
                        X_train=X_train,
                        y_train=y_train,
                        problem_type=problem_type,
                        n_trials=automatic_trials,
                        cv=5,
                        progress_callback=(
                            update_progress
                        ),
                    )

                    tuned_model = (
                        build_tuned_xgboost(
                            study,
                            problem_type,
                        )
                    )

                    st.session_state[
                        "optuna_study"
                    ] = study

                    st.session_state[
                        "tuned_xgboost"
                    ] = tuned_model

                    status.update(
                        label=(
                            "XGBoost optimization "
                            "completed"
                        ),
                        state="complete",
                    )

                except Exception as e:

                    status.update(
                        label=(
                            "Optimization failed"
                        ),
                        state="error",
                    )

                    st.error(
                        f"Optuna failed: {e}"
                    )


# --------------------------------------------------
# Optuna Results
# --------------------------------------------------

if (
    st.session_state.get("optuna_study")
    is not None
):

    study = st.session_state[
        "optuna_study"
    ]

    st.subheader(
        "Optimization Results"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Trials Completed",
        len(study.trials),
    )

    col2.metric(
        "Best CV Score",
        f"{study.best_value:.4f}",
    )

    st.write(
        "**Best Hyperparameters**"
    )

    best_params_df = pd.DataFrame(
        [
            {
                "parameter": key,
                "value": value,
            }
            for key, value
            in study.best_params.items()
        ]
    )

    st.dataframe(
        best_params_df,
        use_container_width=True,
    )


# --------------------------------------------------
# Optimization Not Yet Run
# --------------------------------------------------

if (
    "results" in st.session_state
    and st.session_state.get("optuna_study") is None
    and not st.session_state.get(
        "optimization_skipped",
        False,
    )
):

    st.info(
        "Run hyperparameter optimization to "
        "continue to final model selection."
    )

# --------------------------------------------------
# Final Model Selection & Training
# --------------------------------------------------

if (
    "results" in st.session_state
    and "optuna_study" in st.session_state
    ):

    st.header("Final Model Selection")

    results = st.session_state[
        "results"
    ]

    problem_type = st.session_state[
        "problem_type"
    ]

    # ----------------------------------------------
    # Determine primary metric
    # ----------------------------------------------

    if problem_type == "classification":

        primary_metric = "f1"

    else:

        primary_metric = "r2"

    # ----------------------------------------------
    # Build candidate table
    # ----------------------------------------------

    candidate_rows = []

    best_baseline = results.iloc[0]

    candidate_rows.append(
        {
            "model": best_baseline["model"],
            "score": best_baseline[primary_metric],
            "source": "Baseline comparison",
        }
    )

    # ----------------------------------------------
    # Add tuned XGBoost only if Optuna ran
    # ----------------------------------------------

    study = st.session_state.get(
        "optuna_study"
    )

    if study is not None:

        tuned_score = float(
            study.best_value
        )

        candidate_rows.append(
            {
                "model": "Tuned XGBoost",
                "score": tuned_score,
                "source": "Optuna",
            }
        )

    candidate_df = pd.DataFrame(
        candidate_rows
    )

    

    # --------------------------------------------------
    # Candidate Model Selection
    # --------------------------------------------------

    TIE_TOLERANCE = 1e-4

    baseline_rows = candidate_df[
        candidate_df["source"]
        == "Baseline comparison"
    ]

    baseline_score = (
        float(baseline_rows["score"].iloc[0])
        if not baseline_rows.empty
        else None
    )

    tuned_rows = candidate_df[
        candidate_df["source"]
        == "Optuna"
    ]

    if (
        not tuned_rows.empty
        and baseline_score is not None
    ):

        tuned_score = float(
            tuned_rows["score"].iloc[0]
        )

        score_difference = (
            tuned_score - baseline_score
        )

        if abs(score_difference) <= TIE_TOLERANCE:

            candidate_df["tie"] = False

            candidate_df.loc[
                candidate_df["source"]
                == "Baseline comparison",
                "tie",
            ] = True

            candidate_df = candidate_df.sort_values(
                by=["tie", "score"],
                ascending=[False, False],
            )

        else:

            candidate_df = candidate_df.sort_values(
                by="score",
                ascending=False,
            )

    else:

        candidate_df = candidate_df.sort_values(
            by="score",
            ascending=False,
        )

    candidate_df = candidate_df.drop(
        columns=["tie"],
        errors="ignore",
    ).reset_index(
        drop=True
    )

    st.subheader(
        "Candidate Models"
    )

    st.dataframe(
        candidate_df,
        use_container_width=True,
    )

    # ----------------------------------------------
    # Select winner
    # ----------------------------------------------

    winning_candidate = candidate_df.iloc[0]

    final_model_name = winning_candidate["model"]

    final_score = float(
        winning_candidate["score"]
    )

    st.success(
        f"🏆 Selected model: **{final_model_name}** "
        f"(CV {primary_metric.upper()}: "
        f"{final_score:.4f})"
    )

    st.session_state[
        "selected_final_model"
    ] = final_model_name

    # ----------------------------------------------
    # Train final model
    # ----------------------------------------------

    if st.button("Train Selected Final Model", type="primary"):
        X_train = st.session_state["X_train"]
        y_train = st.session_state["y_train"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        preprocessor = st.session_state["preprocessor"]
        models = st.session_state["models"]

        # ----------------------------------------------
        # Select final model
        # ----------------------------------------------
        if final_model_name == "Tuned XGBoost":
            model = st.session_state["tuned_xgboost"]
        else:
            model = models[final_model_name]

        # ----------------------------------------------
        # Training progress
        # ----------------------------------------------
        with st.status("Training selected final model...", expanded=True) as status:
            st.write(f"**Selected model:** {final_model_name}")

            # Step 1
            st.write("⟳ Building preprocessing pipeline...")
            pipeline = build_model_pipeline(preprocessor, model)
            st.write("✓ Preprocessing pipeline ready.")

            # Step 2
            st.write("⟳ Training final model...")
            pipeline = train_model(pipeline, X_train, y_train)
            st.write("✓ Final model trained.")

            # Step 3
            st.write("⟳ Generating test predictions...")
            predictions = predict(pipeline, X_test)
            probabilities = predict_proba(pipeline, X_test)
            st.write("✓ Predictions generated.")

            # Step 4
            st.write("⟳ Evaluating final model...")
            evaluation = evaluate_model(
                problem_type,
                y_test,
                predictions,
                probabilities,
            )
            st.write("✓ Model evaluation completed.")

            # ------------------------------------------
            # Save final state
            # ------------------------------------------
            st.session_state["pipeline"] = pipeline
            st.session_state["evaluation"] = evaluation
            st.session_state["final_model_name"] = final_model_name

            # Clear stale prediction state
            st.session_state.pop("prediction_results", None)
            st.session_state.pop("prediction_file_signature", None)

            status.update(
                label="Final model training completed",
                state="complete",
            )

        





# --------------------------------------------------
# Evaluation
# --------------------------------------------------

if "evaluation" in st.session_state:

    st.header("Model Evaluation")

    evaluation = st.session_state[
        "evaluation"
    ]

    display_metrics = {
        key: value
        for key, value in evaluation.items()
        if key != "confusion_matrix"
    }

    columns = st.columns(
        len(display_metrics)
    )

    for column, (
        metric,
        value,
    ) in zip(
        columns,
        display_metrics.items(),
    ):

        if value is not None:

            column.metric(
                metric.upper(),
                f"{value:.4f}",
            )




# --------------------------------------------------
# Model Explainability
# --------------------------------------------------

if (
    "pipeline" in st.session_state
    and "X_train" in st.session_state
    and "X_test" in st.session_state
):

    st.header("Model Explainability")

    st.write(
        "Understand which features contributed most "
        "to an individual prediction using SHAP."
    )

    X_train = st.session_state["X_train"]
    X_test = st.session_state["X_test"]
    pipeline = st.session_state["pipeline"]
    problem_type = st.session_state["problem_type"]

    # --------------------------------------------------
    # Test Set Information
    # --------------------------------------------------

    st.info(
        f"**Test set:** {len(X_test):,} samples  |  "
        f"**Features:** {X_test.shape[1]:,}"
    )

    # --------------------------------------------------
    # Prediction Selection
    # --------------------------------------------------

    sample_number = st.selectbox(
        "Select prediction to explain",
        options=list(
            range(1, len(X_test) + 1)
        ),
        format_func=lambda x:
            f"Prediction #{x}",
    )

    sample_index = sample_number - 1

    sample = X_test.iloc[
        [sample_index]
    ]

    # --------------------------------------------------
    # Show Selected Sample
    # --------------------------------------------------

    with st.expander(
        "View selected sample"
    ):

        st.dataframe(
            sample,
            use_container_width=True,
        )

    # --------------------------------------------------
    # Explain Prediction Button
    # --------------------------------------------------

    if st.button(
        "Explain Prediction",
        type="primary",
        key="explain_prediction_button",
    ):

        try:

            with st.spinner(
                "Calculating prediction explanation..."
            ):

                # ------------------------------------------
                # Generate prediction
                # ------------------------------------------

                prediction = predict(
                    pipeline,
                    sample,
                )

                probability = predict_proba(
                    pipeline,
                    sample,
                )

                # ------------------------------------------
                # Calculate SHAP contributions
                # ------------------------------------------

                contributions = (
                    get_feature_contributions(
                        pipeline,
                        X_train,
                        sample,
                    )
                )

            # ------------------------------------------
            # Success
            # ------------------------------------------

            st.success(
                "Prediction explanation generated."
            )

            # ------------------------------------------
            # Prediction Summary
            # ------------------------------------------

            st.subheader(
                "Prediction"
            )

            if problem_type == "classification":

                predicted_class = prediction[0]

                st.metric(
                    "Predicted Class",
                    str(predicted_class),
                )

                if probability is not None:

                    confidence = (
                        float(
                            probability[0].max()
                        )
                        * 100
                    )

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%",
                    )

            else:

                st.metric(
                    "Predicted Value",
                    f"{float(prediction[0]):.4f}",
                )

            # ------------------------------------------
            # SHAP Contributions
            # ------------------------------------------

            contributions_df = pd.DataFrame(
                contributions,
                columns=[
                    "feature",
                    "contribution",
                ],
            )

            st.subheader(
                "Top Feature Contributions"
            )

            st.dataframe(
                contributions_df.head(10),
                use_container_width=True,
            )

            st.bar_chart(
                contributions_df.head(10)
                .set_index("feature")
            )

        except Exception as e:

            # ------------------------------------------
            # Explainability failure
            # ------------------------------------------

            st.error(
                f"Explainability failed: {e}"
            )

            st.info(
                "The trained model and prediction "
                "workflow are still available."
            )


# --------------------------------------------------
# Prediction
# --------------------------------------------------





if "pipeline" in st.session_state:

    st.header("Make Predictions")

    st.write(
        "Upload new data using the same feature "
        "schema as the training dataset."
    )

    prediction_file = st.file_uploader(
        "Upload prediction CSV",
        type=["csv"],
        key="prediction_csv",
    )

    # --------------------------------------------------
    # Prediction File State
    # --------------------------------------------------

    current_prediction_signature = (
        get_file_signature(
            prediction_file
        )
    )

    previous_prediction_signature = (
        st.session_state.get(
            "prediction_file_signature"
        )
    )

    # --------------------------------------------------
    # No file uploaded
    # --------------------------------------------------

    if prediction_file is None:

        st.session_state.pop(
            "prediction_results",
            None,
        )

        st.session_state.pop(
            "prediction_file_signature",
            None,
        )

    else:

        # --------------------------------------------------
        # New file detected
        # --------------------------------------------------

        if (
            previous_prediction_signature
            != current_prediction_signature
        ):

            # Immediately remove results belonging
            # to the previous file.
            st.session_state.pop(
                "prediction_results",
                None,
            )

            st.session_state.pop(
                "prediction_error",
                None,
            )

        st.session_state[
            "prediction_file_signature"
        ] = current_prediction_signature

        # --------------------------------------------------
        # Read prediction dataset
        # --------------------------------------------------

        try:

            prediction_df = pd.read_csv(
                prediction_file
            )

            st.write(
                f"Prediction dataset: "
                f"**{len(prediction_df):,} rows** | "
                f"**{len(prediction_df.columns):,} columns**"
            )

            with st.expander(
                "Preview Prediction Data"
            ):

                st.dataframe(
                    prediction_df.head(20),
                    use_container_width=True,
                )

            # --------------------------------------------------
            # Generate Predictions
            # --------------------------------------------------

            if st.button(
                "Generate Predictions",
                type="primary",
            ):

                from src.prediction.predictor import (
                    generate_predictions,
                )

                pipeline = st.session_state[
                    "pipeline"
                ]

                # Always clear previous results
                # before attempting a new prediction.
                st.session_state.pop(
                    "prediction_results",
                    None,
                )

                st.session_state.pop(
                    "prediction_error",
                    None,
                )

                with st.status(
                    "Generating predictions...",
                    expanded=True,
                ) as status:

                    try:

                        st.write(
                            "⟳ Validating prediction schema..."
                        )

                        prediction_results = (
                            generate_predictions(
                                pipeline,
                                prediction_df,
                            )
                        )

                        st.write(
                            "✓ Schema validation passed."
                        )

                        st.write(
                            "✓ Predictions generated."
                        )

                        st.session_state[
                            "prediction_results"
                        ] = prediction_results

                        status.update(
                            label=(
                                "Predictions generated "
                                "successfully"
                            ),
                            state="complete",
                        )

                    except ValueError as e:

                        st.session_state.pop(
                            "prediction_results",
                            None,
                        )

                        st.session_state[
                            "prediction_error"
                        ] = (
                            f"Schema validation failed: "
                            f"{e}"
                        )

                        status.update(
                            label=(
                                "Schema validation failed"
                            ),
                            state="error",
                        )

                    except Exception as e:

                        st.session_state.pop(
                            "prediction_results",
                            None,
                        )

                        st.session_state[
                            "prediction_error"
                        ] = (
                            f"Prediction failed: "
                            f"{e}"
                        )

                        status.update(
                            label="Prediction failed",
                            state="error",
                        )

        except Exception as e:

            st.session_state.pop(
                "prediction_results",
                None,
            )

            st.session_state[
                "prediction_error"
            ] = (
                f"Could not read prediction file: "
                f"{e}"
            )

        # --------------------------------------------------
        # Display errors
        # --------------------------------------------------

        if (
            "prediction_error"
            in st.session_state
        ):

            st.error(
                st.session_state[
                    "prediction_error"
                ]
            )


# --------------------------------------------------
# Prediction Results
# --------------------------------------------------

if (
    "prediction_results"
    in st.session_state
):

    st.subheader(
        "Prediction Results"
    )

    prediction_results = (
        st.session_state[
            "prediction_results"
        ]
    )

    st.success(
        f"Generated predictions for "
        f"**{len(prediction_results):,} rows**."
    )

    st.dataframe(
        prediction_results,
        use_container_width=True,
    )

    # --------------------------------------------------
    # Download
    # --------------------------------------------------

    csv_data = (
        prediction_results
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label="Download Predictions CSV",
        data=csv_data,
        file_name="predictions.csv",
        mime="text/csv",
    )