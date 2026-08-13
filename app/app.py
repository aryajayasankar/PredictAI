import sys
from pathlib import Path

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

from src.evaluation.metrics import (
    evaluate_model,
)


st.set_page_config(
    page_title="PredictAI",
    page_icon="🤖",
    layout="wide",
)


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
# Upload
# --------------------------------------------------

st.sidebar.header("Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"],
)


if uploaded_file is None:

    st.info(
        "Upload a CSV dataset from the sidebar "
        "to begin."
    )

    st.stop()


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

try:

    df = pd.read_csv(uploaded_file)

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

    st.session_state["results"] = results
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test
    st.session_state["preprocessor"] = preprocessor
    st.session_state["models"] = models
    st.session_state["problem_type"] = problem_type
    st.session_state["target_column"] = target_column

    st.success(
        "Model comparison completed."
    )


# --------------------------------------------------
# Results
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
# Final model
# --------------------------------------------------

if "results" in st.session_state:

    if st.button(
        "Train Best Model",
    ):

        X_train = st.session_state[
            "X_train"
        ]

        y_train = st.session_state[
            "y_train"
        ]

        X_test = st.session_state[
            "X_test"
        ]

        y_test = st.session_state[
            "y_test"
        ]

        preprocessor = st.session_state[
            "preprocessor"
        ]

        models = st.session_state[
            "models"
        ]

        problem_type = st.session_state[
            "problem_type"
        ]

        best_model_name = (
            st.session_state[
                "results"
            ].iloc[0]["model"]
        )

        model = models[
            best_model_name
        ]

        pipeline = build_model_pipeline(
            preprocessor,
            model,
        )

        with st.spinner(
            "Training final model..."
        ):

            pipeline = train_model(
                pipeline,
                X_train,
                y_train,
            )

        predictions = predict(
            pipeline,
            X_test,
        )

        probabilities = predict_proba(
            pipeline,
            X_test,
        )

        evaluation = evaluate_model(
            problem_type,
            y_test,
            predictions,
            probabilities,
        )

        st.session_state[
            "pipeline"
        ] = pipeline

        st.session_state[
            "evaluation"
        ] = evaluation

        st.success(
            "Final model trained."
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
# Raw Data
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