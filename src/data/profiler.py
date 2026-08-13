import pandas as pd
from pathlib import Path


def load_dataset(file_path):
    """Load a CSV dataset."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are currently supported.")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Could not read CSV file: {e}")

    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    return df


def profile_dataset(df):
    """Generate a basic profile of the dataset."""

    profile = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_mb": round(
            df.memory_usage(deep=True).sum() / (1024 ** 2), 2
        ),
        "numerical_columns": df.select_dtypes(
            include="number"
        ).columns.tolist(),
        "categorical_columns": df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist(),
        "datetime_columns": df.select_dtypes(
            include=["datetime"]
        ).columns.tolist(),
    }

    profile["missing_percentage"] = round(
        profile["missing_values"]
        / (profile["rows"] * profile["columns"])
        * 100,
        2,
    )

    return profile


def get_column_summary(df):
    """Generate column-level statistics."""

    summary = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing": df.isna().sum().values,
        "missing_%": (
            df.isna().mean().values * 100
        ).round(2),
        "unique_values": df.nunique(dropna=True).values,
    })

    return summary


def detect_problem_type(df, target_column):
    """Detect whether the target represents classification or regression."""

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    target = df[target_column]

    if target.isna().all():
        raise ValueError("Target column contains only missing values.")

    unique_values = target.nunique(dropna=True)

    if pd.api.types.is_numeric_dtype(target):
        if unique_values <= 10:
            return "classification"
        return "regression"

    return "classification"