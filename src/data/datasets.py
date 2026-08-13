from sklearn.datasets import (
    load_breast_cancer,
    fetch_california_housing,
)

import pandas as pd


def load_breast_cancer_dataset():
    """Load the Breast Cancer Wisconsin dataset."""

    dataset = load_breast_cancer(
        as_frame=True
    )

    df = dataset.frame.copy()

    df["target"] = dataset.target

    return df


def load_california_housing_dataset():
    """Load the California Housing dataset."""

    dataset = fetch_california_housing(
        as_frame=True
    )

    df = dataset.frame.copy()

    df["target"] = dataset.target

    return df