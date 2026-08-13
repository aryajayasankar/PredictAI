import pandas as pd

from src.preprocessing.pipeline import (
    build_preprocessing_pipeline,
)


DATA_PATH = "data/sample_customer_data.csv"


def main():
    print("\n=== PredictAI Preprocessing Test ===\n")

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["churn"])
    y = df["churn"]

    print(f"Original feature shape: {X.shape}")

    preprocessor = build_preprocessing_pipeline(X)

    X_transformed = preprocessor.fit_transform(X)

    print(
        f"Transformed feature shape: "
        f"{X_transformed.shape}"
    )

    assert X_transformed.shape[0] == X.shape[0]

    print("\nPreprocessing completed successfully.")

    print("\n=== Testing Missing Values ===")

    X_missing = X.copy()

    X_missing.loc[0, "age"] = None
    X_missing.loc[1, "contract_type"] = None

    transformed_missing = (
        preprocessor.fit_transform(X_missing)
    )

    assert not pd.isna(transformed_missing).any()

    print("Missing values handled successfully.")

    print("\n=== Testing Unseen Category ===")

    X_unseen = X.copy()

    X_unseen.loc[0, "contract_type"] = "Two-Year"

    transformed_unseen = (
        preprocessor.transform(X_unseen)
    )

    assert transformed_unseen.shape[0] == X.shape[0]

    print("Unseen categories handled successfully.")

    print("\nAll preprocessing tests passed!")


if __name__ == "__main__":
    main()