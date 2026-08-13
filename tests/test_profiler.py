from src.data.profiler import (
    load_dataset,
    profile_dataset,
    get_column_summary,
    detect_problem_type,
)


DATA_PATH = "data/sample_customer_data.csv"


def main():
    print("\n=== PredictAI Data Engine Test ===\n")

    df = load_dataset(DATA_PATH)

    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")

    print("\n=== Dataset Profile ===")

    profile = profile_dataset(df)

    for key, value in profile.items():
        print(f"{key}: {value}")

    print("\n=== Column Summary ===")

    summary = get_column_summary(df)
    print(summary.to_string(index=False))

    print("\n=== Problem Type ===")

    problem_type = detect_problem_type(df, "churn")

    print(f"Target: churn")
    print(f"Detected problem type: {problem_type}")

    assert problem_type == "classification"

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()