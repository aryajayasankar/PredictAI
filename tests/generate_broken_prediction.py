import pandas as pd


df = pd.read_csv(
    "data/schema_variation_prediction.csv"
)


# Remove a required feature
df = df.drop(
    columns=["customer_age"]
)


df.to_csv(
    "data/broken_prediction.csv",
    index=False,
)


print(
    "Created data/broken_prediction.csv"
)

print(
    "\nColumns:"
)

print(
    list(df.columns)
)