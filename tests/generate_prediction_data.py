import pandas as pd


# --------------------------------------------------
# Load training dataset
# --------------------------------------------------

df = pd.read_csv(
    "data/schema_variation.csv"
)


# --------------------------------------------------
# Remove target
# --------------------------------------------------

prediction_df = df.drop(
    columns=["target"]
)


# --------------------------------------------------
# Take unseen samples
# --------------------------------------------------

prediction_df = prediction_df.iloc[
    0:10
].copy()


# --------------------------------------------------
# Save prediction dataset
# --------------------------------------------------

prediction_df.to_csv(
    "data/schema_variation_prediction.csv",
    index=False,
)


print(
    "Created data/schema_variation_prediction.csv"
)

print(
    "\nShape:",
    prediction_df.shape,
)

print(
    "\nColumns:"
)

print(
    list(prediction_df.columns)
)