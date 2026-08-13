import pandas as pd
import numpy as np

np.random.seed(42)

n = 600

df = pd.DataFrame({
    "customer_age": np.random.randint(18, 70, n),

    "annual_spending": np.random.randint(
        5000,
        200000,
        n,
    ),

    "region": np.random.choice(
        [
            "North",
            "South",
            "East",
            "West",
        ],
        n,
    ),

    "membership": np.random.choice(
        [
            "Basic",
            "Premium",
            "VIP",
        ],
        n,
    ),

    "visits_per_month": np.random.randint(
        1,
        25,
        n,
    ),

    "support_tickets": np.random.randint(
        0,
        10,
        n,
    ),
})


# --------------------------------------------------
# Generate target
# --------------------------------------------------

df["target"] = (
    (
        (df["annual_spending"] > 80000)
        & (
            df["membership"].isin(
                ["Premium", "VIP"]
            )
        )
    )
    |
    (
        (df["visits_per_month"] >= 15)
        & (
            df["support_tickets"] <= 3
        )
    )
).astype(int)


df.to_csv(
    "data/schema_variation.csv",
    index=False,
)


print(
    "Created data/schema_variation.csv"
)

print(
    "\nShape:",
    df.shape,
)

print(
    "\nColumns:"
)

print(
    list(df.columns)
)

print(
    "\nTarget distribution:"
)

print(
    df["target"].value_counts()
)