import pandas as pd
import numpy as np

np.random.seed(42)

n = 500

df = pd.DataFrame({
    "age": np.random.randint(18, 65, n),

    "income": np.random.randint(
        20000,
        150000,
        n,
    ),

    "city": np.random.choice(
        [
            "Bangalore",
            "Chennai",
            "Mumbai",
            "Delhi",
        ],
        n,
    ),

    "gender": np.random.choice(
        [
            "Male",
            "Female",
        ],
        n,
    ),

    "education": np.random.choice(
        [
            "School",
            "Bachelor",
            "Master",
            "PhD",
        ],
        n,
    ),
})

# Create a meaningful binary target
df["target"] = (
    (
        (df["income"] > 70000)
        & (df["education"].isin(["Master", "PhD"]))
    )
    |
    (
        (df["age"] > 45)
        & (df["income"] > 50000)
    )
).astype(int)

df.to_csv(
    "data/categorical_classification.csv",
    index=False,
)

print(
    "Created data/categorical_classification.csv"
)

print(
    df.head()
)

print(
    "\nShape:",
    df.shape,
)

print(
    "\nTarget distribution:"
)

print(
    df["target"].value_counts()
)