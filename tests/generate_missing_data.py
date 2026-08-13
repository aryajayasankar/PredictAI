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

# Create target BEFORE introducing missing values
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


# --------------------------------------------------
# Introduce missing values
# --------------------------------------------------

# Numerical missing values
df.loc[
    np.random.choice(
        n,
        size=25,
        replace=False,
    ),
    "income",
] = np.nan

df.loc[
    np.random.choice(
        n,
        size=20,
        replace=False,
    ),
    "age",
] = np.nan


# Categorical missing values
df.loc[
    np.random.choice(
        n,
        size=20,
        replace=False,
    ),
    "city",
] = np.nan

df.loc[
    np.random.choice(
        n,
        size=15,
        replace=False,
    ),
    "education",
] = np.nan


df.to_csv(
    "data/missing_values.csv",
    index=False,
)


print(
    "Created data/missing_values.csv"
)

print(
    "\nMissing values:"
)

print(
    df.isna().sum()
)

print(
    "\nShape:",
    df.shape,
)