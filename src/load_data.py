"""
Step 1: Load and understand the German Credit dataset.

The raw file has NO headers and uses coded values (A11, A12, etc.)
This script maps those codes to readable column names and values,
then does a first look at the data.
"""

import pandas as pd

# The original UCI file is space-separated with no header row.
# These are the 20 feature columns + 1 target column, in the exact
# order they appear in the file (this order is documented by UCI).
COLUMN_NAMES = [
    "checking_account_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment_since",
    "installment_rate_pct",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits_count",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
    "target",  # 1 = good credit risk, 2 = bad credit risk (this is what we predict)
]


def load_raw_data(filepath: str = "data/german_credit.csv") -> pd.DataFrame:
    """
    Load the raw German Credit dataset.

    The file is space-separated (not comma-separated, despite the .csv name
    we gave it) so we tell pandas to split on whitespace with sep="\\s+".
    """
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return df


def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    The original dataset encodes the target as 1 (good) / 2 (bad).
    We convert this to 0 (good, no default risk) / 1 (bad, default risk)
    because most ML libraries expect the "positive class" (the thing
    we're trying to detect) to be labeled 1.
    """
    df = df.copy()
    df["target"] = df["target"].map({1: 0, 2: 1})
    return df


def explore_data(df: pd.DataFrame) -> None:
    """
    First look at the data. This is a step a lot of beginners skip —
    but understanding your data BEFORE modeling is what separates
    someone who understands ML from someone who just calls .fit().
    """
    print("=" * 60)
    print("SHAPE:", df.shape, "→", df.shape[0], "rows,", df.shape[1], "columns")

    print("\n" + "=" * 60)
    print("FIRST 5 ROWS:")
    print(df.head())

    print("\n" + "=" * 60)
    print("MISSING VALUES PER COLUMN:")
    print(df.isnull().sum())

    print("\n" + "=" * 60)
    print("TARGET DISTRIBUTION (this is the class imbalance check):")
    print(df["target"].value_counts())
    print(df["target"].value_counts(normalize=True).round(3) * 100, "% of total")

    print("\n" + "=" * 60)
    print("DATA TYPES:")
    print(df.dtypes)


if __name__ == "__main__":
    df = load_raw_data()
    df = clean_target(df)
    explore_data(df)