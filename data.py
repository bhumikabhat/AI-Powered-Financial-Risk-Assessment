"""Data loading. Uses a real CSV if available, else generates synthetic data.

Priority: explicit path -> cs-training.csv next to this file / in the current
folder -> synthetic data.

Real dataset (free): Kaggle "Give Me Some Credit" (cs-training.csv).
"""
import os

import numpy as np
import pandas as pd

FEATURES = [
    "revolving_utilization", "age", "late_30_59", "debt_ratio", "monthly_income",
    "open_credit_lines", "late_90_plus", "real_estate_loans", "late_60_89", "dependents",
]
TARGET = "default"
DEFAULT_CSV_NAME = "cs-training.csv"

# Map Kaggle "Give Me Some Credit" column names -> our names
KAGGLE_MAP = {
    "SeriousDlqin2yrs": "default",
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "late_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "late_90_plus",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "late_60_89",
    "NumberOfDependents": "dependents",
}


def default_path() -> str | None:
    """Return the path of cs-training.csv if it sits in the project folder."""
    for folder in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        p = os.path.join(folder, DEFAULT_CSV_NAME)
        if os.path.exists(p):
            return p
    return None


def synthetic(n: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "revolving_utilization": rng.beta(2, 5, n),
        "age": rng.integers(21, 80, n),
        "late_30_59": rng.poisson(0.3, n),
        "debt_ratio": rng.gamma(2, 0.2, n),
        "monthly_income": rng.lognormal(8.6, 0.5, n),
        "open_credit_lines": rng.poisson(8, n),
        "late_90_plus": rng.poisson(0.15, n),
        "real_estate_loans": rng.poisson(1, n),
        "late_60_89": rng.poisson(0.15, n),
        "dependents": rng.poisson(0.8, n),
    })
    logit = (
        -3.2 + 2.5 * df.revolving_utilization + 0.45 * df.late_30_59
        + 0.9 * df.late_90_plus + 0.6 * df.late_60_89 + 0.8 * df.debt_ratio
        - 0.02 * (df.age - 40) - 0.00015 * df.monthly_income + 0.05 * df.dependents
    )
    p = 1 / (1 + np.exp(-logit))
    df[TARGET] = (rng.random(n) < p).astype(int)
    df.attrs["source"] = "synthetic"
    return df


def load(path: str | None = None) -> pd.DataFrame:
    path = path or default_path()
    if not path:
        return synthetic()
    df = pd.read_csv(path)
    df = df.rename(columns=KAGGLE_MAP)
    df = df[FEATURES + [TARGET]].copy()
    # basic cleaning: cap extreme outliers, fill missing
    df["revolving_utilization"] = df["revolving_utilization"].clip(upper=2)
    df["debt_ratio"] = df["debt_ratio"].clip(upper=df["debt_ratio"].quantile(0.99))
    for c in ("monthly_income", "dependents"):
        df[c] = df[c].fillna(df[c].median())
    df = df.dropna()
    df.attrs["source"] = os.path.basename(path)
    return df
