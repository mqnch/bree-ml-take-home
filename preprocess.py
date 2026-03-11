"""
Shared preprocessing module for loan default prediction.
=========================================================
Single source of truth for feature engineering, used by
evaluate_model.py, fairness_analysis.py, and baseline_and_features.py.
"""

import pandas as pd
import numpy as np


FEATURE_COLS = [
    "stated_monthly_income",
    "documented_monthly_income",
    "loan_amount",
    "bank_ending_balance",
    "bank_has_overdrafts",
    "bank_has_consistent_deposits",
    "monthly_withdrawals",
    "monthly_deposits",
    "num_documents_submitted",
    "dti_ratio",
    "loan_to_income_ratio",
    "balance_to_loan_ratio",
    "missing_docs_flag",
    "income_discrepancy_ratio",
    "employment_status_employed",
    "employment_status_self_employed",
    "employment_status_unemployed",
]


def prepare_data(csv_path="loan_applications.csv", keep_ongoing=False):
    """
    Load and prepare the loan dataset for modeling.

    Parameters
    ----------
    csv_path : str
        Path to the raw loan_applications.csv file.
    keep_ongoing : bool
        If False (default), drop 'ongoing' applications for binary
        classification. If True, retain them (for survival analysis).

    Returns
    -------
    df : pd.DataFrame
        The full preprocessed DataFrame (includes non-feature columns
        like rule_based_decision, employment_group, etc.).
    X : pd.DataFrame
        Feature matrix ready for modeling.
    y : pd.Series
        Binary target (1 = defaulted, 0 = repaid).
    """
    df = pd.read_csv(csv_path)

    # Preserve original employment status for fairness grouping
    df["employment_group"] = df["employment_status"].copy()

    # Filter ongoing if not needed
    if not keep_ongoing:
        df = df[df["actual_outcome"] != "ongoing"].copy()

    # --- Feature Engineering ---

    # DTI ratio
    df["dti_ratio"] = df["monthly_withdrawals"] / df["monthly_deposits"]

    # Loan-to-income ratio: how large is the loan relative to monthly income?
    # Higher values = applicant is stretching further relative to earnings.
    df["loan_to_income_ratio"] = df["loan_amount"] / df["stated_monthly_income"]

    # Balance-to-loan ratio: can the applicant's savings absorb the loan?
    # Higher values = stronger financial cushion.
    df["balance_to_loan_ratio"] = df["bank_ending_balance"] / df["loan_amount"]

    # Missing-docs flag (binary signal — no median imputation)
    df["missing_docs_flag"] = df["documented_monthly_income"].isnull().astype(int)

    # Income discrepancy ratio — keep NaN for missing docs so XGBoost's
    # native sparsity-aware split finding learns the optimal direction at
    # every node.  A sentinel like -1.0 creates a numerical boundary that
    # can sweep honest edge-case ratios (≈0.95) into high-risk leaves.
    df["income_discrepancy_ratio"] = (
        df["stated_monthly_income"] / df["documented_monthly_income"]
    ).replace([np.inf, -np.inf], np.nan)

    # Clamp honest-range ratios to 1.0: the raw ratio for verified applicants
    # is uniform(0.9, 1.05) from the data generation process, which contains
    # no real signal — only noise that XGBoost overfits to (drawing spurious
    # splits at ~0.96).  Only ratios OUTSIDE the honest range (misrepresenters
    # with ratios >> 1.10) retain their discriminative value.
    honest_mask = (
        df["income_discrepancy_ratio"].between(0.85, 1.10, inclusive="both")
    )
    df.loc[honest_mask, "income_discrepancy_ratio"] = 1.0

    # NOTE: documented_monthly_income NaNs are intentionally preserved.
    # XGBoost's sparsity-aware split finding handles missing values natively,
    # and backfilling with stated_monthly_income would create an information
    # leak (the two columns become identical for missing-doc applicants,
    # defeating the missing_docs_flag signal).

    # Binary target
    df["defaulted"] = (df["actual_outcome"] == "defaulted").astype(int)

    # One-hot encode employment_status
    df = pd.get_dummies(df, columns=["employment_status"], drop_first=False)

    # Convert boolean columns to int for XGBoost compatibility
    for col in FEATURE_COLS:
        if col in df.columns and df[col].dtype == bool:
            df[col] = df[col].astype(int)

    X = df[FEATURE_COLS]
    y = df["defaulted"]

    return df, X, y


def get_canonical_split(csv_path="loan_applications.csv", test_size=0.2, random_state=42):
    """
    Single canonical train/test split for the entire pipeline.

    Splits the FULL dataset (including ongoing loans) so that
    train_survival.py, evaluate_model.py, and fairness_analysis.py
    all operate on the same partition — preventing data leakage.

    Returns
    -------
    df : pd.DataFrame
        Full preprocessed DataFrame (all rows including ongoing).
    X_train, X_test : pd.DataFrame
        Feature matrices for train and test sets.
    y_train, y_test : pd.Series
        Binary targets for train and test sets.
    """
    from sklearn.model_selection import train_test_split

    df, X, y = prepare_data(csv_path, keep_ongoing=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return df, X_train, X_test, y_train, y_test


def add_survival_columns(df):
    """
    Add duration and event columns for survival analysis.

    - event: 1 = defaulted, 0 = repaid/ongoing (censored)
    - duration: days_to_default for defaults, 180 for repaid (full term),
      random 30-180 for ongoing (simulated observation window)
    """
    df = df.copy()
    df["event"] = (df["actual_outcome"] == "defaulted").astype(int)

    np.random.seed(42)  # Reproducible censored durations

    def get_duration(row):
        if row["actual_outcome"] == "defaulted":
            return row["days_to_default"]
        elif row["actual_outcome"] == "repaid":
            return 180
        elif row["actual_outcome"] == "ongoing":
            return np.random.randint(30, 181)
        return np.nan

    df["duration"] = df.apply(get_duration, axis=1)
    return df
