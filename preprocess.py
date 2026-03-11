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

    # Fill documented_monthly_income NaN with stated_monthly_income as proxy
    df["documented_monthly_income"] = df["documented_monthly_income"].fillna(
        df["stated_monthly_income"]
    )

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
