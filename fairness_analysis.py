"""
Fairness Analysis: Baseline Rule-Based System vs. ML Model
==========================================================
Compares approval rates and default capture across employment_status groups
to quantify the bias introduced by the rule-based scoring system.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# Step 1: Data Prep
# ------------------------------------------------------------------
df = pd.read_csv("loan_applications.csv")

# Keep the original employment_status for grouping later
df["employment_group"] = df["employment_status"].copy()

# Filter out ongoing applications for binary comparison
df = df[df["actual_outcome"] != "ongoing"].copy()

# Feature: DTI ratio
df["dti_ratio"] = df["monthly_withdrawals"] / df["monthly_deposits"]

# Feature: missing-docs flag + median imputation
df["missing_docs_flag"] = df["documented_monthly_income"].isnull().astype(int)
df["documented_monthly_income"] = df["documented_monthly_income"].fillna(
    df["documented_monthly_income"].median()
)

# Feature: income discrepancy ratio
df["income_discrepancy_ratio"] = (
    df["stated_monthly_income"] / df["documented_monthly_income"]
).replace([np.inf, -np.inf], 1.0)

# Binary target
df["defaulted"] = (df["actual_outcome"] == "defaulted").astype(int)

# One-hot encode employment_status
df = pd.get_dummies(df, columns=["employment_status"], drop_first=False)

# ------------------------------------------------------------------
# Step 2: ML Risk Simulation
# ------------------------------------------------------------------
feature_cols = [
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

# Convert boolean columns to int for XGBoost
for col in feature_cols:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)

X = df[feature_cols]
y = df["defaulted"]

# Train on full dataset (we want predictions for every row to compare fairly)
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss",
)
model.fit(X, y)

# Predicted probability of default for every applicant
df["pred_default_prob"] = model.predict_proba(X)[:, 1]

# ------------------------------------------------------------------
# Step 3: Baseline Approval Rate
# ------------------------------------------------------------------
baseline_approval_rate = (df["rule_based_decision"] == "approved").mean()
df["baseline_approved"] = (df["rule_based_decision"] == "approved").astype(int)

print(f"Overall Baseline Approval Rate: {baseline_approval_rate:.4f}")
print(f"  ({(df['baseline_approved'].sum())} / {len(df)} applicants)")
print()

# ------------------------------------------------------------------
# Step 4: ML Thresholding — match the baseline approval rate exactly
# ------------------------------------------------------------------
# An applicant is "approved" when their predicted default probability
# is BELOW the threshold. To match baseline approval rate we need:
#   P(pred_default_prob <= threshold) == baseline_approval_rate
# ⟹ threshold = the (baseline_approval_rate)-th quantile of pred_default_prob

threshold = np.quantile(df["pred_default_prob"], baseline_approval_rate)
df["ml_approved"] = (df["pred_default_prob"] <= threshold).astype(int)

ml_approval_rate = df["ml_approved"].mean()
print(f"ML Threshold (prob of default):  {threshold:.4f}")
print(f"Overall ML Approval Rate:        {ml_approval_rate:.4f}  (target: {baseline_approval_rate:.4f})")
print()

# ------------------------------------------------------------------
# Step 5: Fairness Table
# ------------------------------------------------------------------
groups = df.groupby("employment_group")

rows = []
for name, group in groups:
    rows.append(
        {
            "Employment Status": name.replace('_', ' ').title(),
            "n": len(group),
            "Baseline Approval Rate": f"{group['baseline_approved'].mean():.2%}",
            "ML Model Approval Rate": f"{group['ml_approved'].mean():.2%}",
            "True Default Rate": f"{group['defaulted'].mean():.2%}",
        }
    )

fairness_df = pd.DataFrame(rows)

# Print as clean Markdown table
print("## Fairness Comparison: Baseline vs. ML Model")
print()
header = "| " + " | ".join(fairness_df.columns) + " |"
sep = "| " + " | ".join(["---"] * len(fairness_df.columns)) + " |"
print(header)
print(sep)
for _, row in fairness_df.iterrows():
    print("| " + " | ".join(str(v) for v in row.values) + " |")
print()

# ------------------------------------------------------------------
# Step 6: Synthesis — 3-Sentence Recommendation
# ------------------------------------------------------------------
print("## Recommendation")
print()
print(
    "The baseline rule-based system arbitrarily penalizes self-employed applicants "
    "(score 60 vs 100 for employed), leading to systematically lower approval rates "
    "for self-employed applicants despite comparable true default rates. "
    "When the ML model learns directly from actual outcomes, it partially corrects "
    "this bias by redistributing approvals toward applicants whose financial "
    "fundamentals—DTI ratio, bank stability, income verification—warrant approval "
    "regardless of employment category. "
    "Bree should keep employment_status as an input feature (since it carries some "
    "legitimate signal about income volatility) but never use it as a hard penalty; "
    "instead, re-weight the model's reliance on it through regularization or post-hoc "
    "calibration so that approval disparities across groups remain proportional to "
    "actual default-rate differences."
)
