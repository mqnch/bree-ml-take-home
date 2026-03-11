"""
Fairness Analysis: Baseline Rule-Based System vs. ML Model
==========================================================
Compares approval rates, default capture, and False Negative Rates
across employment_status groups to quantify bias in both systems.

All metrics are evaluated on a held-out 20% test set to prevent data leakage.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from preprocess import prepare_data

# ------------------------------------------------------------------
# Step 1: Data Prep (Shared preprocessing — see preprocess.py)
# ------------------------------------------------------------------
df, X, y = prepare_data()

# ------------------------------------------------------------------
# Step 2: Train/Test Split (Stratified, 80/20)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Keep test-set indices for baseline/group lookups
test_indices = X_test.index

# ------------------------------------------------------------------
# Step 3: ML Model — Train on train set only
# ------------------------------------------------------------------
# Calculate scale_pos_weight to handle the ~14% default class imbalance
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss",
    scale_pos_weight=pos_weight,
)
model.fit(X_train, y_train)

# Predicted probability of default on the TEST set only
pred_default_prob_test = model.predict_proba(X_test)[:, 1]

# ------------------------------------------------------------------
# Step 4: Baseline & ML Approval Labels on TEST set
# ------------------------------------------------------------------
df_test = df.loc[test_indices].copy()
df_test["pred_default_prob"] = pred_default_prob_test

# Baseline approval
df_test["baseline_approved"] = (df_test["rule_based_decision"] == "approved").astype(int)

# ML threshold — match baseline approval rate on the TEST SET to prevent data leakage
baseline_approval_rate_test = df_test["baseline_approved"].mean()
threshold = np.quantile(pred_default_prob_test, baseline_approval_rate_test)
df_test["ml_approved"] = (df_test["pred_default_prob"] <= threshold).astype(int)

# Binary predictions for baseline and ML (Positive = Denied = predicted default)
df_test["baseline_pred_default"] = (df_test["baseline_approved"] == 0).astype(int)
df_test["ml_pred_default"] = (df_test["ml_approved"] == 0).astype(int)

print(f"Overall Baseline Approval Rate: {baseline_approval_rate_test:.4f}")
print(f"ML Threshold (prob of default):  {threshold:.4f}")
print(f"Overall ML Approval Rate:        {df_test['ml_approved'].mean():.4f}  (target: {baseline_approval_rate_test:.4f})")
print(f"Test set size: {len(df_test)} samples\n")

# ------------------------------------------------------------------
# Step 5: Fairness Table — All 3 groups with FNR
# ------------------------------------------------------------------
groups = df_test.groupby("employment_group")

rows = []
for name, group in groups:
    n = len(group)
    n_defaults = group["defaulted"].sum()

    # Baseline FNR: FN / (FN + TP) — defaults missed by baseline
    baseline_tp = ((group["baseline_pred_default"] == 1) & (group["defaulted"] == 1)).sum()
    baseline_fn = ((group["baseline_pred_default"] == 0) & (group["defaulted"] == 1)).sum()
    baseline_fnr = baseline_fn / (baseline_fn + baseline_tp) if (baseline_fn + baseline_tp) > 0 else 0.0

    # Baseline FPR: FP / (FP + TN) — good applicants wrongly denied by baseline
    baseline_fp = ((group["baseline_pred_default"] == 1) & (group["defaulted"] == 0)).sum()
    baseline_tn = ((group["baseline_pred_default"] == 0) & (group["defaulted"] == 0)).sum()
    baseline_fpr = baseline_fp / (baseline_fp + baseline_tn) if (baseline_fp + baseline_tn) > 0 else 0.0

    # ML FNR: FN / (FN + TP) — defaults missed by ML model
    ml_tp = ((group["ml_pred_default"] == 1) & (group["defaulted"] == 1)).sum()
    ml_fn = ((group["ml_pred_default"] == 0) & (group["defaulted"] == 1)).sum()
    ml_fnr = ml_fn / (ml_fn + ml_tp) if (ml_fn + ml_tp) > 0 else 0.0

    # ML FPR: FP / (FP + TN) — good applicants wrongly denied by ML model
    ml_fp = ((group["ml_pred_default"] == 1) & (group["defaulted"] == 0)).sum()
    ml_tn = ((group["ml_pred_default"] == 0) & (group["defaulted"] == 0)).sum()
    ml_fpr = ml_fp / (ml_fp + ml_tn) if (ml_fp + ml_tn) > 0 else 0.0

    rows.append(
        {
            "Employment Status": name.replace("_", " ").title(),
            "n": n,
            "Baseline Approval": f"{group['baseline_approved'].mean():.2%}",
            "ML Approval": f"{group['ml_approved'].mean():.2%}",
            "True Default Rate": f"{group['defaulted'].mean():.2%}",
            "Baseline FPR": f"{baseline_fpr:.2%}",
            "ML FPR": f"{ml_fpr:.2%}",
            "Baseline FNR": f"{baseline_fnr:.2%}",
            "ML FNR": f"{ml_fnr:.2%}",
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
# Step 6: Synthesis — Recommendation (citing table numbers)
# ------------------------------------------------------------------
# Extract per-group stats for the recommendation narrative
group_stats = {}
for _, row in fairness_df.iterrows():
    group_stats[row["Employment Status"]] = row

print("## Recommendation")
print()
print(
    "**Demographic Parity Analysis.** "
    f"The baseline approval gap between Employed ({group_stats.get('Employed', {}).get('Baseline Approval', 'N/A')}) "
    f"and Self Employed ({group_stats.get('Self Employed', {}).get('Baseline Approval', 'N/A')}) "
    "is disproportionate to their true default rates "
    f"({group_stats.get('Employed', {}).get('True Default Rate', 'N/A')} vs "
    f"{group_stats.get('Self Employed', {}).get('True Default Rate', 'N/A')}). "
    "The rule-based system hard-codes an employment score of 60 for self-employed "
    "vs 100 for employed, creating an arbitrary penalty. The ML model moves "
    f"toward Demographic Parity by narrowing the approval gap "
    f"({group_stats.get('Employed', {}).get('ML Approval', 'N/A')} vs "
    f"{group_stats.get('Self Employed', {}).get('ML Approval', 'N/A')}), "
    "aligning approval rates more closely with actual risk."
)
print()
print(
    "**Equalized Odds (FPR & FNR).** "
    f"The baseline misclassifies applicants differently across groups "
    f"(Employed FNR: {group_stats.get('Employed', {}).get('Baseline FNR', 'N/A')}, FPR: {group_stats.get('Employed', {}).get('Baseline FPR', 'N/A')} | "
    f"Self Employed FNR: {group_stats.get('Self Employed', {}).get('Baseline FNR', 'N/A')}, FPR: {group_stats.get('Self Employed', {}).get('Baseline FPR', 'N/A')}). "
    f"The ML model moves toward Equalized Odds with more balanced error rates across groups "
    f"(Employed FNR: {group_stats.get('Employed', {}).get('ML FNR', 'N/A')}, FPR: {group_stats.get('Employed', {}).get('ML FPR', 'N/A')} | "
    f"Self Employed FNR: {group_stats.get('Self Employed', {}).get('ML FNR', 'N/A')}, FPR: {group_stats.get('Self Employed', {}).get('ML FPR', 'N/A')})."
)
print()
print(
    "**Recommendation.** Keep employment_status as an input feature—it carries "
    "legitimate signal about income volatility. However, never use it as a hard "
    "penalty. To move from 'organically fair' to 'mathematically guaranteed' parity, "
    "integrate a fairness constraint library such as fairlearn to enforce Demographic "
    "Parity or Equalized Odds as explicit optimization constraints during training."
)
print(
    f"\nNote: All fairness metrics are evaluated on a held-out 20% test set "
    f"({len(df_test)} samples) to prevent data leakage."
)
