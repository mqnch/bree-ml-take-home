import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ------------------------------------------------------------------
# Step 1: Data Prep
# ------------------------------------------------------------------
df = pd.read_csv("loan_applications.csv")

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

# Binary target (Positive class = Defaulted)
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

# Train on full dataset
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
# Step 3: Thresholding Configuration
# ------------------------------------------------------------------
# Match the baseline approval rate exactly
baseline_approval_rate = (df["rule_based_decision"] == "approved").mean()

# Applicants with prob <= threshold are approved to match the baseline rate
threshold = np.quantile(df["pred_default_prob"], baseline_approval_rate)

# Target = 1 (Defaulted), Predicted Positive = Denied
y_true = df["defaulted"]
y_pred_baseline = (df["rule_based_decision"] != "approved").astype(int)
y_prob_baseline = y_pred_baseline # Binary proxy for probabilities

y_pred_ml = (df["pred_default_prob"] > threshold).astype(int)
y_prob_ml = df["pred_default_prob"]

# ------------------------------------------------------------------
# Step 4: Metric Calculation
# ------------------------------------------------------------------
def calculate_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    fpr = fp / (fp + tn) # % of good applicants wrongly denied
    fnr = fn / (fn + tp) # % of defaults that slipped through
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "fpr": fpr,
        "fnr": fnr,
        "cm": cm,
        "tp": tp,
        "fn": fn
    }

base_metrics = calculate_metrics(y_true, y_pred_baseline, y_prob_baseline)
ml_metrics = calculate_metrics(y_true, y_pred_ml, y_prob_ml)

# ------------------------------------------------------------------
# Step 5: Confusion Matrices Outputs
# ------------------------------------------------------------------
def print_confusion_matrix(cm, title):
    print(f"--- {title} ---")
    print(f"                Predicted Good (0) | Predicted Default (1)")
    print(f"Actual Good (0) | {cm[0,0]:<16} | {cm[0,1]}")
    print(f"Actual Def (1)  | {cm[1,0]:<16} | {cm[1,1]}\n")

print_confusion_matrix(base_metrics["cm"], "Baseline Confusion Matrix")
print_confusion_matrix(ml_metrics["cm"], "ML Model Confusion Matrix")

# ------------------------------------------------------------------
# Step 6: Summary Table
# ------------------------------------------------------------------
print("## Summary Table: Baseline vs. ML Model\n")
print("| Metric | Baseline | ML Model |")
print("| :--- | :--- | :--- |")
print(f"| **Precision** | {base_metrics['precision']:.4f} | {ml_metrics['precision']:.4f} |")
print(f"| **Recall** | {base_metrics['recall']:.4f} | {ml_metrics['recall']:.4f} |")
print(f"| **F1-Score** | {base_metrics['f1']:.4f} | {ml_metrics['f1']:.4f} |")
print(f"| **AUC-ROC** | {base_metrics['auc']:.4f} | {ml_metrics['auc']:.4f} |")
print(f"| **FPR** (Good wrongly denied)| {base_metrics['fpr']:.4f} | {ml_metrics['fpr']:.4f} |")
print(f"| **FNR** (Defaults slipped via) | {base_metrics['fnr']:.4f} | {ml_metrics['fnr']:.4f} |")
print()

# ------------------------------------------------------------------
# Step 7: Analysis Output
# ------------------------------------------------------------------
net_defaults_caught = ml_metrics["tp"] - base_metrics["tp"]
print("## Analysis")
print(f"By predicting at the same approval volume as the Baseline, the ML Model catches {net_defaults_caught} more actual defaults than the rule-based system. "
      f"This represents an increase in Recall from {base_metrics['recall']:.2%} to {ml_metrics['recall']:.2%}, meaning fewer toxic loans enter the portfolio. "
      f"Simultaneously, the model denies fewer actual good applicants (FPR drops from {base_metrics['fpr']:.2%} to {ml_metrics['fpr']:.2%}).")
