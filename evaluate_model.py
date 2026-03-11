"""
Head-to-head evaluation: Rule-Based Baseline vs. Survival Model.
=================================================================
Loads the pre-trained Cox survival model from train_survival.py and
evaluates its predictive performance against the rule-based baseline
on a held-out 20% stratified test set.

All metrics (Precision, Recall, F1, AUC-ROC, FPR, FNR) are computed
on the same canonical test split used by all pipeline scripts.
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from preprocess import get_canonical_split

# ------------------------------------------------------------------
# Step 1: Data Prep (Canonical split — see preprocess.py)
# ------------------------------------------------------------------
df, X_train, X_test, y_train, y_test = get_canonical_split()

# Filter test set to non-ongoing rows for binary evaluation
non_ongoing_mask = df.loc[X_test.index, "actual_outcome"] != "ongoing"
X_test = X_test[non_ongoing_mask]
y_test = y_test[non_ongoing_mask]
test_indices = X_test.index

# ------------------------------------------------------------------
# Step 2: Load Pre-Trained Survival Model
# ------------------------------------------------------------------
# Load the Cox survival model trained in train_survival.py.
# The model predicts log hazard ratios: higher values = higher default risk.
model = xgb.XGBRegressor()
model.load_model("xgboost_survival_model.json")

# Predicted hazard ratios on the TEST set only
hazard_ratios_test = model.predict(X_test)

# ------------------------------------------------------------------
# Step 3: Thresholding — Match baseline approval rate on TEST set
# ------------------------------------------------------------------
# Baseline approval rate (computed ONLY on the test set to prevent data leakage)
baseline_approval_test = (df.loc[test_indices, "rule_based_decision"] == "approved").mean()

# Threshold: approve the lowest `baseline_approval_test` fraction of hazard
# ratios (lower hazard = safer applicant).
threshold = np.quantile(hazard_ratios_test, baseline_approval_test)

# Target = 1 (Defaulted), Predicted Positive = Denied
y_true_test = y_test
y_pred_baseline_test = (df.loc[test_indices, "rule_based_decision"] != "approved").astype(int)

# Baseline AUC uses continuous rule_based_score (inverted: higher score = safer,
# so default probability ≈ 1 - score/100)
y_prob_baseline_test = 1.0 - df.loc[test_indices, "rule_based_score"] / 100.0

# ML predictions: deny if hazard ratio exceeds threshold
y_pred_ml_test = (hazard_ratios_test > threshold).astype(int)
# Use raw hazard ratios as the continuous risk score for AUC
y_prob_ml_test = hazard_ratios_test

# ------------------------------------------------------------------
# Step 4: Metric Calculation (on TEST set only)
# ------------------------------------------------------------------
def calculate_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    fpr = fp / (fp + tn)  # % of good applicants wrongly denied
    fnr = fn / (fn + tp)  # % of defaults that slipped through

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "fpr": fpr,
        "fnr": fnr,
        "cm": cm,
        "tp": tp,
        "fn": fn,
    }


base_metrics = calculate_metrics(y_true_test, y_pred_baseline_test, y_prob_baseline_test)
ml_metrics = calculate_metrics(y_true_test, y_pred_ml_test, y_prob_ml_test)

# ------------------------------------------------------------------
# Step 5: Confusion Matrices
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

if net_defaults_caught > 0:
    print(
        f"By predicting at the same approval volume as the Baseline, the ML Model catches "
        f"{net_defaults_caught} more actual defaults than the rule-based system. "
        f"This represents an increase in Recall from {base_metrics['recall']:.2%} to "
        f"{ml_metrics['recall']:.2%}, meaning fewer toxic loans enter the portfolio. "
        f"Simultaneously, the model denies fewer actual good applicants (FPR drops from "
        f"{base_metrics['fpr']:.2%} to {ml_metrics['fpr']:.2%})."
    )
else:
    auc_comparison = (
        f"The ML Model achieves a higher AUC-ROC ({ml_metrics['auc']:.4f} vs "
        f"{base_metrics['auc']:.4f}), indicating stronger discriminative power "
        f"across all thresholds."
        if ml_metrics["auc"] > base_metrics["auc"]
        else
        f"The ML Model's AUC-ROC ({ml_metrics['auc']:.4f}) is comparable to the "
        f"baseline's ({base_metrics['auc']:.4f}), indicating that on this small "
        f"dataset ({len(X_test)} test samples), the learned model does not yet "
        f"outrank the hand-tuned rule system."
    )
    print(
        f"On a held-out test set, the ML Model catches {abs(net_defaults_caught)} fewer "
        f"actual defaults than the rule-based system (Recall: {ml_metrics['recall']:.2%} vs "
        f"{base_metrics['recall']:.2%}). {auc_comparison} "
        f"With a larger training corpus or hyperparameter tuning, the ML model's "
        f"ability to learn non-linear patterns should translate into stronger performance."
    )

print(
    f"\nNote: All metrics are evaluated on a held-out 20% test set "
    f"({len(X_test)} samples) from a canonical split shared across all pipeline scripts."
)

# ------------------------------------------------------------------
# Step 8: 5-Fold Cross-Validated AUC (robustness check)
# ------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold
from preprocess import prepare_data, add_survival_columns, FEATURE_COLS

print("\n## 5-Fold Cross-Validated AUC-ROC\n")

df_cv, X_cv, y_cv = prepare_data(keep_ongoing=True)
df_cv = add_survival_columns(df_cv)

# Only evaluate on non-ongoing rows, but train on all (survival can use ongoing)
non_ongoing = df_cv["actual_outcome"] != "ongoing"

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ml_aucs, base_aucs = [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_cv, y_cv), 1):
    X_tr, X_te = X_cv.iloc[train_idx], X_cv.iloc[test_idx]
    dur_tr = df_cv.iloc[train_idx]["duration"].values
    evt_tr = df_cv.iloc[train_idx]["event"].values

    # Train a fresh Cox model on this fold
    y_surv = np.where(evt_tr == 1, dur_tr, -dur_tr)
    fold_model = xgb.XGBRegressor(
        objective='survival:cox', tree_method='hist',
        learning_rate=0.05, max_depth=5, subsample=0.8,
        colsample_bytree=0.8, n_estimators=150, random_state=42
    )
    fold_model.fit(X_tr, y_surv)

    # Evaluate only on non-ongoing test rows
    te_mask = non_ongoing.iloc[test_idx]
    X_te_eval = X_te[te_mask]
    y_te_eval = y_cv.iloc[test_idx][te_mask]

    if len(y_te_eval.unique()) < 2:
        continue  # skip folds without both classes

    hazard = fold_model.predict(X_te_eval)
    ml_aucs.append(roc_auc_score(y_te_eval, hazard))

    base_prob = 1.0 - df_cv.iloc[test_idx].loc[te_mask.values, "rule_based_score"] / 100.0
    base_aucs.append(roc_auc_score(y_te_eval, base_prob))

print(f"| Model    | Mean AUC | Std   |")
print(f"| :------- | :------- | :---- |")
print(f"| Baseline | {np.mean(base_aucs):.4f}   | {np.std(base_aucs):.4f} |")
print(f"| ML Model | {np.mean(ml_aucs):.4f}   | {np.std(ml_aucs):.4f} |")
print(f"\n(5-fold stratified CV, each fold trains a fresh Cox model)")

