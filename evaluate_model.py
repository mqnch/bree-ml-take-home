import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from preprocess import prepare_data

# ------------------------------------------------------------------
# Step 1: Data Prep (Shared preprocessing — see preprocess.py)
# ------------------------------------------------------------------
df, X, y = prepare_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Keep test-set indices for baseline comparison
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
# Step 4: Thresholding — Match baseline approval rate on TEST set
# ------------------------------------------------------------------
# Baseline approval rate (computed ONLY on the test set to prevent data leakage)
baseline_approval_test = (df.loc[test_indices, "rule_based_decision"] == "approved").mean()

# Threshold: the quantile of predicted probabilities that gives the same approval rate
threshold = np.quantile(pred_default_prob_test, baseline_approval_test)

# Target = 1 (Defaulted), Predicted Positive = Denied
y_true_test = y_test
y_pred_baseline_test = (df.loc[test_indices, "rule_based_decision"] != "approved").astype(int)

# Baseline AUC uses continuous rule_based_score (inverted: higher score = safer,
# so default probability = 1 - score/100)
y_prob_baseline_test = 1.0 - df.loc[test_indices, "rule_based_score"] / 100.0

y_pred_ml_test = (pred_default_prob_test > threshold).astype(int)
y_prob_ml_test = pred_default_prob_test

# ------------------------------------------------------------------
# Step 5: Metric Calculation (on TEST set only)
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
# Step 6: Confusion Matrices Outputs
# ------------------------------------------------------------------
def print_confusion_matrix(cm, title):
    print(f"--- {title} ---")
    print(f"                Predicted Good (0) | Predicted Default (1)")
    print(f"Actual Good (0) | {cm[0,0]:<16} | {cm[0,1]}")
    print(f"Actual Def (1)  | {cm[1,0]:<16} | {cm[1,1]}\n")


print_confusion_matrix(base_metrics["cm"], "Baseline Confusion Matrix")
print_confusion_matrix(ml_metrics["cm"], "ML Model Confusion Matrix")

# ------------------------------------------------------------------
# Step 7: Summary Table
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
# Step 8: Analysis Output
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
    f"({len(X_test)} samples) to prevent data leakage."
)
