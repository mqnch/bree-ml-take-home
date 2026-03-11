"""
Baseline evaluation & survival feature generation.
====================================================
Part 1: Quick sanity-check metrics for the rule-based baseline (full dataset).
Part 2: Generates survival_features.csv by adding duration/event columns
        on top of the shared feature engineering in preprocess.py.
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from preprocess import prepare_data, add_survival_columns, FEATURE_COLS


def main():
    # ---------------------------------------------------------
    # Part 1: Baseline Evaluation
    # Note: This runs on the full non-ongoing dataset as an
    # initial sanity check. See evaluate_model.py for the
    # held-out test set evaluation against the survival model.
    # ---------------------------------------------------------
    print("--- Part 1: Baseline Evaluation ---")

    df_baseline, _, _ = prepare_data()

    # 'denied' or 'flagged_for_review' = 1 (predicting default), 'approved' = 0
    df_baseline["baseline_pred"] = df_baseline["rule_based_decision"].isin(
        ["denied", "flagged_for_review"]
    ).astype(int)

    y_true = df_baseline["defaulted"]
    y_pred = df_baseline["baseline_pred"]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Precision (Default class): {precision:.4f}")
    print(f"Recall (Default class):    {recall:.4f}")
    print(f"F1-Score (Default class):  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f}")
    print()

    # ---------------------------------------------------------
    # Part 2: Feature Engineering for Survival Analysis
    # Uses shared preprocess.py for all feature engineering,
    # then adds survival-specific columns (duration, event)
    # via the shared add_survival_columns() helper.
    # ---------------------------------------------------------
    print("--- Part 2: Feature Engineering (Survival) ---")

    df_surv, X_surv, y_surv = prepare_data(keep_ongoing=True)

    # Add duration and event columns (shared logic from preprocess.py)
    df_surv = add_survival_columns(df_surv)

    # Build survival features: FEATURE_COLS + duration + event
    survival_df = X_surv.copy()
    survival_df["duration"] = df_surv["duration"].values
    survival_df["event"] = df_surv["event"].values

    output_file = "survival_features.csv"
    survival_df.to_csv(output_file, index=False)
    print(f"Feature engineering complete. File saved to '{output_file}'.")


if __name__ == "__main__":
    main()

