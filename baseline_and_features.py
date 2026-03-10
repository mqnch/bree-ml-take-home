import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def main():
    # Load dataset
    df = pd.read_csv('loan_applications.csv')
    
    # ---------------------------------------------------------
    # Part 1: Baseline Evaluation
    # ---------------------------------------------------------
    print("--- Part 1: Baseline Evaluation ---")
    
    # Filter out 'ongoing' applications for baseline calculation
    df_baseline = df[df['actual_outcome'] != 'ongoing'].copy()
    
    # Create baseline_pred column
    # 'denied' or 'flagged_for_review' = 1 (predicting default), 'approved' = 0
    df_baseline['baseline_pred'] = df_baseline['rule_based_decision'].isin(['denied', 'flagged_for_review']).astype(int)
    
    # Create binary actual_truth column
    # 'defaulted' = 1, 'repaid' = 0
    df_baseline['actual_truth'] = (df_baseline['actual_outcome'] == 'defaulted').astype(int)
    
    y_true = df_baseline['actual_truth']
    y_pred = df_baseline['baseline_pred']
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Precision (Default class): {precision:.4f}")
    print(f"Recall (Default class):    {recall:.4f}")
    print(f"F1-Score (Default class):  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # False Positive Rate and False Negative Rate
    # True Negative (TN): Actual 0, Predict 0 (Repaid, Approved)
    # False Positive (FP): Actual 0, Predict 1 (Repaid, Denied/Flagged) -> good applicants wrongly denied
    # False Negative (FN): Actual 1, Predict 0 (Defaulted, Approved) -> defaults that slipped through
    # True Positive (TP): Actual 1, Predict 1 (Defaulted, Denied/Flagged)
    
    tn, fp, fn, tp = cm.ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f}")
    print()
    
    # ---------------------------------------------------------
    # Part 2: Feature Engineering (For the full dataset)
    # ---------------------------------------------------------
    print("--- Part 2: Feature Engineering ---")
    
    # Create event
    df['event'] = (df['actual_outcome'] == 'defaulted').astype(int)
    
    # Create duration
    np.random.seed(42) # Set seed for reproducibility
    def get_duration(row):
        if row['actual_outcome'] == 'defaulted':
            return row['days_to_default']
        elif row['actual_outcome'] == 'repaid':
            return 180
        elif row['actual_outcome'] == 'ongoing':
            return np.random.randint(30, 181) # random int between 30 and 180 (inclusive)
        return np.nan
        
    df['duration'] = df.apply(get_duration, axis=1)
    
    # Missing docs flag
    df['missing_docs_flag'] = df['documented_monthly_income'].isnull().astype(int)
    
    # Income discrepancy ratio
    df['income_discrepancy_ratio'] = (df['stated_monthly_income'] / df['documented_monthly_income']).fillna(1.0)
    
    # DTI ratio
    # Avoid division by zero by replacing 0s with a small number or np.nan or just let pandas handle it (yields inf)
    # Usually we can add a small epsilon or let it be inf depending on context.
    # The instructions say: `dti_ratio (monthly_withdrawals / monthly_deposits)`
    df['dti_ratio'] = df['monthly_withdrawals'] / df['monthly_deposits']
    # If there are any inf, we might want to cap it, but let's stick to the requirement
    
    # Encoding: One-hot encode employment_status
    df = pd.get_dummies(df, columns=['employment_status'], drop_first=False)
    
    # Clean up: Drop redundant text/leakage columns
    cols_to_drop = [
        'applicant_id',
        'actual_outcome',
        'rule_based_score',
        'rule_based_decision',
        'days_to_default'
    ]
    df_cleaned = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Save as survival_features.csv
    output_file = 'survival_features.csv'
    df_cleaned.to_csv(output_file, index=False)
    print(f"Feature engineering complete. File saved to '{output_file}'.")
    
if __name__ == "__main__":
    main()
