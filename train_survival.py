"""
Train XGBoost Survival (Cox) model and generate SHAP explanations.
===================================================================
Uses the canonical split from preprocess.py to prevent data leakage.
Saves the trained model to xgboost_survival_model.json for use by
evaluate_model.py and fairness_analysis.py.
"""

import numpy as np
import xgboost as xgb
from lifelines.utils import concordance_index
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

from preprocess import get_canonical_split, add_survival_columns

# --- Premium Aesthetic Settings ---
sns.set_theme(
    style="whitegrid",
    palette="crest",
    rc={
        'axes.grid': True,
        'grid.alpha': 0.2,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
    }
)

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': (10, 6),
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def main():
    # ------------------------------------------------------------------
    # Step 1: Data Setup (canonical split — shared with all scripts)
    # ------------------------------------------------------------------
    print("Loading data with canonical split...")
    df, X_train, X_test, y_train, y_test = get_canonical_split()

    # Add survival-specific columns (duration, event)
    df = add_survival_columns(df)

    # Extract survival targets aligned to train/test indices
    dur_train = df.loc[X_train.index, "duration"].values
    evt_train = df.loc[X_train.index, "event"].values
    dur_test = df.loc[X_test.index, "duration"].values
    evt_test = df.loc[X_test.index, "event"].values

    # XGBoost Cox expects y = duration, negative for censored observations
    y_surv_train = np.where(evt_train == 1, dur_train, -dur_train)

    # ------------------------------------------------------------------
    # Step 2: Feature Selection / Hyperparameter Tuning
    # ------------------------------------------------------------------
    from sklearn.model_selection import train_test_split
    
    print("Tuning hyperparameters...")
    # Create a small validation set from the training data for tuning
    X_tr_tune, X_val_tune, y_tr_tune, y_val_tune, dur_tr_tune, dur_val_tune, evt_tr_tune, evt_val_tune = train_test_split(
        X_train, y_surv_train, dur_train, evt_train, test_size=0.2, random_state=42
    )

    best_c_index = -1
    best_params = {}

    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 150, 200]
    }

    for md in param_grid['max_depth']:
        for lr in param_grid['learning_rate']:
            for ne in param_grid['n_estimators']:
                tune_model = xgb.XGBRegressor(
                    objective='survival:cox',
                    tree_method='hist',
                    eval_metric='cox-nloglik',
                    learning_rate=lr,
                    max_depth=md,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    n_estimators=ne,
                    random_state=42
                )
                tune_model.fit(X_tr_tune, y_tr_tune)
                val_preds = tune_model.predict(X_val_tune)
                val_c_index = concordance_index(dur_val_tune, -val_preds, evt_val_tune)
                
                if val_c_index > best_c_index:
                    best_c_index = val_c_index
                    best_params = {'max_depth': md, 'learning_rate': lr, 'n_estimators': ne}

    print(f"Best params found: {best_params} (Val C-index: {best_c_index:.4f})")

    print("Training final XGBoost Survival (Cox) model on full train set...")
    xgb_model = xgb.XGBRegressor(
        objective='survival:cox',
        tree_method='hist',
        eval_metric='cox-nloglik',
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=best_params['n_estimators'],
        random_state=42
    )

    xgb_model.fit(X_train, y_surv_train)

    # ------------------------------------------------------------------
    # Step 3: Evaluation (on held-out test set only)
    # ------------------------------------------------------------------
    print("Evaluating model on held-out test set...")
    preds_test = xgb_model.predict(X_test)

    # C-index: pass -preds because higher hazard = shorter survival
    c_index = concordance_index(dur_test, -preds_test, evt_test)
    print(f"Concordance Index (C-index) on test set: {c_index:.4f}")

    # ------------------------------------------------------------------
    # Step 4: Interpretability (SHAP on test set)
    # ------------------------------------------------------------------
    print("Generating SHAP Explanations on unseen test set...")
    os.makedirs('graphs', exist_ok=True)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test)

    # A. SHAP Summary Plot (Beeswarm)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.suptitle("SHAP Summary Plot: Global Feature Importance & Impact (Test Set)", y=0.98)
    sns.despine()
    plt.savefig('graphs/shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved SHAP Summary Plot to 'graphs/shap_summary_plot.png'")

    # B. SHAP Waterfall plot for a specific high-risk applicant
    default_indices_test = np.where(evt_test == 1)[0]
    highest_risk_idx_test = default_indices_test[np.argmax(preds_test[default_indices_test])]

    plt.figure()
    shap.waterfall_plot(shap_values[highest_risk_idx_test], show=False)
    plt.suptitle(f"SHAP Waterfall Plot for High-Risk Applicant (Test Index {highest_risk_idx_test})", y=0.98)
    sns.despine()
    plt.savefig('graphs/shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP Waterfall Plot to 'graphs/shap_waterfall_plot.png' for test applicant {highest_risk_idx_test}")

    # ------------------------------------------------------------------
    # Step 4b: SHAP Interpretation (human-readable summary)
    # ------------------------------------------------------------------
    # Global feature importance: mean |SHAP| across test set
    print("\n--- SHAP Interpretation: Global Feature Importance ---")
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_names = X_test.columns.tolist()
    ranked = sorted(zip(feature_names, mean_abs_shap), key=lambda x: -x[1])

    print("Top features driving default risk (mean |SHAP value| on test set):\n")
    for i, (feat, val) in enumerate(ranked[:5], 1):
        print(f"  {i}. {feat:35s}  mean|SHAP| = {val:.4f}")

    print(
        "\nInterpretation: Features with higher mean |SHAP| have a larger average\n"
        "impact on the model's hazard prediction. A loan reviewer should focus on\n"
        "these signals when auditing a decision.\n"
    )

    # Individual applicant explanation
    print(f"--- SHAP Interpretation: High-Risk Applicant (Test Index {highest_risk_idx_test}) ---")
    applicant_shap = shap_values[highest_risk_idx_test].values
    applicant_features = X_test.iloc[highest_risk_idx_test]
    ranked_individual = sorted(
        zip(feature_names, applicant_shap, applicant_features),
        key=lambda x: -abs(x[1])
    )

    print("Top factors driving THIS applicant's elevated risk:\n")
    for i, (feat, shap_val, feat_val) in enumerate(ranked_individual[:5], 1):
        direction = "↑ increases risk" if shap_val > 0 else "↓ decreases risk"
        print(f"  {i}. {feat:35s}  value={feat_val:<10.2f}  SHAP={shap_val:+.4f}  ({direction})")

    print(
        "\nA loan reviewer would see: this applicant was flagged primarily because of\n"
        f"their {ranked_individual[0][0]} ({ranked_individual[0][2]:.2f}), "
        f"combined with {ranked_individual[1][0]} ({ranked_individual[1][2]:.2f}).\n"
    )

    # ------------------------------------------------------------------
    # Step 5: Save the model
    # ------------------------------------------------------------------
    model_path = 'xgboost_survival_model.json'
    xgb_model.save_model(model_path)
    print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    main()
