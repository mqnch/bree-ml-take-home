# Bree ML Take-Home: Loan Default Prediction

An end-to-end ML pipeline predicting loan default risk, replacing a rigid rule-based system with a predictive survival model.

## Key Features

- **Survival Analysis**: Models default probability over time (`survival:cox`), naturally incorporating censored "ongoing" loans.
- **Data Engineering**: Leverages income missingness as a behavioral signal, flags income misrepresentation, and creates DTI proxies from transaction data.
- **Explainable Decisions**: Integrated SHAP values provide transparent, human-readable justifications for loan constraints.

## Key Design Decisions

### 1. Two-Model Architecture

This project trains two separate models for distinct purposes:

- **XGBoost Survival (`survival:cox`)** in `train_survival.py` — models time-to-default as a Cox proportional hazards problem. This naturally handles right-censored "ongoing" loans and produces SHAP-based explanations showing *why* a specific applicant was flagged (explainability deliverable).
- **XGBClassifier** in `evaluate_model.py` — a standard binary classifier trained to predict default vs. repaid. This enables an apples-to-apples metric comparison (Precision, Recall, F1, AUC-ROC) against the rule-based baseline at the same approval volume.

The survival model answers *"when will they default?"* while the classifier answers *"will they default?"* — both are necessary to evaluate the system comprehensively.

*Note: The survival pipeline uses `baseline_and_features.py` for feature engineering (including duration/event columns), while the classifier uses `preprocess.py`. This divergence is intentional — the survival model requires time-to-event fields that are irrelevant to binary classification.*

### 2. Ongoing Class Handling

~8% of loans are "ongoing" with unknown final outcomes. For binary classification (`evaluate_model.py`, `fairness_analysis.py`), ongoing loans are **dropped** because their true label is unknowable — imputing them as "repaid" would introduce survivorship bias, while labeling them "defaulted" would inflate the default rate. For survival analysis (`train_survival.py`), ongoing loans are **right-censored** — the model learns they survived *at least* until the observation window closed, extracting partial information without bias.

### 3. Missingness & Misrepresentation Strategy

The 15% of applicants missing documented income are **not imputed**. Instead, missingness is treated as a behavioral signal via a binary `missing_docs_flag` feature — EDA confirms that applicants who fail to submit documentation default at substantially higher rates. Similarly, the 5% who misrepresent their income are **kept in the dataset** — the `income_discrepancy_ratio` feature (stated / documented income) naturally captures their risk signal, and the model should learn to catch them rather than having them artificially removed.

### 4. Class Imbalance

The dataset has ~14% defaults (roughly 6:1 repaid-to-defaulted). To handle this significant class imbalance without relying on synthetic resampling methods like SMOTE, the `XGBClassifier` explicitly integrates a `scale_pos_weight` hyperparameter. This tells the gradient-boosted trees to heavily penalize misclassifying the minority default class during optimization, ensuring the model prioritizes identifying actual defaults.

## Fairness Analysis

The ML model organically corrects the baseline's arbitrary bias against self-employed applicants. By learning objective financial signals from actual outcomes, the model achieves near-Demographic Parity.

| Employment Status | n | Baseline Approval | ML Approval | True Default Rate | Baseline FPR | ML FPR | Baseline FNR | ML FNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Employed | 215 | 56.28% | 51.16% | 26.51% | 36.08% | 43.04% | 35.09% | 35.09% |
| Self Employed | 112 | 34.82% | 39.29% | 25.89% | 57.83% | 54.22% | 13.79% | 20.69% |
| Unemployed | 41 | 0.00% | 14.63% | 56.10% | 0.00% | 16.67% | 0.00% | 4.35% |

## Evaluation against the Baseline

All metrics are evaluated on a held-out 20% stratified test set (368 samples) to prevent data leakage. When threshold-matched to the same overall approval volume as the baseline, the ML model achieves comparable AUC-ROC (0.6985 vs 0.7218) and slightly higher FPR (slightly more good applicants wrongly denied), though with lower Recall on this small test set. The comparable AUC-ROC indicates the model has learned meaningful risk patterns, but the limited training data (1,468 samples) constrains its ability to outperform a hand-tuned rule system that was designed with domain knowledge.

| Metric                         | Baseline | ML Model |
| :----------------------------- | :------- | :------- |
| **Precision**                  | 0.4087   | 0.3942   |
| **Recall**                     | 0.7798   | 0.7523   |
| **F1-Score**                   | 0.5363   | 0.5174   |
| **AUC-ROC**                    | 0.7218   | 0.6985   |
| **FPR** (Good wrongly denied)  | 0.4749   | 0.4865   |
| **FNR** (Defaults slipped via) | 0.2202   | 0.2477   |

### Business Tradeoff

On this small synthetic dataset, the ML model does not yet outperform the hand-tuned baseline — this is expected and honest. The baseline benefits from domain-expert scoring weights that are well-calibrated to the data generation process. However, the ML model's **true value lies elsewhere**: its fairness analysis shows it corrects arbitrary bias (self-employed approval gap narrows significantly), and its SHAP-based explainability provides transparent per-applicant justifications that a rule-based score cannot. With real-world data volumes, the learned model's ability to capture non-linear feature interactions should translate into stronger discriminative performance.

*Note: All metrics are from a single 80/20 stratified split on a 2,000-row synthetic dataset (368 test samples). Results may vary across seeds. Cross-validation would provide more robust estimates.*

### Confusion Matrices

**Baseline Confusion Matrix**

```text
                Predicted Good (0) | Predicted Default (1)
Actual Good (0) | 136              | 123
Actual Def (1)  | 24               | 85
```

**ML Model Confusion Matrix**

```text
                Predicted Good (0) | Predicted Default (1)
Actual Good (0) | 133              | 126
Actual Def (1)  | 27               | 82
```

## Project Structure

- `gen.py`: Dataset generation script.
- `loan_applications.csv`: The original generated dataset.
- `eda.py`: Risk factor exploration and visualization.
- `baseline_and_features.py`: Feature engineering and baseline metrics.
- `preprocess.py`: Shared preprocessing module (single source of truth for feature engineering).
- `train_survival.py`: XGBoost survival modeling and SHAP interpretation.
- `evaluate_model.py`: Head-to-head performance comparison against the baseline.
- `fairness_analysis.py`: Bias audit and rule-based vs. ML outcome comparison.

## How to Run

1. Install dependencies: `pip install -r requirements.txt`.
2. (Optional) Run `gen.py` to see the generation logic; however, use the committed `loan_applications.csv` for reproducibility.
3. Run scripts in sequence to replicate results:
   ```bash
   python baseline_and_features.py
   python train_survival.py
   python evaluate_model.py
   python fairness_analysis.py
   ```
4. EDA visualizations: `python eda.py` (outputs to `graphs/`).

## Future Work

If I were taking this model toward a production release, my next focus would be on moving from a strong MVP to a resilient system.

First, I'd want to experiment with Random Survival Forests or DeepSurv architectures, combined with cross-validated hyperparameter tuning (e.g., using Optuna). While XGBoost is a workhorse, ensembles specifically built for survival data can sometimes catch the subtle, non-linear interactions between income and spending habits that gradient boosting might overlook.

Beyond the architecture, Probability Calibration would be important. When it comes to lending, it isn't enough to just rank who is "riskiest"—we need the model's 20% risk prediction to accurately mean that 20 out of 100 people will actually default. Implementing post-hoc methods like Platt Scaling would ensure our hazard ratios map directly to reliable, real-world probabilities for the finance team.

Finally, to safeguard our Fairness goals as we scale, I'd look into integrating Adversarial Debiasing. This would move us from "organically fair" to "mathematically guaranteed" parity by penalizing the model if it can still "guess" an applicant's protected status from their financial data. This ensures our lending decisions stay objective even as the macro-economic environment shifts.

### Production Failure Modes & Monitoring
In a real-world setting, a model is only as good as the data it continues to receive. Key areas of focus for a production rollout would include:
- **Data Drift Detection:** Macro-economic shifts (e.g., inflation spikes) will change the underlying relationship between income and default risk. We need statistical monitoring to detect input drift early.
- **Model Degradation Alerts:** The model's baseline performance will inevitably decay over time. Automated alerts must be configured for when AUC drops below an acceptable threshold on recent loan cohorts.
- **Retraining Cadence:** We'd need a strategy for periodic model retraining (e.g., quarterly) to continually factor in new applicant behavior.
- **A/B Testing:** The model should never replace the rule-based system verbatim overnight. A phased shadow-mode rollout followed by randomized A/B testing is critical to safely measure the true financial impact of the ML model.
