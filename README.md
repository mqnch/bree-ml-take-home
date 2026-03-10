# Bree ML Take-Home: Loan Default Prediction

An end-to-end ML pipeline predicting loan default risk, replacing a rigid rule-based system with a predictive survival model.

## Key Features

- **Survival Analysis**: Models default probability over time (`survival:cox`), naturally incorporating censored "ongoing" loans.
- **Data Engineering**: Leverages income missingness as a behavioral signal, flags income misrepresentation, and creates DTI proxies from transaction data.
- **Explainable Decisions**: Integrated SHAP values provide transparent, human-readable justifications for loan constraints.

## Fairness Analysis

The ML model organically corrects the baseline's arbitrary bias against self-employed applicants. By learning objective financial signals from actual outcomes, the model achieves near-Demographic Parity.

| Group             | Baseline Approval | ML Model Approval | True Default Rate |
| :---------------- | :---------------- | :---------------- | :---------------- |
| **Employed**      | 59.56%            | 50.09%            | 27.21%            |
| **Self-Employed** | 35.51%            | 46.82%            | 28.98%            |

## Evaluation against the Baseline

By predicting at the same overall approval volume as the Baseline, the ML Model successfully flags and catches **128 more actual defaults** than the rule-based system (532 caught by the ML model vs. 404 by the baseline). This represents a massive increase in Recall, severely limiting the number of defaults that slip through undetected. At the same time, the ML model is far more accurate overall, as it simultaneously decreases the False Positive Rate (good applicants wrongly denied).

| Metric                         | Baseline | ML Model |
| :----------------------------- | :------- | :------- |
| **Precision**                  | 0.4102   | 0.5401   |
| **Recall**                     | 0.7413   | 0.9761   |
| **F1-Score**                   | 0.5281   | 0.6954   |
| **AUC-ROC**                    | 0.6456   | 0.9652   |
| **FPR** (Good wrongly denied)  | 0.4500   | 0.3509   |
| **FNR** (Defaults slipped via) | 0.2587   | 0.0239   |

### Confusion Matrices

**Baseline Confusion Matrix**

```text
                Predicted Good (0) | Predicted Default (1)
Actual Good (0) | 710              | 581
Actual Def (1)  | 141              | 404
```

**ML Model Confusion Matrix**

```text
                Predicted Good (0) | Predicted Default (1)
Actual Good (0) | 838              | 453
Actual Def (1)  | 13               | 532
```

## Project Structure

- `gen.py`: Dataset generation script.
- `loan_applications.csv`: The original generated dataset.
- `eda.py`: Risk factor exploration and visualization.
- `baseline_and_features.py`: Feature engineering and baseline metrics.
- `train_survival.py`: XGBoost survival modeling and SHAP interpretation.
- `fairness_analysis.py`: Bias audit and rule-based vs. ML outcome comparison.

## How to Run

1. Install dependencies: `pip install -r requirements.txt`.
2. (Optional) Run `gen.py` to see the generation logic; however, use the committed `loan_applications.csv` for reproducibility.
3. Run `eda.py`, `baseline_and_features.py`, `train_survival.py`, and `fairness_analysis.py` in sequence to replicate results.

## Future Work

If I were taking this model toward a production release, my next focus would be on moving from a strong MVP to a resilient system.

First, I’d want to experiment with Random Survival Forests or DeepSurv architectures. While XGBoost is a workhorse, ensembles specifically built for survival data can sometimes catch the subtle, non-linear interactions between income and spending habits that gradient boosting might overlook.

Beyond the architecture, Probability Calibration would be important. When it comes to lending, it isn't enough to just rank who is "riskiest"—we need the model's 20% risk prediction to accurately mean that 20 out of 100 people will actually default. Implementing post-hoc methods like Platt Scaling would ensure our hazard ratios map directly to reliable, real-world probabilities for the finance team.

Finally, to safeguard our Fairness goals as we scale, I’d look into integrating Adversarial Debiasing. This would move us from "organically fair" to "mathematically guaranteed" parity by penalizing the model if it can still "guess" an applicant's protected status from their financial data. This ensures our lending decisions stay objective even as the macro-economic environment shifts.
