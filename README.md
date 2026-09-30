# Oxygen Saturation Prediction Using Machine Learning

A machine learning project for forecasting a patient's oxygen saturation (SpO₂) 5 minutes into the future using physiological time-series data.

The project explores multiple regression algorithms, temporal feature engineering, patient-level validation, hyperparameter tuning, and model explainability.

> This project is intended for educational and machine-learning research purposes only. It is not intended for clinical diagnosis or medical decision-making.

---

## Project Objective

The goal of this project is to predict:

**SpO₂ 5 minutes into the future**

using historical physiological information such as:

- Current SpO₂
- Previous SpO₂ measurements
- Short-term SpO₂ changes
- Rolling SpO₂ statistics
- Heart rate
- Historical heart-rate measurements

A major goal was to evaluate whether machine-learning models can provide additional predictive value beyond a simple persistence forecast:

> Future SpO₂ = Current SpO₂

---

## Dataset

The project uses physiological time-series data from multiple patients.

The modeling dataset contains data from **50 patients**.

Patients were separated at the patient level to prevent information leakage between training and testing.

The workflow used:

- Training patients for model fitting
- Validation patients for model selection and tuning
- Completely unseen test patients for final evaluation

No individual patient was shared between development and test sets.

---

## Machine Learning Workflow

Raw physiological data  
↓  
Data cleaning and resampling  
↓  
Feature engineering  
↓  
5-minute future SpO₂ target creation  
↓  
Patient-level train / validation / test split  
↓  
Baseline modeling  
↓  
Multiple machine-learning algorithms  
↓  
Patient-aware cross-validation  
↓  
Hyperparameter tuning  
↓  
Final unseen-patient evaluation  
↓  
Error analysis and model explainability

---

## Models Evaluated

The project compared several approaches:

- Persistence Baseline
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- Tuned Gradient Boosting Regressor

---

## Final Test Results

The final locked model was evaluated once on previously unseen test patients.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Persistence Baseline | 0.3746 | 1.2130 | 0.2601 |
| Tuned Gradient Boosting + HR | 0.5375 | **1.1818** | **0.2976** |

The persistence baseline achieved the lowest overall MAE because most 5-minute periods showed little change in oxygen saturation.

The tuned Gradient Boosting model achieved a lower RMSE and higher R², indicating improved performance for some of the larger prediction errors.

---

## Error Analysis

The test data was also analyzed according to the actual magnitude of the 5-minute SpO₂ change.

### Moderate SpO₂ Changes

For changes between 1 and less than 3 percentage points:

- Persistence MAE: **1.1466**
- Gradient Boosting MAE: **1.0058**

### Large SpO₂ Changes

For changes of 3 or more percentage points:

- Persistence MAE: **5.9375**
- Gradient Boosting MAE: **5.1202**
- Persistence RMSE: **7.1681**
- Gradient Boosting RMSE: **6.6332**

This analysis suggests that persistence is extremely effective during stable periods, while temporal SpO₂ and heart-rate features provide additional predictive value when oxygen saturation changes more substantially.

The change groups were used only for retrospective evaluation and were not used as model input features.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Jupyter Notebook
- Joblib
- VS Code
- Git / GitHub

---

## Project Structure

```text
oxygen-saturation-prediction/
│
├── data/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── src/
├── .gitignore
├── requirements.txt
└── README.md
```