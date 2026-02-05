# Customer Churn Prediction

Predicting bank customer churn using machine learning.

## Overview

This project builds a predictive model to identify customers likely to leave the bank, enabling proactive retention strategies.

| Item | Details |
|------|---------|
| Model | XGBoost Classifier |
| ROC-AUC | ~0.85 |
| Dataset | 10,000 customers |

## Dataset

**File:** `data/Churn_Modelling.csv`

**Key Features:**
- CreditScore, Age, Tenure, Balance
- Geography, Gender, NumOfProducts
- IsActiveMember, HasCrCard

**Target:** `Exited` (1 = Churned, 0 = Stayed)

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

Usage
```
jupyter notebook main.ipynb
```


Methodology
```



Data Split - 80/20 train-test (stratified)
Preprocessing - Encoding, scaling, feature engineering
Models - Decision Tree, Random Forest, XGBoost
Tuning - RandomizedSearchCV with 5-fold CV
Evaluation - Threshold optimization for recall
Results
Metric	Value
Accuracy	0.815
Precision	0.452
Recall	0.753
F1-Score	0.565
ROC-AUC	0.852
Key Findings
Top Predictors:

Age - Older customers churn more
NumOfProducts - 3+ products = high risk
Balance - High balance + inactive = risk
Geography - German customers churn more

```




```
Project Structure


├── data/
│   └── Churn_Modelling.csv
├── main.ipynb
└── readme.md
```

```
Author

Ankit Luhar
```

```
License

MIT License
```
