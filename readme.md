# 🏦 Customer Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-green.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A machine learning project to predict customer churn for a bank using tree-based models. The goal is to identify customers likely to leave, enabling proactive retention strategies.

![Churn Prediction Banner](https://img.shields.io/badge/ML-Churn%20Prediction-red?style=for-the-badge)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

Customer churn (customer attrition) is a critical problem for banks and financial institutions. Acquiring new customers costs **5-7 times more** than retaining existing ones. This project builds a predictive model to identify at-risk customers before they leave.

### Project Highlights

| Aspect | Details |
|--------|---------|
| **Model** | XGBoost Classifier (Tuned) |
| **Best ROC-AUC** | ~0.85 |
| **Recall (Optimized)** | ~0.75 at threshold 0.30 |
| **Key Predictors** | Age, NumOfProducts, Balance |

---

## 💼 Business Problem

### The Challenge

A bank is experiencing customer churn and wants to:
1. **Identify** customers likely to leave
2. **Understand** the key factors driving churn
3. **Take action** with targeted retention campaigns

### Why This Matters

Cost of Losing a Customer >> Cost of Retention Campaign

False Negative (Missing a churner) = Lost revenue + Acquisition cost
False Positive (Wrong prediction) = Small marketing cost

Cost of Losing a Customer >> Cost of Retention Campaign

False Negative (Missing a churner) = Lost revenue + Acquisition cost
False Positive (Wrong prediction) = Small marketing cost


Not Churned (0): 79.6%
Churned (1): 20.4% ← Imbalanced dataset




---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

Step 2: Create Virtual Environment (Recommended)

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

Step 3: Install Dependencies

pip install -r requirements.txt

Requirements File

# requirements.txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
jupyter>=1.0.0
