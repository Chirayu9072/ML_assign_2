## 📌 Problem Statement
The objective of this project is to implement and evaluate multiple machine learning classification models on the **Adult Census Income Dataset**.  
The task is to predict whether an individual's income exceeds $50K per year based on demographic and employment attributes.

---
- **Dataset:** Adult Census Income (UCI Machine Learning Repository)  
- **Instances:** ~48,842  
- **Features:** 14 attributes (categorical + numerical)  
- **Target Variable:** `income` (binary: `<=50K` or `>50K`)  
- **Preprocessing Steps:**  
  - Missing values (`?`) replaced with `NaN` and dropped  
  - Label encoding applied to categorical features  
  - Train-test split (80/20)  
  - Standard scaling applied to numerical features  

---
The following six models were implemented on the same dataset:
1. Logistic Regression  
2. Decision Tree Classifier  
3. k-Nearest Neighbor Classifier  
4. Naive Bayes Classifier (GaussianNB)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  
---

## 📊 Comparison Table of Evaluation Metrics
| ML Model Name       | Accuracy | AUC     | Precision | Recall  | F1      | MCC     |
|---------------------|----------|---------|-----------|---------|---------|---------|
| Logistic Regression | 0.8247   | 0.8568  | 0.7113    | 0.4596  | 0.5584  | 0.4722  |
| Decision Tree       | 0.8128   | 0.7527  | 0.6068    | 0.6365  | 0.6213  | 0.4974  
| kNN                 | 0.8353   | 0.8572  | 0.6786    | 0.6022  | 0.6381  | 0.5335  |
| Naive Bayes         | 0.8093   | 0.8613  | 0.7064    | 0.3584  | 0.4755  | 0.4060  |
| Random Forest       | 0.8609   | 0.9110  | 0.7487    | 0.6372  | 0.6884  | 0.6029  |
| XGBoost             | 0.8719   | 0.9270  | 0.7661    | 0.6754  | 0.7179  | 0.6376  |

---


| ML Model Name       | Observation about model performance |
|---------------------|--------------------------------------|
| Logistic Regression | Performs well on linear relationships; interpretable but less powerful on complex data. |
| Decision Tree       | Easy to interpret; prone to overfitting; moderate accuracy. |
| kNN                 | Sensitive to scaling; slower with large datasets; moderate performance. |
| Naive Bayes         | Fast and simple; assumes feature independence; decent baseline. |
| Random Forest       | Strong performance; handles categorical + numerical features well; robust against overfitting. |
| XGBoost             | Best overall performance; powerful ensemble method; slightly higher computational cost. |

---

