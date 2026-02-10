## Credit Card Risk Prediction

## Overview
This project predicts **credit card default risk** using machine learning techniques.  
It helps identify high-risk customers based on financial and behavioral data, supporting better credit approval and risk management decisions.

---

## Problem Statement
Credit card defaults cause significant losses for financial institutions.  
Traditional rule-based systems are limited and often inaccurate.

This project applies **data-driven machine learning methods** to classify customers into **low-risk** and **high-risk** categories.

---

## Dataset
**File:** `creditcard.csv`

The dataset contains customer demographic and financial information such as:
- Credit loans
- Monthly income
- No.of Dependents
- Own House or Rent 

**Target Variable:**  
Indicates whether a customer is good customer or not.


---

## Workflow
- Data loading and inspection  
- Missing value handling  
- Encoding categorical features  
- Outlier detection and treatment  
- Feature transformation and filtering  
- Statistical hypothesis testing  
- Model training and evaluation  

---

## Algorithms Used
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Other classification models (implemented in `algos.py`)  

---

## Model Evaluation
The models are evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-score  

---

## Technologies Used
- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib / Seaborn  

---

## 📊 Conclusion: Good vs. Bad Customers

The analysis successfully segments the customer base into two distinct risk profiles:

### ✅ Good Customers (Low Risk)
* **Profile:** Reliable individuals with a high probability of repayment.
* **Key Indicators:** Consistent on-time payments, low credit utilization (typically <30%), and a stable financial history.
* **Recommendation:** Suitable for credit approval, limit increases, and premium financial products.

### ⚠️ Bad Customers (High Risk)
* **Profile:** Individuals exhibiting behaviors linked to potential default.
* **Key Indicators:** Frequent payment delays, high credit utilization (maxing out limits), and irregular repayment patterns.
* **Recommendation:** Requires stricter credit policies, lower credit limits, or rejection of credit applications to mitigate loss.

---

### Comparison Summary

| Feature | Good Customer (Low Risk) | Bad Customer (High Risk) |
| :--- | :--- | :--- |
| **Payment History** | Consistent / On-time | Frequent Delays |
| **Credit Usage** | Controlled | High / Maxed Out |
| **Default Risk** | Low | High |
| **Business Action** | Reward & Retain | Monitor & Restrict 

---

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.8+
* pip (Python package manager)

### Setup
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/credit-card-risk-prediction.git](https://github.com/your-username/credit-card-risk-prediction.git)
   cd credit-card-risk-prediction

----

## Project Developed by :

V Nikhil Kumar

Aspiring AI/ML ENgineer & Data Scientist



