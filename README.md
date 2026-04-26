# IEEE-CIS Fraud Detection: High-Fidelity XGBoost Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-1.5%2B-orange)
![Pandas](https://img.shields.io/badge/Pandas-Memory_Optimized-green)
![Kaggle](https://img.shields.io/badge/Kaggle-Top_1700-informational)

## 📌 Project Overview
This repository contains the source code and methodology for detecting fraudulent credit card transactions within the highly imbalanced **IEEE-CIS Fraud Detection** dataset. 

Moving beyond standard baseline models, this project tackles high-dimensionality (400+ features) and extreme class imbalance (~3.5% fraud) through advanced feature engineering, memory optimization, and gradient boosting. By engineering unique "Golden User Identifiers" (UIDs) to track temporal behavioral deviations, the final model achieved a **Private Leaderboard AUC of 0.9208** (Rank 1658) on Kaggle.

This project was developed as part of the Advanced Data Management curriculum (CSCI 5000) at the University of Arkansas.

---

## 📁 Repository Structure

* `01_EDA_and_Insights.ipynb`: Exploratory Data Analysis, class imbalance visualization, and temporal trend discovery.
* `02_XGBoost_Modeling.ipynb`: Memory optimization, feature engineering (UID creation), model training, and GroupKFold cross-validation.
* `memory_reducer.py`: Utility script for downcasting Pandas datatypes to prevent session crashes.

*(Note: The raw IEEE-CIS dataset is not included in this repository due to size constraints. It can be downloaded directly from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection).)*

---

## 🚀 Key Engineering Challenges & Solutions

### 1. Memory Constraints & Data Architecture
The raw dataset contains over 590,540 rows and consumes roughly 2.5 GB of RAM, causing out-of-memory errors in standard Colab environments. 
* **Solution:** Implemented a custom datatype downcasting script that converts `float64` to `float32` and `int64` to `int32/int8`, reducing the total memory footprint by over **60%** prior to model training.

### 2. Entity Linking: The "Golden UID"
Because the dataset is anonymized and lacks explicit account numbers, standard models treat every transaction as independent. 
* **Solution:** Engineered a proxy identifier to link transactions back to specific entities. By subtracting the time-delta `D1` feature from the `TransactionDT`, we derived the absolute "Account Start Day."
* **UID Logic:** `UID = card1_addr1_AccountStartDay`
* This allowed for **Group Aggregations** (e.g., `TransactionAmt_uid_mean`), enabling the model to detect when a specific user's transaction deviated from their historical spending pattern.

### 3. Preventing Temporal Leakage
Fraud patterns evolve over time. Randomly splitting the data leads to overfitting because the model learns "future" patterns to predict the past.
* **Solution:** Utilized a **6-fold GroupKFold** cross-validation strategy, grouping by month (`DT_M`). This ensured the model was always validated on an entire month of unseen, future data.

---

## 🧠 Modeling Strategy

Initial baselines were established using Logistic Regression and Random Forest. However, these models struggled with the 434-dimensional space and the masked categorical `V-columns`. 

The final architecture relies on **XGBoost**:
* **Tree Method:** `hist` (Histogram-based optimization for rapid CPU execution).
* **Hyperparameters:** `learning_rate=0.02`, `max_depth=12`, `subsample=0.8`, `colsample_bytree=0.4`.
* **Imbalance Handling:** Evaluated purely on **AUC-ROC** rather than accuracy to accurately reflect True Positive Rates across varying thresholds.

---

## 📊 Final Results

Our engineered features dominated the XGBoost information gain metrics, proving that logical temporal aggregations are significantly more valuable than raw model complexity.

> **Kaggle Public AUC:** 0.9540  
> **Kaggle Private AUC:** 0.9208  
> **Final Leaderboard Rank:** 1658  

---

## ⚙️ How to Run

1.  Clone this repository: `git clone https://github.com/your-username/IEEE-CIS-Fraud-Detection.git`
2.  Download the dataset from Kaggle and place the `.csv` files into a `/data/` directory.
3.  Run the Jupyter Notebooks sequentially. Ensure you have at least 16GB of RAM available or utilize Google Colab.

---

## 👥 Contributors
* **[Your Name]** - Data Engineering, Modeling, & Optimization
* *Team Members (Person 2, Person 3, Person 4)*
