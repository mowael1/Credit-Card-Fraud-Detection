# **🦹‍♂️ Credit Card Fraud Detection 💳**  

<!-- ![Fraud Detection](Credit-Card-Fraud-Detection.png) -->
<p align="center">
  <img src="./Credit-Card-Fraud-Detection.png" alt="Image" style="width: 100%;">
</p>


## **1. Project Overview 🧐**

This project focuses on building and evaluateing different classification models to identify fraudulent credit card transactions. The dataset is **highly imbalanced**, making it a perfect scenario for experimenting with different resampling and ensemble techniques.

## **2. Goals 🎯**
  - Handle class imbalance effectively using SMOTE and undersampling techniques.
  - Train and compare multiple models.
  - Perform extensive exploratory data analysis (EDA).
  - Apply feature engineering and scaling techniques.
  - Focus on maximizing recall to reduce false negatives.


## **3. Dataset 📂**

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
- **Size:** The dataset contains 284,807 transactions.  
- **Duration:** The dataset contains transactions made over a period of two days.
- **Features:**
  - 28 anonymized PCA features: V1 to V28.
  - Time and Amount.
  - Target class: 0 (legit), 1 (fraud).

> Note: Only 0.17% of the transactions are fraud, creating a severe class imbalance. This makes fraud detection especially challenging and necessitates the use of techniques such as SMOTE and undersampling to ensure models are trained effectively on minority class instances.

---

## **Models Implemented**
Four machine learning models were trained and evaluated:  

### 1️⃣ **Logistic Regression**
- A simple and interpretable model.
- Tuned using **L2 regularization** and **optimized solver selection**.
- Hyperparameter tuning performed using **Grid Search**.
- Handles class imbalance via **weighted class adjustments**.

### 2️⃣ **Random Forest Classifier**
- An ensemble-based model that uses **bagging**.
- Tuned using:
  - Number of trees (`n_estimators`)
  - Maximum tree depth (`max_depth`)
  - Class weights to handle imbalance.
- **RandomizedSearchCV** was used to explore hyperparameters efficiently.
  
### 3️⃣ **Voting Classifier (Ensemble)**
- Combines **Logistic Regression** and **Random Forest**.
- Uses **soft voting** to balance the strengths of both models.
- Weights were optimized for improved **F1-score**.

### 4️⃣ **XGBoost Classifier**
- A boosting-based model for fraud detection.
- Handles class imbalance well through its built-in scale_pos_weight parameter.
- **RandomizedSearchCV** was used for hyperparameter tuning to optimize performance.
- More efficient and optimized for high performance in imbalanced classification.

---

## **Effect of Resampling Techniques on Performance**
We experimented with different resampling techniques to mitigate class imbalance:

### **1️⃣ No Resampling (Original Data)**
- The models performed well, but the severe imbalance caused **lower recall** for fraud cases.
- XGBoost achieved the best balance between **precision and recall**.
<img src="https://i.imgur.com/4g544LJ.png" alt="Image" style="width: 75%;">
### **2️⃣ SMOTE (Synthetic Minority Over-Sampling Technique)**
- **Logistic Regression improved significantly**, but still underperforms compared to tree-based models.
- **Random Forest and Voting Classifier** showed marginal improvements.
- **XGBoost maintained its strong performance** with better recall and precision.
<img src="https://i.imgur.com/TyUyJSa.png" alt="Image" style="width: 75%;">
### **3️⃣ Random Undersampling**
- **Drastic impact on Logistic Regression**, with poor recall in some cases.
- **Random Forest and Voting Classifier improved** with optimized thresholding.
- **XGBoost suffered a slight decline in PR-AUC but still maintained high recall.**
<img src="https://i.imgur.com/Z3CKFKr.png" alt="Image" style="width: 75%;">

---

## **Performance Metrics**
We compared models using **default and optimized thresholds** across different resampling strategies. Key evaluation metrics:

<img src="https://i.imgur.com/4g544LJ.png" alt="Image" style="width: 100%;">

---

## **Observations & Insights**

---

### 1. **Overall Model Ranking**
- **Tree‐based methods** (Random Forest, XGBoost) tend to outperform Logistic Regression in terms of F1‐score (fraud class) across all sampling strategies.
- **XGBoost** in particular often achieves the **highest F1** when combined with threshold optimization, reflecting a good balance between fraud precision and recall.

---

### 2. **Effect of Threshold Optimization**
- Moving away from the default 0.50 threshold **dramatically changes** the trade‐off between precision and recall, which is critical in fraud detection:
  - In some cases (e.g. undersampling), the default threshold yields extremely high recall but very low precision (or vice versa). Optimizing the cutoff “corrects” this imbalance and significantly boosts F1‐scores.
  - Always consider tuning your decision threshold rather than relying on 0.50, because fraud data are highly imbalanced.

---

### 3. **Undersampling vs. SMOTE vs. No Sampling**

#### **No Sampling**
- Models trained on the original data distribution can **under‐detect fraud** if the class is very rare.  
- You see moderately good F1‐scores with threshold tuning—but in many fraud problems, you might still crave higher recall.

#### **Random Undersampling**
- Can yield **high recall but poor precision** at the default threshold, because the model “sees” more balanced data and is more inclined to predict fraud.
- Once the threshold is optimized, precision goes up substantially, and F1 can become quite high.  
- The downside is that you are throwing away a lot of majority‐class examples, which sometimes hurts generalizability.

#### **SMOTE Oversampling**
- Often the **best balance**—you preserve majority‐class samples *and* synthetically increase minority‐class examples, so the model learns more nuanced decision boundaries.
- Notably, you see better F1‐scores overall, especially when combined with threshold tuning.  
- For example, Random Forest or XGBoost with SMOTE + threshold optimization typically shows strong precision *and* recall, leading to the highest F1 among all sampling methods in many cases.

---

### 4. **Precision–Recall Trade‐Off**
- The “best” model or setting depends on whether you value **catching as many frauds as possible** (high recall) or **ensuring few false alarms** (high precision). 
- F1‐score is a combined measure, but in practice you might prioritize recall (e.g. flagging potential fraud for manual review) or precision (e.g. minimizing unnecessary investigations).  
- Threshold tuning lets you push the operating point toward higher recall or higher precision based on business cost constraints.

---

### 5. **Key Practical Insights**
1. **Always consider threshold tuning** in fraud detection. Default 0.50 often does not reflect optimal operating points.  
2. **SMOTE** (or other oversampling) tends to improve minority‐class metrics compared to not sampling or simple undersampling, especially for tree‐based ensembles.  
3. **Precision vs. Recall** is a policy decision—there is no “one right metric” without considering the real‐world costs of false positives and false negatives.  
4. **Random Forest and XGBoost** generally deliver the strongest performance, but simpler models like Logistic Regression can still be useful for interpretability or as a baseline.

- We typically want a **good recall** to minimize missed frauds, while keeping an acceptable precision to avoid too many false alerts.
- **Ensemble methods + SMOTE + threshold tuning** is often a winning recipe in highly imbalanced scenarios like fraud detection.

## **How to Run the Project**

To train a model, use the command:

```bash
python credit_fraud_train.py --train_data_path path/to/train.csv --val_data_path path/to/val.csv --model xgboost --output_path ./output
```
Supported --model options:
- logistic → Logistic Regression
- random_forest → Random Forest
- voting → Voting Classifier (Logistic + Random Forest)
- xgboost → XGBoost Classifier
