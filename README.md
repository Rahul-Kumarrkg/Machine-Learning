# 💳 Credit Card Fraud Detection Using Machine Learning.

# 🎯 1. Goal of Credit Card Fraud Detection Using Machine Learning
The main goal of this project is:
To accurately identify fraudulent credit card transactions using machine learning techniques while minimizing false positives and false negatives, even in the presence of highly imbalanced data.

# 🔍 2. Introduction
Credit card fraud is a major problem for financial institutions and cardholders. Due to the increasing volume of transactions and the subtlety of fraud patterns, traditional rule-based systems often fail. **Machine Learning (ML)** offers the capability to detect complex patterns and adapt over time.


# ⚠️ 3. Problem Statement
Detect whether a given transaction is fraudulent or not.  
The main challenges:
- **Highly imbalanced dataset** – less than 0.2% of transactions are fraudulent.
- **Real-time detection** – models should be fast and scalable.
- **Minimizing false negatives** – missing a fraud is more costly than a false alert.


# ✅ 4. Specific Objectives:
- **Detect fraud in real-time** to prevent financial loss.
- **Improve detection accuracy** by learning complex patterns from data.
- **Handle class imbalance** using techniques like SMOTE, under/over-sampling.
- **Minimize false negatives** (frauds not detected).
- **Maintain a low false positive rate** to avoid blocking genuine users.
- **Develop a model that is scalable and deployable** for real-world use.


# 📊 5. Dataset Details
- **Source:** Public dataset from Kaggle Credit Card Fraud Detection.
- **Dataset Link:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Records:** 284,807 transactions.
- **Frauds:** 492 fraud cases (0.172% of total).
-  **Features:**
    - **V1 to V28:** Result of PCA transformation to preserve confidentiality.
    - **Time:** Seconds elapsed between transaction and first transaction.
    - **Amount:** Transaction amount.
    - **Class:** Target variable (0 = normal, 1 = fraud).

# ⚙️ 6. Workflow
A. **Data Preprocessing:**
1. **Check missing/null values** – this dataset is clean.
2. **Feature scaling:**
    - Scale Time and Amount using StandardScaler.
3. **Train-Test Split:**
    - Use StratifiedShuffleSplit or train_test_split(..., stratify=y) to maintain fraud ratio.

B. **Data Balancing:**
 Due to heavy class imbalance:

- **Why not just accuracy?**
    - A model that predicts everything as "not fraud" still has 99.8% accuracy – useless.

- **Techniques used:**
    - **Under-sampling** – Reduce normal transactions to match fraud count.
    - **Over-sampling** – Duplicate fraud examples.
    - **SMOTE** – Generate synthetic samples of fraud class using K-nearest neighbors.


# 🧠 7. Model Building
Multiple models are trained and compared:

| Model               | Pros                                   | Cons                                |
| ------------------- | -------------------------------------- | ----------------------------------- |
| Logistic Regression | Simple, interpretable                  | May underperform on non-linear data |
| Decision Tree       | Fast, handles imbalance to some extent | Overfits easily                     |
| Random Forest       | High accuracy, robust                  | Slower, less interpretable          |
| XGBoost             | Best performance with tuning           | Complex, requires parameter tuning  |
| ANN (Optional)      | Captures complex patterns              | Data-hungry, harder to train        |


# 📈 8. Model Evaluation Metrics
Because of class imbalance, we focus on:

- **Confusion Matrix**
- **Precision** = TP / (TP + FP)
- **Recall (Sensitivity)** = TP / (TP + FN
- **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)
- **ROC-AUC Score** – Area under ROC curve
- **Goal:** Maximize Recall, reduce False Negatives, but keep Precision reasonable to avoid too many false alerts.


# 📊 9. Results (Example)
| Metric        | Value |
| ------------- | ----- |
| Precision     | 0.91  |
| Recall        | 0.88  |
| F1-Score      | 0.89  |
| ROC-AUC Score | 0.98  |


Interpretation:
- **High recall** = fewer frauds are missed.
- **Acceptable precision** = false alarms kept in check.
- **ROC-AUC near 1** = excellent model discrimination.


# 🧠 10. Key Learnings
- Handling imbalanced data is critical.
- **SMOTE + Random Forest or XGBoost** works wel
- ML models can outperform rule-based systems by learning hidden patterns.
- Real-world deployments must also consider latency, drift, and explainability.


# 📌 11. Tools Used
| Category      | Tools/Libraries                                                    |
| ------------- | ------------------------------------------------------------------ |
| Programming   | Python                                                             |
| Libraries     | Scikit-learn, imbalanced-learn, Pandas, NumPy, Matplotlib, Seaborn |
| IDE/Platforms | Jupyter Notebook / VSCode / Google Colab                           |


