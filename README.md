# 🌐 Fuzzy Random Forest for Customer Churn Prediction

A Machine Learning project that integrates **Fuzzy Logic with Random Forest** for customer churn prediction and risk assessment across three datasets: Telco, Bank, and Telecom.

The project compares a baseline Random Forest model with a Fuzzy-enhanced Random Forest model by generating fuzzy membership features and incorporating them into the training process. It also categorizes customers into different churn risk levels based on predicted probabilities.

---

## 🎯 Project Overview

This project explores how fuzzy feature engineering can be combined with traditional machine learning to analyze customer churn behavior.

The notebook implements a complete machine learning pipeline, including:

1. Data Loading and Inspection
2. Exploratory Data Analysis (EDA)
3. Data Cleaning and Missing Value Handling
4. Data Preprocessing and Feature Transformation
5. Fuzzy Membership Function Design
6. Fuzzy Feature Generation
7. Baseline Random Forest Model Training
8. Fuzzy Random Forest Model Training
9. Model Performance Comparison
10. Customer Churn Risk Assessment
11. Feature Importance Analysis
12. Conclusion and Recommendations

---

## 📊 Datasets

Three customer churn datasets were used for experimentation.

| Dataset | Records | Target Variable |
|---|---:|---|
| Telco Customer Churn | 7,043 | `Churn` |
| Bank Customer Churn | 10,000 | `churn` |
| Telecom Churn | 243,553 | `churn` |

**Dataset Files:**
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- `Bank Customer Churn Prediction.csv`
- `telecom_churn.csv`

> **Note:** Dataset files are not included in this repository. Download them separately and place them in the project root directory, or update the file paths in the notebook.

---

## ⚙️ Methodology

### 1. Data Cleaning and Preprocessing

- Removed unnecessary whitespace from column values.
- Converted `TotalCharges` into numeric format for the Telco dataset.
- Removed unnecessary ID columns.
- Handled missing values using median imputation for numerical features and mode imputation for categorical features.
- Applied `StandardScaler` to numerical features.
- Applied `OneHotEncoder` to categorical features using `ColumnTransformer`.

### 2. Fuzzy Logic Integration

Fuzzy Logic was integrated into the feature engineering stage to represent selected numerical attributes through membership degrees ranging from 0 to 1.

The following membership functions were implemented:

- Triangular Membership Function
- Left Shoulder Membership Function
- Right Shoulder Membership Function

**Selected Fuzzy Features:**

| Dataset | Selected Features |
|---|---|
| Telco | `tenure`, `MonthlyCharges` |
| Bank | `credit_score`, `age` |
| Telecom | `age`, `data_used` |

The generated fuzzy membership values were combined with the preprocessed input features before training the Fuzzy Random Forest model.

### 3. Machine Learning Models

Two approaches were implemented and compared:

- **Baseline Random Forest:** Trained using the original preprocessed features.
- **Fuzzy Random Forest:** Trained using the original preprocessed features combined with generated fuzzy membership features.

**Model Configuration:**
- Algorithm: `RandomForestClassifier`
- Train-Test Split: 70:30
- Stratified Sampling: Enabled
- Random State: 42

### 4. Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

---

## 📈 Experimental Results

The following table presents the performance comparison between the baseline and fuzzy-enhanced approaches.

| Dataset | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Telco | Baseline RF | 0.7837 | 0.6171 | 0.4884 | 0.5453 | 0.8131 |
| Telco | Fuzzy RF | 0.7795 | 0.6063 | 0.4831 | 0.5377 | 0.8174 |
| Bank | Baseline RF | 0.8633 | 0.7799 | 0.4583 | 0.5773 | 0.8600 |
| Bank | Fuzzy RF | 0.8640 | 0.7812 | 0.4615 | 0.5802 | 0.8589 |
| Telecom | Baseline RF | 0.7991 | 0.2203 | 0.0009 | 0.0018 | 0.4985 |
| Telecom | Fuzzy RF | 0.7990 | 0.2174 | 0.0010 | 0.0020 | 0.4980 |

### 🔍 Key Observations

- Fuzzy feature integration resulted in relatively small performance changes across the three datasets.
- The Telco dataset showed a slight ROC-AUC improvement with the Fuzzy Random Forest approach.
- The Bank dataset showed marginal improvements in accuracy, precision, recall, and F1-score, while ROC-AUC decreased slightly.
- Both models struggled to identify churn cases in the Telecom dataset, despite achieving approximately 80% accuracy.
- Low churn-class recall highlights the impact of class imbalance on model performance.

These findings indicate that fuzzy feature integration does not necessarily improve every evaluation metric or dataset.

---

## 🚦 Customer Churn Risk Assessment

In addition to binary churn prediction, the project introduces a customer risk categorization approach based on predicted churn probabilities.

| Risk Level | Probability Range |
|---|---|
| 🟢 Low Risk | ≤ 0.30 |
| 🟡 Medium Risk | > 0.30 and ≤ 0.70 |
| 🔴 High Risk | > 0.70 |

This approach provides an additional interpretation layer to help identify customers who may require retention-focused attention.

---

## 🛠️ Technologies and Tools

**Programming Language:**
- Python 3

**Libraries:**
- pandas
- NumPy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
- joblib

**Development Environment:**
- Google Colab / Jupyter Notebook

---

## 🚀 Installation and Execution

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib jupyter
```

### 3. Prepare the Datasets

Download the three CSV datasets and place them in the project root directory.

### 4. Launch the Notebook

```bash
jupyter notebook fuzzy_rf_churn_project.ipynb
```

### 5. Run the Notebook

Execute all cells sequentially, from data loading and preprocessing to model training, evaluation, and customer risk assessment.

---

## 📁 Project Structure

```text
Fuzzy-Random-Forest-Customer-Churn/
│
├── fuzzy_rf_churn_project.ipynb
├── README.md
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── Bank Customer Churn Prediction.csv
└── telecom_churn.csv
```

*Note: The dataset files are excluded from the repository and need to be downloaded separately.*

---

## ⚠️ Limitations and Future Work

### Current Limitations

- Fuzzy membership functions are manually defined using descriptive statistics rather than optimized through a learning process.
- Class imbalance significantly affects churn-class recall, particularly in the Telecom dataset.
- Fuzzy feature integration produces only marginal performance changes in the current experiments.
- The Telecom dataset demonstrates very limited churn-class detection performance.

### Future Improvements

- Optimize fuzzy membership functions using data-driven approaches.
- Explore class imbalance handling techniques such as SMOTE and class weighting.
- Investigate Adaptive Neuro-Fuzzy Inference Systems (ANFIS).
- Perform hyperparameter optimization.
- Compare performance with additional algorithms such as Gradient Boosting and Logistic Regression.

---

## 👩‍💻 Project Information

**Project Title:** Fuzzy Random Forest for Customer Churn Prediction

**Project Domain:** Artificial Intelligence and Intelligent Computing

**Core Areas:** Fuzzy Logic, Random Forest, Machine Learning, Feature Engineering, Customer Churn Prediction

**Developer:** Himadree Chowdhury

**GitHub:** [@HimadreeChy](https://github.com/HimadreeChy)

---

## 📄 License

This project is developed for academic and educational purposes.
