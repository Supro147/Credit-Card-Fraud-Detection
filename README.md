
# Credit Card Fraud Detection Using Machine Learning

## 1. Problem Statement

Credit card fraud is a major challenge in the financial industry, resulting in significant financial losses for customers and financial institutions. Detecting fraudulent transactions accurately is essential for minimizing these losses.

The objective of this project is to develop and evaluate multiple machine learning classification models to identify fraudulent credit card transactions.

Five classification algorithms are implemented and compared using different evaluation metrics to understand their performance in detecting fraudulent transactions.

The models implemented are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (Linear SVM)

The objective is to identify the model that provides the best performance for credit card fraud detection.

## 2. Dataset Description

**Dataset Name:** Credit Card Fraud Detection

**Dataset Source:** Kaggle – ULB Machine Learning Group

**Dataset Link:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains European credit card transactions made during September 2013.

### Dataset Details

| Attribute | Description |
|---|---|
| Total Records | 284,807 |
| Total Columns | 31 |
| Input Features | 30 |
| Target Variable | Class |
| Legitimate Transactions | 284,315 |
| Fraudulent Transactions | 492 |
| Missing Values | 0 |

### Feature Description

The dataset contains 30 input features:

- **Time:** Number of seconds elapsed between the first transaction and the transaction.
- **V1–V28:** Anonymized features obtained using Principal Component Analysis (PCA).
- **Amount:** Transaction amount.
- **Class:** Target variable indicating whether a transaction is fraudulent or legitimate.

Target variable interpretation:

- Class 0 – Legitimate Transaction
- Class 1 – Fraudulent Transaction

### Data Preprocessing

The following preprocessing steps were performed:

1. Checked the dataset for missing values. No missing values were found.
2. Identified and removed 1,081 duplicate records.
3. Separated the independent features and target variable.
4. Split the dataset into training and testing sets using a 70:30 ratio.
5. Applied stratified sampling to preserve the original class distribution.
6. Applied random undersampling to the training dataset to address class imbalance.

The training dataset was undersampled using a 20:1 ratio of legitimate transactions to fraudulent transactions, retaining all 331 fraudulent training records and selecting 6,620 legitimate training records.

The resulting training dataset contained 6,951 records.

The test dataset was kept separate and was not subjected to undersampling. A stratified sample of 20,000 test records was used for model evaluation.

Feature scaling using StandardScaler was applied within the pipelines for Logistic Regression, KNN, Gaussian Naive Bayes and Linear SVM. Decision Tree was trained without feature scaling.

## 3. GitHub Repository Link

**GitHub Repository:** https://github.com/Supro147/Credit-Card-Fraud-Detection

The repository contains:

- Complete Python source code.
- Jupyter Notebook containing model implementation and evaluation.
- Streamlit application source code (`app.py`).
- Requirements file (`requirements.txt`).
- Test dataset (`test_data.csv`).
- Saved trained models for all five classifiers.
- Project documentation (`README.md`).

## 4. Models Used and Evaluation

Five machine learning classification algorithms were implemented using the same dataset and evaluated using the same held-out test sample.

The following six evaluation metrics were calculated:

1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coefficient (MCC)

### Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9736 | 0.9431 | 0.0475 | 0.7879 | 0.0897 | 0.1896 |
| Decision Tree | 0.9914 | 0.8142 | 0.1160 | 0.6364 | 0.1963 | 0.2693 |
| KNN | 0.9990 | 0.8452 | 0.7333 | 0.6667 | 0.6984 | 0.6987 |
| Naive Bayes | 0.9761 | 0.9248 | 0.0468 | 0.6970 | 0.0878 | 0.1766 |
| Support Vector Machine | 0.9772 | 0.9423 | 0.0546 | 0.7879 | 0.1022 | 0.2038 |

### Observations on Model Performance

**1. Logistic Regression**

Logistic Regression achieved an accuracy of 97.36% and an AUC score of 0.9431. It achieved a recall of 78.79%, indicating its ability to identify a significant proportion of fraudulent transactions. However, its precision was relatively low at 4.75%, indicating a high number of false positive predictions.

**2. Decision Tree Classifier**

The Decision Tree classifier achieved an accuracy of 99.14%. Its AUC score was 0.8142, while its recall was 63.64%. The model demonstrated reasonable classification performance, but its precision of 11.60% and F1 score of 0.1963 indicate room for improvement.

**3. K-Nearest Neighbors (KNN)**

KNN achieved the highest accuracy of 99.90%, precision of 73.33%, F1 score of 0.6984 and MCC score of 0.6987 among the five models. It achieved a recall of 66.67% and an AUC score of 0.8452. The model demonstrated a strong balance between precision and recall and produced the best overall results based on the evaluated metrics.

**4. Gaussian Naive Bayes**

Gaussian Naive Bayes achieved an accuracy of 97.61% and an AUC score of 0.9248. Its recall was 69.70%, indicating that it detected a considerable proportion of fraudulent transactions. However, its precision of 4.68% and F1 score of 0.0878 were relatively low.

**5. Support Vector Machine**

The Linear Support Vector Machine achieved an accuracy of 97.72% and an AUC score of 0.9423. It achieved a recall of 78.79%, matching Logistic Regression. However, its precision was 5.46%, indicating that it generated a relatively high number of false positive predictions.

### Overall Winner for the Dataset

Based on the evaluation results, **K-Nearest Neighbors (KNN)** performed best overall among the five classification models.

It achieved the highest accuracy, precision, F1 score and Matthews Correlation Coefficient.

Although Logistic Regression achieved the highest AUC score of 0.9431, KNN demonstrated a better balance between precision and recall at the selected classification threshold.

Therefore, KNN was selected as the overall best-performing model for this experiment.

## 5. Streamlit Application

An interactive web application was developed using Streamlit to demonstrate and compare the performance of the five trained classification models.

### Application Features

1. CSV dataset upload functionality.
2. Model selection dropdown containing all five classifiers.
3. Display of evaluation metrics: Accuracy, AUC, Precision, Recall, F1 Score and MCC.
4. Confusion matrix visualization.
5. Classification report.
6. Comparison of all five models on the uploaded test dataset.

The application loads previously trained models and evaluates the uploaded test dataset without retraining the models.

**Live Streamlit Application:** [Add your deployed Streaml

## 6. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Git and GitHub

## 7. Conclusion

This project demonstrates the implementation and evaluation of five machine learning classification algorithms for credit card fraud detection.

The dataset presented a highly imbalanced classification problem, with fraudulent transactions representing only a small proportion of the total records.

Different preprocessing techniques and model configurations were applied to address this challenge.

Based on the experimental results, KNN achieved the strongest overall performance, particularly in terms of precision, F1 score and MCC.

The project also demonstrates an end-to-end machine learning workflow, including data preprocessing, model training, performance evaluation, model serialization and deployment through an interactive Streamlit application.
