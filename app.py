
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)




st.title("Credit Card Fraud Detection")
st.markdown(
    """
    ### Machine Learning Classification Dashboard

    An interactive application to evaluate and compare five
    machine learning algorithms for detecting fraudulent
    credit card transactions.
    """
)

st.divider()


st.header("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="Total Transactions",
    value="284,807"
)

col2.metric(
    label="Fraud Transactions",
    value="492"
)

col3.metric(
    label="Legitimate Transactions",
    value="284,315"
)

col4.metric(
    label="Fraud Percentage",
    value="0.173%"
)

st.caption(
    "Dataset: Credit Card Fraud Detection (ULB) | "
    "30 input features | Binary classification"
)

st.divider()

st.header("Trained Model Performance")

st.markdown(
    """
    The following results represent the previously trained
    classification models evaluated on the same 20,000-row
    stratified test sample.
    """
)

comparison_data = {
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Support Vector Machine"
    ],

    "Accuracy": [
        0.9736, 0.9914, 0.9990, 0.9761, 0.9772
    ],

    "AUC": [
        0.9431, 0.8142, 0.8452, 0.9248, 0.9423
    ],

    "Precision": [
        0.0475, 0.1160, 0.7333, 0.0468, 0.0546
    ],

    "Recall": [
        0.7879, 0.6364, 0.6667, 0.6970, 0.7879
    ],

    "F1 Score": [
        0.0897, 0.1963, 0.6984, 0.0878, 0.1022
    ],

    "MCC": [
        0.1896, 0.2693, 0.6987, 0.1766, 0.2038
    ]
}

comparison_df = pd.DataFrame(comparison_data)

st.dataframe(
    comparison_df.style.format({
        "Accuracy": "{:.4f}",
        "AUC": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1 Score": "{:.4f}",
        "MCC": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)


st.subheader("Model Performance Comparison")

chart_data = comparison_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score", "MCC"]
]

st.bar_chart(chart_data)

st.info(
    "KNN achieved the highest Accuracy, Precision, "
    "F1 Score and MCC among the five evaluated models."
)



st.subheader("Overall Best Performing Model")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Selected Best Model",
    "KNN"
)

col2.metric(
    "Best Accuracy",
    "99.90%"
)

col3.metric(
    "Best F1 Score",
    "0.6984"
)

st.markdown(
    """
    **Observation:**

    K-Nearest Neighbors achieved the strongest overall
    performance based on Accuracy, Precision, F1 Score
    and Matthews Correlation Coefficient.

    Logistic Regression achieved the highest AUC score
    of 0.9431 among the five models.
    """
)

st.divider()


MODEL_PATH = "model"

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}


@st.cache_resource
def load_models():

    loaded_models = {}

    for name, filename in model_files.items():

        path = os.path.join(MODEL_PATH, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        loaded_models[name] = joblib.load(path)

    return loaded_models


try:
    models = load_models()

except Exception as e:
    st.error(f"Error loading trained models: {e}")
    st.stop()


st.header("Test Your Models")

st.markdown(
    """
    Upload a test dataset in CSV format to evaluate the
    performance of the trained classification models.

    The CSV must contain all 30 input features and the
    target column `Class`.
    """
)

uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)



st.sidebar.title("Model Configuration")

selected_model = st.sidebar.selectbox(
    "Select Classification Model",
    list(model_files.keys())
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    ### Available Models

    1. Logistic Regression
    2. Decision Tree
    3. K-Nearest Neighbors
    4. Gaussian Naive Bayes
    5. Support Vector Machine
    """
)


if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset Preview")

        st.write("Dataset Shape:", df.shape)

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        # Check target column

        if "Class" not in df.columns:

            st.error(
                "The uploaded CSV must contain the "
                "'Class' target column."
            )

            st.stop()

        # Separate features and target

        X = df.drop(columns=["Class"])
        y = df["Class"]

        # Expected feature columns

        selected_estimator = models[selected_model]

        if hasattr(selected_estimator, "feature_names_in_"):

            expected_columns = list(
                selected_estimator.feature_names_in_
            )

        else:

            expected_columns = list(X.columns)

        # Validate feature columns

        if set(X.columns) != set(expected_columns):

            missing_columns = list(
                set(expected_columns) - set(X.columns)
            )

            extra_columns = list(
                set(X.columns) - set(expected_columns)
            )

            st.error(
                f"Feature mismatch. Missing columns: "
                f"{missing_columns}. Extra columns: {extra_columns}"
            )

            st.stop()

        # Ensure correct feature order

        X = X[expected_columns]

        # Validate missing values

        if X.isnull().any().any() or y.isnull().any():

            st.error(
                "The uploaded dataset contains missing values. "
                "Please clean the dataset before uploading."
            )

            st.stop()

        # Validate target values

        if not set(y.unique()).issubset({0, 1}):

            st.error(
                "Target column must contain only 0 and 1."
            )

            st.stop()

        if y.nunique() != 2:

            st.error(
                "The uploaded dataset must contain both "
                "legitimate and fraudulent transactions."
            )

            st.stop()

        # Convert target to integer

        y = y.astype(int)

        st.subheader("Model Performance on Uploaded Data")

        uploaded_results = {}

        for name, model in models.items():

            predictions = model.predict(X)

            if hasattr(model, "predict_proba"):

                scores = model.predict_proba(X)[:, 1]

            else:

                scores = model.decision_function(X)

            uploaded_results[name] = {

                "Accuracy": accuracy_score(
                    y, predictions
                ),

                "AUC": roc_auc_score(
                    y, scores
                ),

                "Precision": precision_score(
                    y, predictions,
                    zero_division=0
                ),

                "Recall": recall_score(
                    y, predictions,
                    zero_division=0
                ),

                "F1 Score": f1_score(
                    y, predictions,
                    zero_division=0
                ),

                "MCC": matthews_corrcoef(
                    y, predictions
                )
            }

        uploaded_results_df = pd.DataFrame(
            uploaded_results
        ).T

        st.dataframe(
            uploaded_results_df.style.format("{:.4f}"),
            use_container_width=True
        )


        st.divider()

        st.subheader(
            f"Detailed Evaluation: {selected_model}"
        )

        model = models[selected_model]

        y_pred = model.predict(X)

        if hasattr(model, "predict_proba"):

            y_scores = model.predict_proba(X)[:, 1]

        else:

            y_scores = model.decision_function(X)

        selected_auc = roc_auc_score(y, y_scores)

        selected_accuracy = accuracy_score(y, y_pred)

        selected_precision = precision_score(
            y, y_pred, zero_division=0
        )

        selected_recall = recall_score(
            y, y_pred, zero_division=0
        )

        selected_f1 = f1_score(
            y, y_pred, zero_division=0
        )

        selected_mcc = matthews_corrcoef(y, y_pred)


        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            f"{selected_accuracy:.4f}"
        )

        col2.metric(
            "AUC Score",
            f"{selected_auc:.4f}"
        )

        col3.metric(
            "Precision",
            f"{selected_precision:.4f}"
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Recall",
            f"{selected_recall:.4f}"
        )

        col5.metric(
            "F1 Score",
            f"{selected_f1:.4f}"
        )

        col6.metric(
            "MCC Score",
            f"{selected_mcc:.4f}"
        )

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y,
            y_pred,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[
                "Legitimate",
                "Fraud"
            ],
            yticklabels=[
                "Legitimate",
                "Fraud"
            ],
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")

        ax.set_title(
            f"Confusion Matrix - {selected_model}"
        )

        st.pyplot(fig)

        plt.close(fig)

        st.subheader("Classification Report")

        report = classification_report(
            y,
            y_pred,
            labels=[0, 1],
            target_names=[
                "Legitimate",
                "Fraud"
            ],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).T

        st.dataframe(
            report_df.style.format("{:.4f}"),
            use_container_width=True
        )



        st.success(
            f"Evaluation completed successfully using "
            f"{selected_model}!"
        )

    except Exception as e:

        st.error(
            f"An error occurred while processing the dataset: {e}"
        )


else:

    st.info(
        "Upload your test CSV file above to view detailed "
        "model evaluation, confusion matrix and classification report."
    )


st.divider()

st.markdown(
    """
    <div style="text-align: center;">

    **Credit Card Fraud Detection Using Machine Learning**

    Developed as part of BITS Pilani M.Tech (AIML)
    Machine Learning Assignment 2.

    Models: Logistic Regression | Decision Tree | KNN |
    Naive Bayes | Support Vector Machine

    </div>
    """,
    unsafe_allow_html=True
)
