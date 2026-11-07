
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

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
st.write(
    "A machine learning application to detect fraudulent "
    "credit card transactions using five classification models."
)

st.markdown("---")


st.header("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", "284,807")
col2.metric("Features", "30")
col3.metric("Fraud Transactions", "492")
col4.metric("Fraud Percentage", "0.173%")

st.write(
    """
    **Dataset:** Kaggle Credit Card Fraud Detection Dataset

    The dataset contains anonymized credit card transaction features.
    The target variable is `Class`, where:

    - `0` represents a legitimate transaction.
    - `1` represents a fraudulent transaction.
    """
)

st.markdown("---")


MODEL_PATH = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}


@st.cache_resource
def load_models():

    models = {}

    for model_name, file_name in MODEL_FILES.items():

        file_path = os.path.join(MODEL_PATH, file_name)

        if os.path.exists(file_path):
            models[model_name] = joblib.load(file_path)

    return models


models = load_models()


if len(models) != len(MODEL_FILES):

    st.error(
        "Some model files are missing. "
        "Please check that all five .joblib files exist inside the model folder."
    )

    st.stop()


st.header("Upload Test Dataset")

st.write(
    """
    Upload a CSV file containing the transaction features and the
    actual target column (`Class`). Select a trained model to evaluate
    its performance on the uploaded dataset.
    """
)

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

selected_model_name = st.selectbox(
    "Select a Machine Learning Model",
    list(MODEL_FILES.keys())
)

st.caption(
    "The uploaded CSV must contain all 30 input features and the target column `Class`."
)


def evaluate_model(model, X, y):

    predictions = model.predict(X)

    # Get scores for ROC-AUC
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)[:, 1]

    elif hasattr(model, "decision_function"):

        probabilities = model.decision_function(X)

    else:

        probabilities = predictions

    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": roc_auc_score(y, probabilities),
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y, predictions, zero_division=0
        ),
        "MCC": matthews_corrcoef(y, predictions)
    }

    return predictions, metrics


if uploaded_file is not None:

    try:

        uploaded_data = pd.read_csv(uploaded_file)

        st.success("CSV file uploaded successfully!")

        st.subheader("Uploaded Dataset Preview")

        st.dataframe(
            uploaded_data.head(),
            use_container_width=True
        )

        st.write(
            f"**Number of rows:** {uploaded_data.shape[0]:,}"
        )

        st.write(
            f"**Number of columns:** {uploaded_data.shape[1]}"
        )

        # Check target column
        if "Class" not in uploaded_data.columns:

            st.error(
                "The uploaded dataset must contain a target column named 'Class'."
            )

            st.stop()

        # Extract features and target
        X = uploaded_data.drop(columns=["Class"])
        y = uploaded_data["Class"]

        # Get expected feature names from the selected model
        selected_model = models[selected_model_name]

        if hasattr(selected_model, "feature_names_in_"):

            expected_features = list(
                selected_model.feature_names_in_
            )

        else:

            st.error(
                "The selected model does not contain saved feature names. "
                "Please check how the model was trained."
            )

            st.stop()

        # Check missing and extra columns
        missing_features = [
            feature for feature in expected_features
            if feature not in X.columns
        ]

        extra_features = [
            feature for feature in X.columns
            if feature not in expected_features
        ]

        if missing_features:

            st.error(
                f"Missing required features: {missing_features}"
            )

            st.stop()

        if extra_features:

            st.warning(
                f"Extra columns will be ignored: {extra_features}"
            )

        # Reorder features to match training order
        X = X[expected_features]

        # Check missing values
        if X.isnull().values.any() or y.isnull().any():

            st.error(
                "The uploaded dataset contains missing values. "
                "Please clean the dataset before uploading."
            )

            st.stop()

        # Check numeric features
        if not all(
            pd.api.types.is_numeric_dtype(X[col])
            for col in X.columns
        ):

            st.error(
                "All input features must be numeric."
            )

            st.stop()

        # Validate target
        if not set(y.unique()).issubset({0, 1}):

            st.error(
                "The target column 'Class' must contain only 0 and 1."
            )

            st.stop()

        if y.nunique() != 2:

            st.error(
                "The uploaded dataset must contain both classes: "
                "0 (legitimate) and 1 (fraudulent), to calculate AUC."
            )

            st.stop()

        # Evaluate selected model
        predictions, metrics = evaluate_model(
            selected_model,
            X,
            y
        )

        st.markdown("---")

        st.header("📈 Model Evaluation Results")

        st.subheader(
            f"Results for {selected_model_name}"
        )

        # Display metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)

        metric_col1.metric(
            "Accuracy",
            f"{metrics['Accuracy']:.4f}"
        )

        metric_col2.metric(
            "AUC",
            f"{metrics['AUC']:.4f}"
        )

        metric_col3.metric(
            "Precision",
            f"{metrics['Precision']:.4f}"
        )

        metric_col4, metric_col5, metric_col6 = st.columns(3)

        metric_col4.metric(
            "Recall",
            f"{metrics['Recall']:.4f}"
        )

        metric_col5.metric(
            "F1 Score",
            f"{metrics['F1 Score']:.4f}"
        )

        metric_col6.metric(
            "MCC",
            f"{metrics['MCC']:.4f}"
        )

        # Confusion matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y,
            predictions,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_title("Confusion Matrix")

        st.pyplot(fig)

        plt.close(fig)

        # Classification report
        st.subheader("Classification Report")

        report = classification_report(
            y,
            predictions,
            target_names=["Legitimate", "Fraud"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"An error occurred while processing the dataset: {e}"
        )


st.markdown("---")

st.header("Model Performance Comparison")

st.write(
    """
    The following table shows the performance obtained during
    the notebook evaluation on the same held-out test sample.
    These are the previously recorded results, not results calculated
    from the currently uploaded CSV.
    """
)

comparison_data = {
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Naive Bayes",
        "Support Vector Machine"
    ],

    "Accuracy": [
        0.9736,
        0.9914,
        0.9990,
        0.9761,
        0.9772
    ],

    "AUC": [
        0.9431,
        0.8142,
        0.8452,
        0.9248,
        0.9423
    ],

    "Precision": [
        0.0475,
        0.1160,
        0.7333,
        0.0468,
        0.0546
    ],

    "Recall": [
        0.7879,
        0.6364,
        0.6667,
        0.6970,
        0.7879
    ],

    "F1 Score": [
        0.0897,
        0.1963,
        0.6984,
        0.0878,
        0.1022
    ],

    "MCC": [
        0.1896,
        0.2693,
        0.6987,
        0.1766,
        0.2038
    ]
}

comparison_df = pd.DataFrame(comparison_data)

st.dataframe(
    comparison_df.set_index("Model").style.format("{:.4f}"),
    use_container_width=True
)

st.subheader("Model Performance Visualization")

chart_metric = st.selectbox(
    "Choose a metric to compare",
    [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC"
    ]
)

fig, ax = plt.subplots(figsize=(10, 5))

sns.barplot(
    data=comparison_df,
    x="Model",
    y=chart_metric,
    ax=ax
)

ax.set_title(
    f"Model Comparison Based on {chart_metric}"
)

ax.set_xlabel("Machine Learning Models")
ax.set_ylabel(chart_metric)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)


st.markdown("---")

st.header("Overall Best Performing Model")

st.write(
    """
    Based on the recorded test evaluation, **K-Nearest Neighbors (KNN)**
    achieved the highest Accuracy, Precision, F1 Score, and MCC.

    Logistic Regression achieved the highest AUC.

    KNN is the overall best-performing model based on the combined
    classification metrics in this experiment. However, because the
    dataset is highly imbalanced, Accuracy should not be considered
    in isolation.
    """
)

st.markdown("---")

st.caption(
    "BITS Pilani | M.Tech Artificial Intelligence and Machine Learning | "
    "Machine Learning Assignment 2"
)
