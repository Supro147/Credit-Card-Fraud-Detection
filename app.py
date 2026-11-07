
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
st.write(
    "An interactive comparison of five machine learning "
    "classification models for detecting fraudulent transactions."
)

MODEL_PATH = "model"

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

# Load saved models
@st.cache_resource
def load_models():
    loaded = {}

    for name, filename in model_files.items():
        path = os.path.join(MODEL_PATH, filename)
        loaded[name] = joblib.load(path)

    return loaded

try:
    models = load_models()
except Exception as e:
    st.error(f"Could not load models: {e}")
    st.stop()

st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose a classification model",
    list(model_files.keys())
)

uploaded_file = st.file_uploader(
    "Upload your test dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.write("Dataset shape:", df.shape)
    st.dataframe(df.head())

    if "Class" not in df.columns:
        st.error("The uploaded CSV must contain the Class target column.")
        st.stop()

    X = df.drop(columns=["Class"])
    y = df["Class"]

    expected_columns = list(models[selected_model].feature_names_in_) \
        if hasattr(models[selected_model], "feature_names_in_") else None

    if expected_columns is not None and list(X.columns) != expected_columns:
        st.error("Input features do not match the trained model.")
        st.stop()

    if y.nunique() != 2:
        st.error("The target must contain both classes (0 and 1).")
        st.stop()

    if not set(y.unique()).issubset({0, 1}):
        st.error("Class values must be 0 and 1.")
        st.stop()

    st.subheader("Model Performance Comparison")

    results = {}

    for name, model in models.items():

        predictions = model.predict(X)

        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)[:, 1]
        else:
            scores = model.decision_function(X)

        results[name] = {
            "Accuracy": accuracy_score(y, predictions),
            "AUC": roc_auc_score(y, scores),
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

    results_df = pd.DataFrame(results).T
    st.dataframe(
        results_df.style.format("{:.4f}"),
        use_container_width=True
    )

    st.subheader(f"Detailed Evaluation: {selected_model}")

    model = models[selected_model]
    predictions = model.predict(X)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Accuracy",
            f"{results[selected_model]['Accuracy']:.4f}"
        )

    with col2:
        st.metric(
            "Precision",
            f"{results[selected_model]['Precision']:.4f}"
        )

    with col3:
        st.metric(
            "Recall",
            f"{results[selected_model]['Recall']:.4f}"
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric(
            "AUC",
            f"{results[selected_model]['AUC']:.4f}"
        )

    with col5:
        st.metric(
            "F1 Score",
            f"{results[selected_model]['F1 Score']:.4f}"
        )

    with col6:
        st.metric(
            "MCC",
            f"{results[selected_model]['MCC']:.4f}"
        )

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, predictions)

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

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification Report")

    report = classification_report(
        y,
        predictions,
        target_names=["Legitimate", "Fraud"],
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).T.style.format("{:.4f}"),
        use_container_width=True
    )

else:
    st.info("Please upload your test CSV file to evaluate the models.")

    st.write("Available models:")
    for name in model_files:
        st.write("- " + name)
