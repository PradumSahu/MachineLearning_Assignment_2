import streamlit as st
import pandas as pd
import joblib

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
    page_title="ML Classification Models",
    layout="wide"
)

st.title("Machine Learning Assignment 2")
st.subheader("Classification Model Comparison")

models = {
    "Logistic Regression":
        "model/logistic_regression.pkl",

    "Decision Tree":
        "model/decision_tree.pkl",

    "KNN":
        "model/knn.pkl",

    "Naive Bayes":
        "model/naive_bayes.pkl",

    "Random Forest":
        "model/random_forest.pkl"
}

uploaded_file = st.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

selected_model = st.selectbox(
    "Select Model",
    list(models.keys())
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model = joblib.load(models[selected_model])

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    st.header("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}")
        st.metric("Precision", f"{precision:.4f}")

    with col2:
        st.metric("AUC", f"{auc:.4f}")
        st.metric("Recall", f"{recall:.4f}")

    with col3:
        st.metric("F1 Score", f"{f1:.4f}")
        st.metric("MCC", f"{mcc:.4f}")

    st.header("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Pred 0", "Pred 1"]
    )

    st.dataframe(cm_df)

    st.header("Classification Report")

    report = classification_report(
        y,
        y_pred,
        output_dict=True
    )

    st.dataframe(
        pd.DataFrame(report).transpose()
    )