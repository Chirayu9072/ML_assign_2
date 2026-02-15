import streamlit as st
import pandas as pd
import os
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


st.title("Adult Census Income Classification")


columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]


if "data" not in st.session_state:
    st.session_state.data = pd.read_csv("adult.data", header=None, names=columns)

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file is not None:
    st.session_state.data = pd.read_csv(uploaded_file, names=columns)

st.write("Preview of dataset:", st.session_state.data.head())


if "processed_data" not in st.session_state:
    data = st.session_state.data.replace("?", pd.NA).dropna()
    categorical_cols = data.select_dtypes(include="object").columns
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    st.session_state.processed_data = data

data = st.session_state.processed_data
X = data.drop("income", axis=1)
y = data["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

st.write("Preprocessing complete. Training data shape:", X_train.shape)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "kNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}


if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}

results = []

for name, model in models.items():
    filename = f"model/{name.replace(' ', '_').lower()}.pkl"
    os.makedirs("model", exist_ok=True)

    if name in st.session_state.trained_models:
        trained_model = st.session_state.trained_models[name]
    elif os.path.exists(filename):
        trained_model = joblib.load(filename)
        st.session_state.trained_models[name] = trained_model
    else:
        model.fit(X_train, y_train)
        joblib.dump(model, filename)
        trained_model = model
        st.session_state.trained_models[name] = trained_model

    preds = trained_model.predict(X_test)
    probs = trained_model.predict_proba(X_test)[:, 1] if hasattr(trained_model, "predict_proba") else None

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "MCC": matthews_corrcoef(y_test, preds),
        "AUC": roc_auc_score(y_test, probs) if probs is not None else None
    }
    results.append(metrics)

results_df = pd.DataFrame(results)
st.write("Model Comparison Table", results_df)

if st.button("Retrain Models"):
    st.session_state.trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        filename = f"model/{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(model, filename)
        st.session_state.trained_models[name] = model
    st.success("Models retrained successfully! Please reselect a model to view updated results.")

selected_model = st.selectbox("Select a model to view details", list(models.keys()))

if selected_model:
    model = st.session_state.trained_models[selected_model]
    preds = model.predict(X_test)

  
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)


    st.text("Classification Report:")
    st.text(classification_report(y_test, preds))

    st.subheader("Model Performance Observation")
    observations = {
        "Logistic Regression": "Achieved ~82% accuracy. Performs well on linearly separable features, but struggles with complex non-linear relationships. Higher precision for class 0, weaker recall for class 1.",
        "Decision Tree": "Simple and interpretable, but tends to overfit. Performance is moderate and unstable depending on depth/split criteria.",
        "kNN": "Sensitive to feature scaling and choice of k. Reasonable performance but slower on larger datasets. Recall for minority class is weaker.",
        "Naive Bayes": "Fast and lightweight. Works well with categorical features but independence assumption limits accuracy. Precision/recall lower than other models.",
        "Random Forest": "Strong performance due to averaging across multiple trees. More robust than a single Decision Tree, with balanced precision/recall.",
        "XGBoost": "Best overall (~87% accuracy). Handles feature interactions and imbalanced data effectively. Strong precision, recall, and F1 scores."
    }
    st.write(f"**{selected_model}:** {observations[selected_model]}")
