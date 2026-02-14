import streamlit as st
import pandas as pd

# Column names from UCI Adult dataset
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

st.title("Adult Census Income Classification")

# Option 1: Load from repo
data = pd.read_csv("adult.data", header=None, names=columns)

# Option 2: Allow upload (mandatory for assignment)
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, names=columns)

st.write("Preview of dataset:", data.head())


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Handle missing values
data = data.replace("?", pd.NA).dropna()

# Encode categorical features
categorical_cols = data.select_dtypes(include="object").columns
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Split features and target
X = data.drop("income", axis=1)
y = data["income"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

st.write("Preprocessing complete. Training data shape:", X_train.shape)
