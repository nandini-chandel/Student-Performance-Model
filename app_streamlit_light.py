import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.title("📊 Student Performance Predictor (Simple Edition)")

# Upload CSV or use sample
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("No file uploaded. Using sample dataset.")
    df = pd.DataFrame({
        "Hours_Studied": [2, 4, 6, 8, 10, 12],
        "Attendance": [60, 70, 80, 85, 90, 95],
        "Previous_Scores": [50, 60, 65, 70, 75, 80],
        "Score": [55, 65, 70, 75, 85, 90]
    })

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Features and target
features = [c for c in df.columns if c != "Score"]
target = "Score"

# Choose mode
mode = st.radio("Prediction Mode", ["Regression", "Classification"])

X = df[features]
y = df[target]

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if mode == "Regression":
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    st.write("RMSE:", mean_squared_error(y_test, preds, squared=False))
elif mode == "Classification":
    # Convert numeric target to Pass/Fail
    y_class = (y >= y.median()).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    st.write("Accuracy:", accuracy_score(y_test, preds))

# Prediction input
st.subheader("Try Prediction")
input_data = {}
for col in features:
    val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    input_data[col] = val

input_df = pd.DataFrame([input_data])
if st.button("Predict"):
    result = model.predict(input_df)[0]
    st.success(f"Predicted Value: {result}")
