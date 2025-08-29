import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load dataset & model
df = pd.read_csv("student-mat.csv", sep=";")
model_data = joblib.load("student_model.pkl")
model = model_data["model"]
features = model_data["features"]

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "EDA", "Prediction", "About"])

# ----------------- HOME PAGE -----------------
if page == "Home":
    st.title("🎓 Student Performance Predictions")
    st.markdown("""
    Welcome to the **Student Performance Predictions App**.  
    Use the sidebar to explore:
    - 📊 **EDA:** Explore patterns and correlations in student data.
    - 🤖 **Prediction:** Enter student details and predict performance.
    - ℹ️ **About:** Learn more about this project.
    """)

# ----------------- EDA PAGE -----------------
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")

    st.markdown("### Select a column to visualize distributions")
    col = st.selectbox("Choose column", df.columns)

    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

    st.markdown("### Correlation Heatmap")
    if st.checkbox("Show Heatmap"):
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    st.markdown("### Scatterplot")
    x_axis = st.selectbox("X-axis", df.columns, index=0)
    y_axis = st.selectbox("Y-axis", df.columns, index=len(df.columns)-1)
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
    st.pyplot(fig)

# ----------------- PREDICTION PAGE -----------------
elif page == "Prediction":
    st.title("🤖 Predict Student Performance")

    user_input = {}
    for col in features:
        user_input[col] = st.number_input(f"Enter {col}:", value=0)

    input_df = pd.DataFrame([user_input])
    pred = model.predict(input_df)[0]
    st.success(f"🎯 Predicted Final Grade (G3): {pred}")

# ----------------- ABOUT PAGE -----------------
elif page == "About":
    st.title("ℹ️ About this Project")
    st.markdown("""
    This project uses **machine learning** to predict student performance 
    based on demographic, social, and academic features.  

    Built with:
    - Streamlit
    - Scikit-learn
    - Seaborn & Matplotlib
    """)
