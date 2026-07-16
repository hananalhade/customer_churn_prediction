import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction App")

# Load model
model = joblib.load("model.pkl")

# Load data
df = pd.read_csv("telecom_db.csv")

st.subheader("Data Preview")
st.dataframe(df.head())

st.subheader("Make a Prediction")
tenure = st.slider("Tenure", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)

if st.button("Predict Churn"):
    input_data = pd.DataFrame([[tenure, monthly_charges]], columns=["tenure", "MonthlyCharges"])
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("⚠️ This customer is likely to CHURN")
    else:
        st.success("✅ This customer is likely to STAY")
