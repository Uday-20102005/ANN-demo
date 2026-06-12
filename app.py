# app.py

import streamlit as st
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "churn_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_artifacts()


st.title("📊 Customer Churn Prediction Using ANN")
st.markdown("Predict whether a customer is likely to leave the bank or stay.")

tab1, tab2 = st.tabs(["Prediction", "Model Information"])

with tab1:

    st.subheader("Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )

    with col2:
        balance = st.number_input(
            "Account Balance",
            min_value=0.0,
            value=50000.0
        )

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=60000.0
        )

    if st.button("Predict Churn"):

        data = np.array([
            [credit_score, age, balance, estimated_salary]
        ])

        data_scaled = scaler.transform(data)

        prediction = model.predict(data_scaled)[0][0]

        if prediction > 0.5:
            st.error("⚠️ Customer is likely to Churn")
            st.write(f"Churn Probability: **{prediction*100:.2f}%**")
        else:
            st.success("✅ Customer is likely to Stay")
            st.write(f"Retention Probability: **{(1-prediction)*100:.2f}%**")


with tab2:

    st.subheader("ANN Architecture")

    st.markdown("""
    - Input Layer (4 Features)
    - Dense Layer (16 Neurons, ReLU)
    - Dense Layer (8 Neurons, ReLU)
    - Output Layer (1 Neuron, Sigmoid)
    """)

    st.subheader("Input Features")

    st.write("""
    - CreditScore
    - Age
    - Balance
    - EstimatedSalary
    """)

    st.subheader("Output")

    st.write("""
    Binary Classification:
    - 0 → Customer Stays
    - 1 → Customer Churns
    """)

    st.subheader("Loss Function")

    st.write("Binary Crossentropy")

    st.subheader("Optimizer")

    st.write("Adam")