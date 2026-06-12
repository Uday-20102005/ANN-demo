import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

st.set_page_config(
    page_title="ANN Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# Load model and preprocessors
model = tf.keras.models.load_model("models/ann_churn.keras")

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/geography_encoder.pkl", "rb") as f:
    geography_encoder = pickle.load(f)

with open("models/gender_encoder.pkl", "rb") as f:
    gender_encoder = pickle.load(f)

st.title("ANN Customer Churn Prediction")

st.markdown("Enter customer details below.")

credit_score = st.slider(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

geography = st.selectbox(
    "Geography",
    geography_encoder.classes_.tolist()
)

gender = st.selectbox(
    "Gender",
    gender_encoder.classes_.tolist()
)

age = st.slider(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

tenure = st.slider(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)

num_products = st.slider(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=1
)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)

if st.button("Predict Churn"):

    geography_encoded = geography_encoder.transform([geography])[0]
    gender_encoded = gender_encoder.transform([gender])[0]

    input_data = np.array([[
        credit_score,
        geography_encoded,
        gender_encoded,
        age,
        tenure,
        balance,
        num_products,
        has_cr_card,
        is_active_member,
        estimated_salary
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled, verbose=0)

    churn_probability = float(prediction[0][0])

    st.subheader(
        f"Churn Probability: {churn_probability:.2%}"
    )

    if churn_probability >= 0.5:
        st.error("Customer is likely to leave.")
    else:
        st.success("Customer is likely to stay.")