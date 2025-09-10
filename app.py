import pickle
import streamlit as st
import numpy as np

# Load the trained KNN model
with open("knn_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📱 Smartphone Addiction Prediction App")

# Input fields
daily_usage = st.number_input("Daily Usage Hours", min_value=0.0, step=0.1)
social_media = st.number_input("Time on Social Media (hrs)", min_value=0.0, step=0.1)
gaming = st.number_input("Time on Gaming (hrs)", min_value=0.0, step=0.1)
apps_used = st.number_input("Apps Used Daily", min_value=0, step=1)
phone_checks = st.number_input("Phone Checks Per Day", min_value=0, step=1)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, step=0.1)

# Predict button
if st.button("Predict Addiction Level"):
    input_data = np.array([[daily_usage, social_media, gaming, apps_used, phone_checks, sleep_hours]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Addiction Level: {prediction[0]:.2f}")

