# app.py
import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --- Load the KNN model safely ---
model_path = os.path.join(os.path.dirname(__file__), "knn_model.pkl")

try:
    with open(model_path, "rb") as f:
        knn_model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: knn_model.pkl file not found!")
    st.stop()

st.title("KNN Classifier Demo")

# --- Example input fields ---
st.write("Enter features to predict:")
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")
# Add more features as per your model

input_data = np.array([[feature1, feature2, feature3]])

# --- Optionally scale input if your model was trained on scaled data ---
# scaler = StandardScaler()
# input_data = scaler.fit_transform(input_data)  # Only if you need scaling

# --- Make prediction ---
if st.button("Predict"):
    prediction = knn_model.predict(input_data)
    st.success(f"Predicted Class: {prediction[0]}")
