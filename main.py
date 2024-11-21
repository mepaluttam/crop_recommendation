import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Configure the Streamlit page
st.set_page_config(page_title="Crop Recommendation", layout="centered")

# Header
st.title("🌾 Crop Recommendation System")
st.write("Fill in the required inputs to get a recommended crop.")

# Load the saved model and label encoder
with open('pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Input fields for user
st.header("Input Parameters")

N = st.number_input("Nitrogen Content (N):", min_value=0, max_value=300, step=1)
P = st.number_input("Phosphorus Content (P):", min_value=0, max_value=300, step=1)
K = st.number_input("Potassium Content (K):", min_value=0, max_value=300, step=1)
temperature = st.slider("Temperature (°C):", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.slider("Humidity (%):", min_value=0.0, max_value=100.0, value=50.0)
ph = st.number_input("Soil pH:", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
rainfall = st.number_input("Rainfall (mm):", min_value=0.0, max_value=1000.0, step=1.0)

# Prepare input data for the pipeline
if st.button("Predict"):
    input_data = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    )

    # Apply the pipeline for prediction
    c = np.expm1(pipeline.predict(input_data))  # Reverse log transformation
    predicted_class = int(np.round(c[0]))  # Round to nearest integer

    # Decode the predicted class label
    crop_name = label_encoder.inverse_transform([predicted_class])

    # Show the result in Streamlit
    st.success(f"✅ The recommended crop is: **{crop_name[0]}**")
