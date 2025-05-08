import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("model.pkl")

# App Configuration
st.set_page_config(page_title="🌾 Crop Recommendation", page_icon="🌱")
st.title("🌾 Crop Recommendation System")
st.markdown("Enter soil and climate values to get the best crop suggestion for your land.")

# Input fields with realistic defaults
n = st.number_input("🧪 Nitrogen (N)", min_value=0.0, value=90.0, step=1.0)
p = st.number_input("🧪 Phosphorous (P)", min_value=0.0, value=42.0, step=1.0)
k = st.number_input("🧪 Potassium (K)", min_value=0.0, value=43.0, step=1.0)
temperature = st.number_input("🌡️ Temperature (°C)", value=25.0, step=0.1)
humidity = st.number_input("💧 Humidity (%)", value=80.0, step=0.1)
ph = st.number_input("⚖️ pH Value", value=6.5, step=0.01)
rainfall = st.number_input("🌧️ Rainfall (mm)", value=100.0, step=0.1)

# Predict button
if st.button("🔍 Predict Crop"):
    if all(val == 0 for val in [n, p, k, temperature, humidity, ph, rainfall]):
        st.warning("⚠️ All input values are 0. Please enter realistic values.")
    else:
        try:
            # Create DataFrame with proper column names
            features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            input_data = pd.DataFrame([[n, p, k, temperature, humidity, ph, rainfall]], columns=features)

            # Make prediction
            prediction = model.predict(input_data)[0]
            st.success(f"🌱 Recommended Crop: **{prediction}**")
        except Exception as e:
            st.error(f"❌ Error: {e}")
