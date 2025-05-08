import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests

# Load the model
model = joblib.load("model.pkl")

# Fetch future weather
def get_future_weather(city_name, api_key):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city_name}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:
        temperatures, humidities, rainfalls = [], [], []
        for forecast in data['list']:
            temperatures.append(forecast['main']['temp'])
            humidities.append(forecast['main']['humidity'])
            # Rainfall might not be present in all forecasts
            rain = forecast.get('rain', {}).get('3h', 0)
            rainfalls.append(rain)

        avg_temp = np.mean(temperatures)
        avg_humidity = np.mean(humidities)
        total_rainfall = np.sum(rainfalls)  # Using sum instead of average for rainfall
        return avg_temp, avg_humidity, total_rainfall
    else:
        st.error(f"❌ Error fetching weather data: {data.get('message', 'Unknown error')}")
        return None, None, None

# Page configuration
st.set_page_config(page_title="Smart Crop Recommendation", page_icon="🌾")

# Page title
st.markdown("<h1 style='color:#00FFAA;'>🌾 Smart Crop Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("Get crop recommendations based on soil conditions and future weather forecast")

# Input section
st.header("🌍 Location Information")
city = st.text_input("Enter your City Name", placeholder="e.g., Mumbai, Delhi, Bangalore")

st.header("🧪 Soil Conditions")
col1, col2, col3 = st.columns(3)
with col1:
    n = st.number_input("Nitrogen (N)", min_value=0.0, value=90.0, step=1.0)
with col2:
    p = st.number_input("Phosphorous (P)", min_value=0.0, value=42.0, step=1.0)
with col3:
    k = st.number_input("Potassium (K)", min_value=0.0, value=43.0, step=1.0)

ph = st.slider("pH Value", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

# Prediction button
if st.button("🔍 Predict Crops", type="primary"):
    if not city:
        st.warning("Please enter a city name")
    else:
        with st.spinner("Fetching weather data and making predictions..."):
            # Get weather data
            api_key = "911927a9190c8e7343cdbf43a338268a"  # Consider moving this to secrets
            temperature, humidity, rainfall_predicted = get_future_weather(city, api_key)
            
            if temperature is not None:
                # Display weather info
                st.success(f"Weather forecast for {city}")
                weather_col1, weather_col2, weather_col3 = st.columns(3)
                with weather_col1:
                    st.metric("🌡️ Temperature", f"{temperature:.1f} °C")
                with weather_col2:
                    st.metric("💧 Humidity", f"{humidity:.1f}%")
                with weather_col3:
                    st.metric("🌧️ Rainfall", f"{rainfall_predicted:.1f} mm")
                
                # Prepare input data
                input_data = pd.DataFrame([[n, p, k, temperature, humidity, ph, rainfall_predicted]],
                                         columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
                
                # Get predictions
                try:
                    probabilities = model.predict_proba(input_data)[0]
                    top_indices = np.argsort(probabilities)[-3:][::-1]  # Get top 3 crops
                    top_crops = model.classes_[top_indices]
                    top_probs = probabilities[top_indices]
                    
                    # Display results
                    st.header("🌱 Recommended Crops")
                    
                    for i, (crop, prob) in enumerate(zip(top_crops, top_probs)):
                        # Create columns for better layout
                        col1, col2 = st.columns([1, 4])
                        
                        with col1:
                            # Display emoji based on rank
                            if i == 0:
                                st.markdown(f"<h3>🥇</h3>", unsafe_allow_html=True)
                            elif i == 1:
                                st.markdown(f"<h3>🥈</h3>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<h3>🥉</h3>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"<h3 style='color:#00FFAA'>{crop}</h3>", unsafe_allow_html=True)
                            st.progress(int(prob * 100))
                            st.caption(f"{prob * 100:.1f}% confidence")
                
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

# Add some info at the bottom
st.markdown("---")
st.info("ℹ️ This system recommends crops based on soil nutrients (N, P, K), pH level, and 5-day weather forecast for your location.")