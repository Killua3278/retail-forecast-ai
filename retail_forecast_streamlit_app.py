# Filename: multimodal_forecasting_app.py
# Full AI-Powered Retail Forecast System with Satellite + Mobility + Social Media Integration + Improvement Recommendations

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import math
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Mock Data Fetching Functions (Replace with APIs if available)
# -----------------------------

def fetch_satellite_image(store_coords):
    return Image.open("sample_satellite.jpg")  # must include this sample in your repo

def fetch_social_sentiment(store_name):
    sentiment_score = np.random.uniform(0.3, 0.9)
    return round(sentiment_score, 2)

def fetch_foot_traffic(store_name):
    visits = np.random.randint(100, 400)
    dwell_time = round(np.random.uniform(5.0, 12.0), 1)
    return visits, dwell_time

# -----------------------------
# 2. Vision Model: Parking Lot Detection / People Density (Mocked)
# -----------------------------

def extract_cv_features(image):
    percent_lot_filled = round(np.random.uniform(0.2, 0.95), 2)
    people_detected = np.random.randint(0, 50)
    return percent_lot_filled, people_detected

# -----------------------------
# 3. Load Sample Historical Dataset
# -----------------------------

def load_training_data():
    data = pd.read_csv("historical_store_features.csv")  # must include this CSV in your repo
    return data

# -----------------------------
# 4. Train Model
# -----------------------------

def train_model(df):
    X = df.drop(columns=["check_ins", "store_id"])
    y = df["check_ins"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    metrics = {
        "MAE": round(mean_absolute_error(y_test, y_pred), 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mean_absolute_percentage_error(y_test, y_pred) * 100, 2)
    }
    return model, metrics

# -----------------------------
# 5. UI Input: Store Location and Name
# -----------------------------

def get_store_info():
    st.sidebar.header("📍 Store Info")
    store_name = st.sidebar.text_input("Store Name", value="Example Cafe")
    lat = st.sidebar.number_input("Latitude", value=40.7128)
    lon = st.sidebar.number_input("Longitude", value=-74.0060)
    return store_name, (lat, lon)

# -----------------------------
# 6. Generate Recommendations
# -----------------------------

def generate_recommendations(features):
    suggestions = []
    if features["social_sentiment"].iloc[0] < 0.5:
        suggestions.append("⬆️ Increase online engagement with positive campaigns and influencer posts.")
    if features["lot_fill"].iloc[0] < 0.5:
        suggestions.append("🚗 Improve signage or parking availability to increase convenience.")
    if features["avg_dwell_time"].iloc[0] < 6.0:
        suggestions.append("☕ Add in-store attractions or promotions to increase customer stay time.")
    if features["visits"].iloc[0] < 200:
        suggestions.append("📣 Launch a local marketing push to boost foot traffic (flyers, loyalty apps).")
    if not suggestions:
        suggestions.append("✅ Store is performing well! Keep optimizing social and physical visibility.")
    return suggestions

# -----------------------------
# 7. Make Prediction
# -----------------------------

def forecast_checkins(model, store_coords, store_name):
    image = fetch_satellite_image(store_coords)
    percent_lot, num_people = extract_cv_features(image)
    visits, dwell_time = fetch_foot_traffic(store_name)
    sentiment = fetch_social_sentiment(store_name)

    now = datetime.datetime.now()
    week = now.isocalendar()[1]
    month = now.month

    features = pd.DataFrame({
        "visits": [visits],
        "avg_dwell_time": [dwell_time],
        "weather_score": [0.85],
        "social_sentiment": [sentiment],
        "lot_fill": [percent_lot],
        "visible_people": [num_people],
        "week_num": [week],
        "month": [month]
    })

    pred = model.predict(features)[0]
    recs = generate_recommendations(features)
    return int(pred), features, image, recs

# -----------------------------
# 8. Streamlit Frontend
# -----------------------------

def render_dashboard(metrics, prediction, features, image, recs):
    st.set_page_config(page_title="📊 AI Retail Forecaster", layout="wide")
    st.title("🌐 AI-Powered Retail Forecasting System")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔮 Predicted Weekly Check-ins")
        st.success(f"Expected Check-ins: {prediction}")
        st.subheader("📊 Input Features")
        st.dataframe(features)
    with col2:
        st.subheader("🛰️ Satellite View")
        st.image(image, caption="Simulated Satellite Snapshot")

    st.markdown("---")
    st.subheader("📈 Model Performance")
    st.metric("MAE", metrics['MAE'])
    st.metric("RMSE", metrics['RMSE'])
    st.metric("MAPE", f"{metrics['MAPE']}%")

    st.markdown("---")
    st.subheader("💡 Recommendations to Improve Performance")
    for rec in recs:
        st.info(rec)

    st.caption("Built by Akhil Ramesh | Streamlit + PyTorch + Geospatial Intelligence")

# -----------------------------
# 9. Main Logic
# -----------------------------

def main():
    store_name, coords = get_store_info()
    df = load_training_data()
    model, metrics = train_model(df)
    prediction, features, image, recs = forecast_checkins(model, coords, store_name)
    render_dashboard(metrics, prediction, features, image, recs)

if __name__ == '__main__':
    main()
