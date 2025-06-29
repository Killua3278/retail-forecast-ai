import streamlit as st
import pandas as pd
import numpy as np
from yolo_integration import load_yolo_model, extract_cv_features
from PIL import Image
import requests
import io
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load and prepare training data
def load_training_data():
    data = pd.read_csv("historical_store_features.csv")
    return data

def train_forecast_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = round(mean_squared_error(y_test, y_pred, squared=False), 2)
    return model, {"RMSE": rmse}

import requests
from PIL import Image
import io

def fetch_satellite_image(coords):
    lat, lon = coords

    # Replace with your actual Mapbox token (free tier available)
    access_token = "YOUR_MAPBOX_ACCESS_TOKEN"
    zoom = 17
    width = 600
    height = 600

    url = (
        f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
        f"{lon},{lat},{zoom}/{width}x{height}?access_token={access_token}"
    )

    response = requests.get(url)

    if response.status_code != 200:
        print("URL used:", url)
        print("Response content:", response.text[:200])
        raise Exception(f"Failed to fetch image: status code {response.status_code}")

    content_type = response.headers.get('Content-Type', '')
    if not content_type.startswith('image'):
        raise Exception(f"Unexpected content type: {content_type}")

    image = Image.open(io.BytesIO(response.content))
    return image



def generate_recommendations(predicted_checkins):
    tips = []
    if predicted_checkins < 150:
        tips.append("🛍️ Consider hosting promotions to boost foot traffic.")
        tips.append("📱 Increase your store’s social media visibility.")
        tips.append("🪑 Improve outdoor seating and walkability appeal.")
    else:
        tips.append("✅ Store performance looks strong this week.")
    return tips

# Streamlit UI
st.title("📊 Retail AI Forecasting App")
st.write("Predict weekly check-ins for a retail store using computer vision, foot traffic & sentiment.")

store_name = st.text_input("Enter Store Name")
lat = st.number_input("Latitude", value=40.7128)
lon = st.number_input("Longitude", value=-74.0060)

if st.button("Predict Sales"):
    with st.spinner("Fetching data & analyzing imagery..."):
        # Load CV model
        yolo_model = load_yolo_model()

        # Fetch satellite image
        sat_image = fetch_satellite_image((lat, lon))
        st.image(sat_image, caption="Satellite View", use_column_width=True)

        # Extract visual features
        lot_fill, people = extract_cv_features(sat_image, yolo_model)

        # Load dataset and train model
        df = load_training_data()
        X = df[["visits", "avg_dwell_time", "weather_score", "social_sentiment", "lot_fill", "visible_people", "week_num", "month"]]
        y = df["check_ins"]
        model, metrics = train_forecast_model(X, y)

        # Fake social sentiment for demo
        social_sentiment = np.random.uniform(0.4, 0.9)
        weather_score = np.random.uniform(0.7, 1.0)
        visits = np.random.randint(100, 500)
        dwell_time = np.random.uniform(4.5, 9.5)
        week = pd.Timestamp.today().isocalendar()[1]
        month = pd.Timestamp.today().month

        # Predict
        input_data = [[visits, dwell_time, weather_score, social_sentiment, lot_fill, people, week, month]]
        predicted = model.predict(input_data)[0]

        # Output
        st.subheader(f"📈 Predicted Weekly Check-ins: {int(predicted)}")
        st.write("Model RMSE:", metrics["RMSE"])

        # Tips
        st.markdown("### 🧠 Recommendations")
        for tip in generate_recommendations(predicted):
            st.markdown(tip)

        st.success("Analysis complete.")
