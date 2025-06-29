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

    # NASA Earth imagery API (free and doesn't require billing)
    nasa_api_key = "R94OuSX31TsYywYbM2mTgQ4E4Op1WMfTLKbyrI53"  # ✅ correct You can get your own free key at https://api.nasa.gov
    metadata_url = (
        f"https://api.nasa.gov/planetary/earth/assets"
        f"?lon={lon}&lat={lat}&dim=0.1&api_key={nasa_api_key}"
    )

    meta_response = requests.get(metadata_url)
    if meta_response.status_code != 200:
        print("Metadata request failed:", meta_response.text)
        raise Exception(f"Failed to fetch metadata: status code {meta_response.status_code}")

    image_url = meta_response.json().get("url")
    if not image_url:
        raise Exception("No image URL found in NASA API response")

    image_response = requests.get(image_url)
    if image_response.status_code != 200:
        print("Image request failed:", image_response.text)
        raise Exception(f"Failed to download image: status code {image_response.status_code}")

    image = Image.open(io.BytesIO(image_response.content))
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
