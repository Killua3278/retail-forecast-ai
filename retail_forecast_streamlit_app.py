# Filename: interactive_app.py
# AI-Driven Retail Forecasting with Interactive Input (Streamlit - Production Build)

import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import streamlit as st
import math

# -----------------------------
# 1. Simulated Data Ingestion
# -----------------------------

def load_mock_data():
    data = {
        'store_id': ['A', 'A', 'A', 'B', 'B', 'B'],
        'week': pd.date_range('2024-01-01', periods=3, freq='W').tolist() * 2,
        'visits': [130, 145, 160, 200, 180, 210],
        'avg_dwell_time': [8.2, 8.4, 9.0, 6.5, 6.8, 7.2],
        'weather_score': [0.9, 0.8, 0.95, 0.85, 0.88, 0.9],
        'social_sentiment': [0.6, 0.65, 0.7, 0.4, 0.45, 0.55],
        'check_ins': [120, 140, 155, 180, 175, 205]
    }
    return pd.DataFrame(data)

# -----------------------------
# 2. Feature Engineering
# -----------------------------

def prepare_features(df):
    df['week_num'] = df['week'].dt.isocalendar().week
    df['month'] = df['week'].dt.month
    df = pd.get_dummies(df, columns=['store_id'], drop_first=True)
    X = df.drop(columns=['week', 'check_ins'])
    y = df['check_ins']
    return X, y

# -----------------------------
# 3. Model Training
# -----------------------------

def train_forecast_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = math.sqrt(mean_squared_error(y_test, y_pred))

    metrics = {
        'MAE': round(mean_absolute_error(y_test, y_pred), 2),
        'RMSE': round(rmse, 2),
        'MAPE': round(mean_absolute_percentage_error(y_test, y_pred) * 100, 2)
    }
    return model, metrics

# -----------------------------
# 4. Interactive User Input
# -----------------------------

def get_user_input():
    st.sidebar.header("🧾 Input: Your Retail Location")

    store_name = st.sidebar.selectbox("Store ID", ["A", "B"])
    visits = st.sidebar.slider("Weekly Foot Traffic", min_value=50, max_value=300, value=150)
    dwell_time = st.sidebar.slider("Avg Dwell Time (minutes)", min_value=2.0, max_value=15.0, value=7.0)
    weather_score = st.sidebar.slider("Weather Score", min_value=0.0, max_value=1.0, value=0.85)
    sentiment = st.sidebar.slider("Social Media Sentiment", min_value=0.0, max_value=1.0, value=0.6)

    today = datetime.date.today()
    week_num = today.isocalendar()[1]
    month = today.month

    input_data = pd.DataFrame({
        'visits': [visits],
        'avg_dwell_time': [dwell_time],
        'weather_score': [weather_score],
        'social_sentiment': [sentiment],
        'week_num': [week_num],
        'month': [month],
        'store_id_B': [1 if store_name == 'B' else 0]
    })
    return store_name, input_data

# -----------------------------
# 5. Streamlit Frontend
# -----------------------------

def dashboard(metrics, prediction, store_name):
    st.set_page_config(page_title="Retail Forecast AI (Interactive)", layout="centered")
    st.title("📈 Hyperlocal Retail Sales Forecasting - Interactive Edition")

    st.markdown("""
    This AI tool predicts **next week's sales** (via check-ins) for your retail store using:
    - Weekly foot traffic
    - Dwell time
    - Weather impact
    - Social sentiment data
    
    Enter your own values in the sidebar!
    """)

    st.header("📊 Model Performance")
    st.metric("MAE", metrics['MAE'])
    st.metric("RMSE", metrics['RMSE'])
    st.metric("MAPE", f"{metrics['MAPE']}%")

    st.header(f"🔮 Forecasted Check-ins for Store {store_name}")
    st.success(f"➡️ Expected Check-ins: {int(prediction)}")

    st.caption("Built by Akhil Ramesh | Powered by GPT-4")

# -----------------------------
# 6. Main App Logic
# -----------------------------

def main():
    df = load_mock_data()
    X, y = prepare_features(df)
    model, metrics = train_forecast_model(X, y)

    store_name, user_input = get_user_input()
    prediction = model.predict(user_input)[0]
    dashboard(metrics, prediction, store_name)

if __name__ == '__main__':
    main()
