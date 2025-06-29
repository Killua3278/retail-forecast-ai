# Filename: app.py
# AI-Driven Retail Forecasting - Streamlit App (Production-Ready Version)

import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import streamlit as st

# -----------------------------
# 1. Simulated Data Ingestion (Mock for Deployment)
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

    metrics = {
        'MAE': round(mean_absolute_error(y_test, y_pred), 2),
        'RMSE': round(mean_squared_error(y_test, y_pred, squared=False), 2),
        'MAPE': round(mean_absolute_percentage_error(y_test, y_pred) * 100, 2)
    }
    return model, metrics

# -----------------------------
# 4. Future Week Prediction
# -----------------------------

def predict_next_week(model):
    next_week_data = pd.DataFrame({
        'visits': [170, 220],
        'avg_dwell_time': [9.2, 7.5],
        'weather_score': [0.9, 0.87],
        'social_sentiment': [0.72, 0.58],
        'week_num': [27, 27],
        'month': [6, 6],
        'store_id_B': [0, 1]
    })
    predictions = model.predict(next_week_data)
    return {'Store A': round(predictions[0]), 'Store B': round(predictions[1])}

# -----------------------------
# 5. Streamlit Frontend
# -----------------------------

def dashboard(metrics, predictions):
    st.set_page_config(page_title="Retail Forecast AI", layout="centered")
    st.title("📈 Hyperlocal Retail Sales Forecasting")

    st.markdown("""
    This AI tool predicts upcoming sales (via check-ins) for small retail stores using:
    - Weekly foot traffic
    - Dwell time
    - Local weather favorability
    - Social media sentiment
    """)

    st.header("📊 Model Evaluation")
    st.metric("Mean Absolute Error (MAE)", metrics['MAE'])
    st.metric("Root Mean Squared Error (RMSE)", metrics['RMSE'])
    st.metric("Mean Absolute Percentage Error (MAPE)", f"{metrics['MAPE']}%")

    st.header("🔮 Predicted Check-ins for Next Week")
    st.json(predictions)

    st.caption("Built by Akhil Ramesh | Powered by GPT-4")

# -----------------------------
# 6. Main Entry Point
# -----------------------------

def main():
    df = load_mock_data()
    X, y = prepare_features(df)
    model, metrics = train_forecast_model(X, y)
    predictions = predict_next_week(model)
    dashboard(metrics, predictions)

if __name__ == '__main__':
    main()
