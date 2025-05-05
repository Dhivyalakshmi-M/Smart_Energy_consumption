import streamlit as st
import pandas as pd
import numpy as np
try:
    import seaborn as sns
except ImportError:
    import os
    os.system('pip install seaborn')  # Installs seaborn if not already installed
    import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import timedelta
try:
    import matplotlib.pyplot as plt
except ImportError:
    import os
    os.system('pip install matplotlib')  # Installs matplotlib if not already installed
    import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    url = 'https://drive.google.com/file/d/1wUHpMb0D_PJ1Fl3-mW2OcSTj6Q-eGTbs/view?usp=drive_link'
    df = pd.read_csv(url, sep=';', parse_dates={'datetime': ['Date', 'Time']},
                     infer_datetime_format=True, na_values=['?'], low_memory=False)
    df.dropna(inplace=True)
    df['Global_active_power'] = df['Global_active_power'].astype(float)
    df.set_index('datetime', inplace=True)
    return df

# Feature Engineering (adding additional features)
def add_features(data):
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    return data

# Evaluate model performance
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

# Display metrics in a beautiful table format
def display_metrics(mae, mse, rmse, r2):
    metrics_data = {
        'Metric': ['MAE', 'MSE', 'RMSE', 'R²'],
        'Value': [f'{mae:.2f}', f'{mse:.2f}', f'{rmse:.2f}', f'{r2:.2f}']
    }
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df.style.set_table_styles(
        [{'selector': 'thead th', 'props': [('background-color', '#2E3B55'), ('color', 'white')]},
         {'selector': 'tbody td', 'props': [('background-color', '#f5f5f5'), ('color', '#2E3B55')]}]
    ))

# Energy Usage Prediction with ARIMA
def arima_forecast(data):
    st.subheader("📉 ARIMA Forecast (Next 30 Days)")
    daily = data['Global_active_power'].resample('D').mean().dropna()
    model = ARIMA(daily, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(30)
    combined = pd.concat([daily[-60:], forecast])
    
    st.line_chart(combined)

    # Performance evaluation
    mae, mse, rmse, r2 = evaluate_model(daily[-30:], forecast)
    display_metrics(mae, mse, rmse, r2)

# LSTM Energy Usage Prediction
def lstm_forecast(data):
    st.subheader("🔮 LSTM Forecast (Next 30 Days)")
    df = data['Global_active_power'].resample('D').mean().dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values.reshape(-1, 1))

    def create_seq(data, n_steps):
        X, y = [], []
        for i in range(n_steps, len(data)):
            X.append(data[i - n_steps:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    X, y = create_seq(scaled, 10)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(10, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, verbose=0)

    input_seq = scaled[-10:]
    predictions = []
    for _ in range(30):
        pred = model.predict(input_seq.reshape(1, 10, 1), verbose=0)
        predictions.append(pred[0][0])
        input_seq = np.append(input_seq[1:], pred).reshape(10, 1)

    future = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)
    future_series = pd.Series(future.flatten(), index=future_dates)

    st.line_chart(pd.concat([df[-60:], future_series]))

    # Performance evaluation
    mae, mse, rmse, r2 = evaluate_model(df[-30:], future.flatten())
    display_metrics(mae, mse, rmse, r2)

# Anomaly Detection (outlier detection for energy spikes or drops)
def detect_anomalies(data):
    st.subheader("⚠️ Anomaly Detection (Spikes/Drops in Energy Usage)")
    daily = data['Global_active_power'].resample('D').mean().dropna()
    z_scores = (daily - daily.mean()) / daily.std()
    anomalies = daily[z_scores.abs() > 3]  # Outliers beyond 3 standard deviations
    st.write(f"Anomalies Detected: {len(anomalies)}")
    st.line_chart(daily)
    st.markdown(f"Anomalies: {anomalies}")

# Power Usage Optimization (suggest optimal times to use power)
def power_usage_optimization(data):
    st.subheader("🔋 Power Usage Optimization (Reducing Electricity Bills)")
    daily = data['Global_active_power'].resample('D').mean().dropna()
    peak_hours = daily.idxmax()  # Find peak usage day
    off_peak_hours = daily.idxmin()  # Find off-peak usage day

    st.write(f"Peak Usage Time: {peak_hours.strftime('%Y-%m-%d')}")
    st.write(f"Off-Peak Usage Time: {off_peak_hours.strftime('%Y-%m-%d')}")

def energy_consumption_dashboard(data):
    st.subheader("📊 Energy Consumption Dashboard")

    # Convert datetime index to Unix timestamps for the slider
    min_date = data.index.min()
    max_date = data.index.max()
    min_timestamp = int(min_date.timestamp())
    max_timestamp = int(max_date.timestamp())

    # Slider for selecting date range using Unix timestamps
    start_date_timestamp = st.slider(
        'Start Date:',
        min_value=min_timestamp,
        max_value=max_timestamp,
        value=int(min_date.timestamp())
    )
    end_date_timestamp = st.slider(
        'End Date:',
        min_value=min_timestamp,
        max_value=max_timestamp,
        value=int(max_date.timestamp())
    )

    # Convert the selected Unix timestamps back to datetime
    start_date = pd.to_datetime(start_date_timestamp, unit='s')
    end_date = pd.to_datetime(end_date_timestamp, unit='s')

    filtered_data = data[(data.index >= start_date) & (data.index <= end_date)]

    # Dropdown for selecting energy metric
    metric = st.selectbox("Select Energy Metric", ['Global_active_power', 'Global_reactive_power', 'Voltage'])
    st.write(f"Displaying {metric} for the selected range")

    # Plot selected metric
    st.line_chart(filtered_data[metric])

    # Daily, Monthly, Yearly Aggregation
    daily = filtered_data[metric].resample('D').mean().dropna()
    monthly = filtered_data[metric].resample('M').mean().dropna()
    yearly = filtered_data[metric].resample('Y').mean().dropna()

    # Buttons to view different aggregation views
    view_type = st.radio("View Aggregation", ('Daily', 'Monthly', 'Yearly'))
    
    if view_type == 'Daily':
        st.line_chart(daily)
    elif view_type == 'Monthly':
        st.line_chart(monthly)
    else:
        st.line_chart(yearly)


# Main app
def main():
    # Page configuration and style
    st.set_page_config(page_title="Smart Energy Management", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
        <style>
            .title { font-size: 36px; color: #2E3B55; text-align: center; font-weight: bold; }
            .subheader { font-size: 24px; color: #3A539B; font-weight: 600; }
            .st-TextInput, .st-selectbox, .st-button { background-color: #f5f5f5; border-radius: 8px; }
            .st-TextInput input { padding-left: 12px; }
            .stMarkdown { font-size: 18px; color: #4A4A4A; }
            /* Sidebar customization */
            .css-1lcbmhc {
                background-color: #2E3B55 !important;
                color: white !important;
                font-weight: bold;
            }
            .css-ffhzg2 {
                font-weight: bold !important;
                color: white !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("⚡ Smart Energy Forecasting and Monitoring Dashboard")

    data = load_data()
    data = add_features(data)

    st.sidebar.title("📊 Choose Analysis")
    choice = st.sidebar.radio("Select:", [
        "View Raw Data", 
        "Energy Usage Prediction (ARIMA)", 
        "Energy Usage Prediction (LSTM)",
        "Anomaly Detection", 
        "Power Usage Optimization",
        "Energy Consumption Dashboard"
    ])

    if choice == "View Raw Data":
        st.subheader("📂 Raw Data Sample")
        st.dataframe(data.head(100))
    elif choice == "Energy Usage Prediction (ARIMA)":
        arima_forecast(data)
    elif choice == "Energy Usage Prediction (LSTM)":
        lstm_forecast(data)
    elif choice == "Anomaly Detection":
        detect_anomalies(data)
    elif choice == "Power Usage Optimization":
        power_usage_optimization(data)
    elif choice == "Energy Consumption Dashboard":
        energy_consumption_dashboard(data)

# Run the app
main()
