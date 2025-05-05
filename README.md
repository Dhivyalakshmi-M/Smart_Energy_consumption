# Smart_Energy_consumption
A Streamlit-based dashboard for predicting and optimizing household energy consumption using ARIMA and LSTM models. Features include energy forecasting, anomaly detection, and power usage optimization. Built with Pandas, Scikit-learn, Keras, and Plotly for interactive visualizations.

# ⚡ Smart Energy Forecasting and Monitoring Dashboard

A Streamlit-based interactive dashboard for analyzing, forecasting, and optimizing household energy consumption using real-time data and AI models.

---

## 🔍 Features

- 📈 **Energy Usage Forecasting**  
  Predict future consumption using:
  - ARIMA (Time Series Model)
  - LSTM (Deep Learning Model)

- ⚠️ **Anomaly Detection**  
  Identify abnormal spikes or drops in energy usage.

- 🔋 **Power Usage Optimization**  
  Recommend best times to use appliances to reduce electricity bills.

- 📊 **Interactive Energy Dashboard**  
  Visualize energy metrics daily, monthly, or yearly. Filter by date range and metric.

- 🧠 **Model Evaluation Metrics**  
  MAE, RMSE, R² values shown in clean tables.

---

## 🧰 Tech Stack

- **Frontend**: Streamlit  
- **Data Handling**: Pandas, NumPy  
- **Visualization**: Matplotlib, Seaborn, Plotly  
- **Machine Learning**: Scikit-learn, ARIMA (statsmodels), LSTM (Keras + TensorFlow)

- requirements.txt
txt
Copy
Edit
streamlit
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
statsmodels
keras

---

## 📁 Dataset

- Household Power Consumption  
- File Format: `.txt`  
- Path: Update `url` in `load_data()` to your local file

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
