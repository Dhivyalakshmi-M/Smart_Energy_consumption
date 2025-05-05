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
  Recommend the best times to use appliances to reduce electricity bills.

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

---
## 🚀 How to Run

### 1. Install Dependencies

Clone the repository and install the required dependencies using `pip`:

```bash
git clone https://github.com/your-username/smart-energy-consumption.git
cd smart-energy-consumption
pip install -r requirements.txt
2. Run the App
To run the Streamlit app locally, use the following command:

bash
Copy
Edit
streamlit run app.py
This will open the app in your default web browser.

3. Deploy to Streamlit Cloud
For deploying your app online using Streamlit Cloud:

Push your repository to GitHub.

Visit Streamlit Cloud and sign in.

Click on "New app" and connect your GitHub repository.

Follow the on-screen instructions to deploy.

🧑‍💻 How to Use
Upload Dataset:
Use the file uploader in the sidebar to upload the household_power_consumption.txt file.

Visualize Data:
Use the dropdown to select the energy consumption metric you want to visualize (e.g., Global Active Power, Voltage, etc.).

Energy Forecasting:
Select between ARIMA or LSTM models to forecast future energy consumption.

Anomaly Detection:
View any identified anomalies in energy usage, which may indicate unusual spikes or drops.

Optimization Recommendations:
Get suggestions on the best times to use energy-intensive appliances based on historical data.

🧰 Tech Stack
Frontend: Streamlit

Data Handling: Pandas, NumPy

Visualization: Matplotlib, Seaborn, Plotly

Machine Learning: Scikit-learn, ARIMA (statsmodels), LSTM (Keras + TensorFlow)

📄 Example of Output
Energy Usage Forecasting (ARIMA / LSTM)
ARIMA: Shows the forecasted consumption for the next 7 days based on past data.

LSTM: Provides a deep learning model's prediction over a longer period, with advanced features like error metrics.

Anomaly Detection
Spikes and Dips: Identifies any abnormal usage patterns that may suggest issues such as faulty appliances or energy waste.

Power Usage Optimization
Recommendations: Suggestions on optimizing appliance usage based on past consumption trends.

📦 Requirements
Make sure to install the required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
requirements.txt:

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
💡 Contributions
Feel free to fork the repository, submit issues, and pull requests. Contributions are always welcome!

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
