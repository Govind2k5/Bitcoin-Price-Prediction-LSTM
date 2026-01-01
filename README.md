# ₿ Bitcoin Price Predictor using LSTM

A Deep Learning project that predicts Bitcoin prices using a **Long Short-Term Memory (LSTM)** neural network built with **PyTorch**. This project features an interactive web application built with **Streamlit** that allows users to visualize historical trends, generate forecasts, and test custom input data.

## 🚀 Features

* **Deep Learning Model:** Custom LSTM architecture trained on historical Bitcoin data (OHLCV).
* **Price Prediction:** Predicts the next-day closing price based on the past 60 days of data.
* **Future Forecasting:** Generates a 30-day future price trend forecast.
* **Interactive Web UI:** User-friendly interface built with Streamlit to interact with the model.
* **Custom Inputs:** Users can input their own 60-day price sequences or generate random test data to see how the model reacts.
* **Data Visualization:** Interactive charts for historical data, model training loss, and prediction vs. actual comparisons.

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Deep Learning:** PyTorch (LSTM, Linear Layers)
* **Web Framework:** Streamlit
* **Data Processing:** Pandas, NumPy, Scikit-learn (MinMaxScaler)
* **Visualization:** Matplotlib

## 📂 Project Structure

```text
├── app.py                   # The main Streamlit web application
├── bitcoin_lstm_model.pth   # The trained PyTorch model file
├── scaler.pkl               # The fitted MinMaxScaler (for inverse scaling)
├── requirements.txt         # List of dependencies
├── bitcoin_data.csv         # (Optional) Dataset used for training
└── README.md                # Project documentation
```
## ⚙️ Installation & Usage
1. Clone the Repository
   ```git clone [https://github.com/YOUR_USERNAME/Bitcoin-Price-Prediction-LSTM.git](https://github.com/YOUR_USERNAME/Bitcoin-Price-Prediction-LSTM.git)
   cd Bitcoin-Price-Prediction-LSTM
   ```
2. Install Dependencies
   ```pip install -r requirements.txt```
3. Run the Application
   ```streamlit run app.py```

## 📊 Model Performance
The model was trained for 100 epochs using the Adam optimizer and Mean Squared Error (MSE) loss.
Training Loss: Converged to < 0.001
Validation: The model successfully captures general market trends without significant overfitting
