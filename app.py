import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt

# Define the Model Class
class BitcoinLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(BitcoinLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Load Resources 
@st.cache_resource
def load_resources():
    # Load Scaler
    scaler = joblib.load('scaler.pkl')
    
    # Load Model
    model = BitcoinLSTM()
    model.load_state_dict(torch.load('bitcoin_lstm_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model, scaler

model, scaler = load_resources()

# UI Layout 
st.title("₿ Bitcoin Price Predictor (LSTM)")
st.write("This app uses a Deep Learning (LSTM) model trained on historical Bitcoin data to predict future prices.")

# Sidebar for inputs
st.sidebar.header("User Input")
input_option = st.sidebar.radio("Choose Input Method:", ["Use Random Test Data", "Enter Custom Data"])

input_data = []

if input_option == "Use Random Test Data":
    if st.sidebar.button("Generate Random Sequence"):
        # Generate random dummy data behaving like stock (random walk)
        start_price = 40000
        volatility = 0.02
        days = 60
        prices = [start_price]
        for _ in range(days):
            change = np.random.normal(0, volatility)
            prices.append(prices[-1] * (1 + change))
        input_data = prices[1:] # Keep last 60
        st.session_state['custom_data'] = input_data # Save to session
    elif 'custom_data' in st.session_state:
        input_data = st.session_state['custom_data']

else:
    # Custom Input Box
    st.sidebar.write("Enter 60 closing prices separated by commas:")
    user_text = st.sidebar.text_area("Paste data here", height=150)
    if user_text:
        try:
            input_data = [float(x.strip()) for x in user_text.split(',')]
            if len(input_data) != 60:
                st.sidebar.error(f"Expected 60 data points, got {len(input_data)}.")
                input_data = []
        except ValueError:
            st.sidebar.error("Invalid format. Please enter numbers separated by commas.")

# --- 4. Prediction Logic ---
if len(input_data) == 60:
    st.subheader("1. Input Data Visualization")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(input_data, label='Past 60 Days', color='blue')
    ax.set_title("Price Trend (Input)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    st.pyplot(fig)

    # Prepare data
    input_array = np.array(input_data).reshape(-1, 1)
    scaled_input = scaler.transform(input_array)
    input_tensor = torch.from_numpy(scaled_input).float().unsqueeze(0) # Shape: (1, 60, 1)

    # Predict Next Day
    with torch.no_grad():
        prediction = model(input_tensor)
    
    predicted_price = scaler.inverse_transform(prediction.numpy())[0][0]

    st.subheader("2. Prediction")
    st.metric(label="Predicted Price for Next Day", value=f"${predicted_price:,.2f}")

    # Future Forecast (30 Days)
    st.subheader("3. Future Forecast (Next 30 Days)")
    future_days = 30
    current_seq = input_tensor
    future_preds = []

    for _ in range(future_days):
        with torch.no_grad():
            pred = model(current_seq)
            future_preds.append(pred.item())
            new_pred_reshaped = pred.unsqueeze(1)
            current_seq = torch.cat((current_seq[:, 1:, :], new_pred_reshaped), dim=1)

    future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

    # Plot Future
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(range(60), input_data, label='History', color='blue')
    ax2.plot(range(60, 60+future_days), future_prices, label='Forecast', color='green', linestyle='--')
    ax2.set_title("30-Day Future Forecast")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

else:
    st.info("Please generate data or enter 60 values to see predictions.")