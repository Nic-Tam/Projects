import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
import requests
from datetime import datetime, timedelta
from keras.models import load_model
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Set up
st.title('Stock Trend Prediction')

symbol = st.text_input('Enter Stock Ticker', 'COST')

# Calculate dynamic start and end dates
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)

# Pulling data from Yahoo Finance
df = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

# Rename columns
df = df.rename(columns={
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
})

# Ensure numeric
df = df.apply(pd.to_numeric)

# Set index name
df.index.name = 'Date'

# Reset index but keep Date column
df = df.reset_index()

# Drop 'Price' if it exists (just in case)
df = df.drop(columns=['Price'], errors='ignore')

# Display stats
st.subheader('Data from last 5 years')
st.write(df.describe())

# Visualization - Closing Price
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['close'])
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.xticks(rotation=45)
st.pyplot(fig)

# MA Visualization
st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df['close'].rolling(100).mean()
ma200 = df['close'].rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df['Date'], ma100, 'r', label='MA100')
plt.plot(df['Date'], ma200, 'g', label='MA200')
plt.plot(df['Date'], df['close'], 'b', label='Close')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# Train-Test Split
train, test = train_test_split(df[['Date', 'close']], test_size=0.30, shuffle=False)

# Fit scaler on training data
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(train[['close']])

# Load pre-trained model
model = load_model('keras_model.h5')

# Prepare test input
past_100_days = train.tail(100)
final_df = pd.concat([past_100_days, test])
final_df = final_df.reset_index(drop=True)

input_data = scaler.fit_transform(final_df[['close']])

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predict
y_predicted = model.predict(x_test)

# Scale back to original
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Use actual dates for x-axis
prediction_dates = final_df['Date'][100:]

# Plot predicted vs real
st.subheader('Predicted vs Real Stock Price')
fig2 = plt.figure(figsize=(12,6))
plt.plot(prediction_dates, y_test, 'b', label='Real Price')
plt.plot(prediction_dates, y_predicted, 'r', label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig2)

# Future Stock Price Prediction
st.subheader('Future Stock Price Prediction')

n_future_days = st.slider('How many days into the future?', min_value=1, max_value=60, value=30)

# Get the last 100 days' close prices
last_100_days = df['close'].values[-100:]
last_100_scaled = scaler.transform(last_100_days.reshape(-1, 1))

future_predictions = []
input_seq = list(last_100_scaled.flatten())  # convert to flat list

for _ in range(n_future_days):
    x_input = np.array(input_seq[-100:]).reshape(1, 100, 1)
    pred_scaled = model.predict(x_input)[0][0]
    input_seq.append(pred_scaled)
    future_predictions.append(pred_scaled)

# Rescale back to original values
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Create future dates
last_date = df['Date'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_future_days)

# Plot
fig3 = plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['close'], label='Historical Close')
plt.plot(future_dates, future_predictions, 'r--', label='Future Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
st.pyplot(fig3)
