import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.dates as mdates

# Load preprocessed data
data = pd.read_csv('preprocessed_data/AAPL_preprocessed_data.csv')

# Ensure 'Date' column is datetime type
data["Date"] = pd.to_datetime(data["Date"])

# Filter data for evaluation (unseen future data)
eval_start_date = "2020-01-01"  # Start date for evaluation data
eval_end_date = "2020-12-31"    # End date for evaluation data
data = data[(data["Date"] >= eval_start_date) & (data["Date"] <= eval_end_date)]

# Debug: Check data after filtering
print(f"Data shape after filtering for evaluation: {data.shape}")
if data.empty:
    raise ValueError("No data available for evaluation after filtering. Check date range or dataset.")

# Features and target column
features = ['Adj Close', 'Volume', 'MA7', 'MA30', 'EMA9', 'EMA50', 'Volatility', 'RSI',
            'BB_Width', 'MACD', 'MACD_Signal', 'Momentum', 'OBV', 'Daily Return']
target = 'Target'

# Extract the date column
dates = data['Date'].values  # Get all dates initially

# Scale features and target using scalers fitted on training data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Load scalers from training (assumes they were saved during training)
scaler_X.fit(pd.read_csv('preprocessed_data/AAPL_preprocessed_data.csv')[features])
scaler_y.fit(pd.read_csv('preprocessed_data/AAPL_preprocessed_data.csv')[[target]])

X = scaler_X.transform(data[features])
y = scaler_y.transform(data[[target]])

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Sequence length for LSTM
sequence_length = 50

# Prepare sequential data
def create_sequences(X, y, dates, sequence_length):
    sequences_X, sequences_y, sequence_dates = [], [], []
    for i in range(len(X) - sequence_length):
        sequences_X.append(X[i:i + sequence_length])
        sequences_y.append(y[i + sequence_length])
        sequence_dates.append(dates[i + sequence_length])  # Align dates with the output sequence
    return torch.stack(sequences_X), torch.stack(sequences_y), sequence_dates

X_seq, y_seq, aligned_dates = create_sequences(X, y, dates, sequence_length)

# Load the model
from model3_architecture import StockPredictionModel3

input_size = len(features)  # Number of input features
d_model = 128               # Model dimension (used as hidden size)
num_heads = 4               # Number of attention heads
num_layers = 2              # Number of Transformer layers
ff_dim = 512                # Dimension of feed-forward layers
output_size = 1             # Single output (e.g., next day's price)

model = StockPredictionModel3(input_size, d_model, num_heads, num_layers, output_size, ff_dim)

# Load trained model weights
model.load_state_dict(torch.load('model3.pth'))
model.eval()

# Predict on evaluation data
eval_loader = DataLoader(TensorDataset(X_seq, y_seq), batch_size=64, shuffle=False)
predictions = []
actuals = []

with torch.no_grad():
    for batch_X, batch_y in eval_loader:
        y_pred = model(batch_X)
        predictions.append(y_pred.numpy())
        actuals.append(batch_y.numpy())

# Convert predictions and actuals to numpy arrays
predictions = np.concatenate(predictions, axis=0)
actuals = np.concatenate(actuals, axis=0)

# Rescale predictions and actuals back to original scale
predictions = scaler_y.inverse_transform(predictions)
actuals = scaler_y.inverse_transform(actuals)

# Calculate metrics
mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
r2 = r2_score(actuals, predictions)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(aligned_dates, actuals, label="Actual Prices", color="blue")
plt.plot(aligned_dates, predictions, label="Predicted Prices", color="green")
plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()

# Format x-axis for better readability
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show every 3rd month
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format as YYYY-MM
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Calculate and plot rate of change (slope)
actual_slope = pd.Series(actuals.flatten()).diff().fillna(0).values
predicted_slope = pd.Series(predictions.flatten()).diff().fillna(0).values

# Smooth the slopes using a rolling average
window_size = 5  # Adjust the window size to control smoothing
smoothed_actual_slope = pd.Series(actual_slope).rolling(window=window_size).mean().fillna(0)
smoothed_predicted_slope = pd.Series(predicted_slope).rolling(window=window_size).mean().fillna(0)

# Plot smoothed rate of change
plt.figure(figsize=(12, 6))
plt.plot(aligned_dates[1:], smoothed_actual_slope[1:], label="Smoothed Actual Rate of Change", color="blue", linewidth=2)
plt.plot(aligned_dates[1:], smoothed_predicted_slope[1:], label="Smoothed Predicted Rate of Change", color="orange", linewidth=2)
plt.title("Smoothed Actual vs Predicted Rate of Change (Slope)")
plt.xlabel("Date")
plt.ylabel("Rate of Change")
plt.legend()
plt.grid()
plt.show()
