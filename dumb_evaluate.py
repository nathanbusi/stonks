import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load preprocessed data
data = pd.read_csv('preprocessed_data/AAPL_preprocessed_data.csv')

# Ensure 'Date' column is datetime type
data["Date"] = pd.to_datetime(data["Date"])

# Filter data for evaluation range (matching your previous split)
start_date = "2021-01-01"  # Adjust this as needed
end_date = "2023-12-31"
evaluation_data = data[(data["Date"] >= start_date) & (data["Date"] <= end_date)]

# Debug: Check the evaluation dataset
print(f"Evaluation Data shape: {evaluation_data.shape}")
if evaluation_data.empty:
    raise ValueError("No data available for evaluation. Check date range or dataset.")

# Extract target prices and dates
actual_prices = evaluation_data['Adj Close'].values
dates = evaluation_data['Date'].values

# Baseline Strategies
# 1. Last Known Value (No-Change Model)
naive_predictions = np.roll(actual_prices, 1)  # Shift by 1 day
naive_predictions[0] = actual_prices[0]  # Handle first day edge case

# 2. Moving Average Model (3-day window)
moving_avg_predictions = (
    pd.Series(actual_prices).rolling(window=3).mean().shift(1).fillna(method='bfill').values
)

# Metrics Calculation
def calculate_metrics(true_values, predictions):
    mse = mean_squared_error(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    return mse, mae, r2

naive_mse, naive_mae, naive_r2 = calculate_metrics(actual_prices, naive_predictions)
moving_avg_mse, moving_avg_mae, moving_avg_r2 = calculate_metrics(actual_prices, moving_avg_predictions)

# Print Results
print("Baseline Evaluation:")
print(f"No-Change Model - MSE: {naive_mse:.4f}, MAE: {naive_mae:.4f}, R²: {naive_r2:.4f}")
print(f"Moving Average Model - MSE: {moving_avg_mse:.4f}, MAE: {moving_avg_mae:.4f}, R²: {moving_avg_r2:.4f}")

# Plot Baseline Predictions vs Actual Prices
plt.figure(figsize=(12, 6))
plt.plot(dates, actual_prices, label="Actual Prices", color="blue")
plt.plot(dates, naive_predictions, label="No-Change Model", color="orange", linestyle="--")
plt.plot(dates, moving_avg_predictions, label="3-Day Moving Average", color="green", linestyle="--")
plt.title("Baseline Predictions vs Actual Prices")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()

# Format x-axis for better readability
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show every 3rd month
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format as YYYY-MM
plt.xticks(rotation=45)

plt.grid()
plt.show()

