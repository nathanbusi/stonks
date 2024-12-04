import tkinter as tk
from tkinter import messagebox
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from model3_architecture import StockPredictionModel3
import numpy as np

class StockPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Predictor")
        self.root.geometry("600x400")

        self.setup_gui()

        # Load preprocessed data and scalers
        self.data = pd.read_csv('preprocessed_data/AAPL_preprocessed_data.csv')
        self.data["Date"] = pd.to_datetime(self.data["Date"])
        self.features = ['Adj Close', 'Volume', 'MA7', 'MA30', 'EMA9', 'EMA50', 'Volatility', 'RSI',
                         'BB_Width', 'MACD', 'MACD_Signal', 'Momentum', 'OBV', 'Daily Return']
        self.target = 'Target'

        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.scaler_X.fit(self.data[self.features])
        self.scaler_y.fit(self.data[[self.target]])

        # Load the trained model
        input_size = len(self.features)
        d_model = 128
        num_heads = 4
        num_layers = 2
        ff_dim = 512
        output_size = 1

        self.model = StockPredictionModel3(input_size, d_model, num_heads, num_layers, output_size, ff_dim)
        self.model.load_state_dict(torch.load('model3.pth'))
        self.model.eval()

    def setup_gui(self):
        self.label = tk.Label(self.root, text="Pick a date between 2021-01-01 and 2023-12-31:",
                              font=("Helvetica", 14))
        self.label.pack(pady=20)

        self.date_entry = tk.Entry(self.root, font=("Helvetica", 14))
        self.date_entry.pack(pady=10)

        self.predict_button = tk.Button(self.root, text="Predict", font=("Helvetica", 14),
                                        command=self.predict_stock_price)
        self.predict_button.pack(pady=10)

        self.result_label = tk.Label(self.root, text="", font=("Helvetica", 16))
        self.result_label.pack(pady=20)

    def predict_stock_price(self):
        input_date = self.date_entry.get()

        try:
            input_date = pd.to_datetime(input_date)
        except ValueError:
            self.result_label.config(text="Invalid date format! Please use YYYY-MM-DD.", fg="red")
            return

        if input_date < pd.to_datetime("2021-01-01") or input_date > pd.to_datetime("2023-12-31"):
            self.result_label.config(text="Date out of range! Pick a date within the evaluation period.", fg="red")
            return

        # Check if date exists in dataset
        if input_date not in self.data["Date"].values:
            self.result_label.config(text="No data available for this date.", fg="red")
            return

        # Get the actual stock price for the selected date
        actual_price = self.data.loc[self.data["Date"] == input_date, "Adj Close"].values[0]

        # Get the last `sequence_length` days of features
        start_idx = self.data[self.data["Date"] == input_date].index[0] - 50
        if start_idx < 0:
            self.result_label.config(text="Not enough prior data to predict for this date.", fg="red")
            return

        eval_data = self.data.iloc[start_idx:start_idx + 50]
        input_data = eval_data[self.features].values
        input_data = self.scaler_X.transform(input_data)
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Shape: (1, sequence_length, input_size)

        # Make prediction
        with torch.no_grad():
            prediction = self.model(input_tensor)
            predicted_price = self.scaler_y.inverse_transform(prediction.numpy())[0, 0]

        # Display both predicted and actual stock prices
        self.result_label.config(
            text=f"Predicted Stock Price: ${predicted_price:.2f}\nActual Stock Price: ${actual_price:.2f}",
            fg="green"
        )


# Main Application
if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictionGUI(root)
    root.mainloop()
