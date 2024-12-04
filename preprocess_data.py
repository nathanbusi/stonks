import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def preprocess_stock_data(input_file, output_dir="preprocessed_data"):
    try:
        raw_data = pd.read_csv(input_file)
        stock_name = os.path.basename(input_file).split("_")[0].upper()

        print(f"Processing {input_file}...")

        # Ensure the 'Date' column exists
        if "Date" not in raw_data.columns:
            raise ValueError(f"'Date' column is missing in {input_file}. Please check the file format.")

        raw_data["Date"] = pd.to_datetime(raw_data["Date"])
        raw_data.set_index("Date", inplace=True)

        # Ensure numeric columns are properly converted
        numeric_columns = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
        raw_data[numeric_columns] = raw_data[numeric_columns].apply(pd.to_numeric, errors="coerce")
        raw_data = raw_data.dropna()

        # Add Moving Averages
        raw_data["MA7"] = raw_data["Adj Close"].rolling(window=7).mean()
        raw_data["MA30"] = raw_data["Adj Close"].rolling(window=30).mean()

        # Add Exponential Moving Averages (EMAs)
        raw_data["EMA9"] = raw_data["Adj Close"].ewm(span=9, adjust=False).mean()
        raw_data["EMA50"] = raw_data["Adj Close"].ewm(span=50, adjust=False).mean()

        # Add Volatility
        raw_data["Volatility"] = raw_data["High"] - raw_data["Low"]

        # Add RSI
        delta = raw_data["Adj Close"].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        raw_data["RSI"] = 100 - (100 / (1 + rs))

        # Add Bollinger Bands
        raw_data["BB_Mid"] = raw_data["Adj Close"].rolling(window=20).mean()
        raw_data["BB_Upper"] = raw_data["BB_Mid"] + 2 * raw_data["Adj Close"].rolling(window=20).std()
        raw_data["BB_Lower"] = raw_data["BB_Mid"] - 2 * raw_data["Adj Close"].rolling(window=20).std()
        raw_data["BB_Width"] = raw_data["BB_Upper"] - raw_data["BB_Lower"]

        # Add MACD
        fast_ema = raw_data["Adj Close"].ewm(span=12, adjust=False).mean()
        slow_ema = raw_data["Adj Close"].ewm(span=26, adjust=False).mean()
        raw_data["MACD"] = fast_ema - slow_ema
        raw_data["MACD_Signal"] = raw_data["MACD"].ewm(span=9, adjust=False).mean()

        # Add Momentum
        raw_data["Momentum"] = raw_data["Adj Close"] - raw_data["Adj Close"].shift(10)

        # Add OBV
        raw_data["OBV"] = (np.where(raw_data["Adj Close"] > raw_data["Adj Close"].shift(1),
                                    raw_data["Volume"],
                                    np.where(raw_data["Adj Close"] < raw_data["Adj Close"].shift(1),
                                             -raw_data["Volume"], 0))).cumsum()

        # Add Daily Returns
        raw_data["Daily Return"] = raw_data["Adj Close"].pct_change()

        # Add Target
        raw_data["Target"] = raw_data["Adj Close"].shift(-1)

        # Drop rows with NaN values introduced by rolling/shifting
        raw_data = raw_data.dropna()

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{stock_name}_preprocessed_data.csv")
        raw_data.to_csv(output_file)
        print(f"Preprocessed data for {stock_name} saved to {output_file}.")

        # Visualization
        plt.figure(figsize=(12, 6))
        plt.plot(raw_data["Adj Close"], label="Adj Close", color="blue", linewidth=1)
        plt.plot(raw_data["MA7"], label="7-Day MA", color="orange", linewidth=1)
        plt.plot(raw_data["MA30"], label="30-Day MA", color="green", linewidth=1)
        plt.fill_between(raw_data.index, raw_data["BB_Lower"], raw_data["BB_Upper"], color="gray", alpha=0.3, label="Bollinger Bands")
        plt.title(f"{stock_name} Stock Price with Averages and Bollinger Bands")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid()
        visualization_file = os.path.join(output_dir, f"{stock_name}_visualization.png")
        plt.savefig(visualization_file)
        plt.close()
        print(f"Visualization saved to {visualization_file}.")

        return output_file

    except Exception as e:
        print(f"An error occurred during preprocessing of {input_file}: {e}")
        return None

if __name__ == "__main__":
    input_dir = "stock_data"
    output_dir = "preprocessed_data"

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            input_file_path = os.path.join(input_dir, file_name)
            preprocess_stock_data(input_file_path, output_dir)
