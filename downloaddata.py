import yfinance as yf
import os


def download_stock_data(tickers, output_dir="stock_data", start_date="2015-01-01", end_date="2023-12-31"):
    """
    Downloads historical stock data for the specified tickers and saves them as CSV files.

    Args:
        tickers (list): List of stock ticker symbols to download (e.g., ['AAPL', 'MSFT']).
        output_dir (str): Directory to save the downloaded CSV files.
        start_date (str): Start date for historical data.
        end_date (str): End date for historical data.
    """
    os.makedirs(output_dir, exist_ok=True)

    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date)

        # Ensure the 'Date' column is included
        data.reset_index(inplace=True)  # Reset index to expose 'Date'

        # Save the CSV with correct format
        output_file = os.path.join(output_dir, f"{ticker.lower()}_stock_data.csv")
        data.to_csv(output_file, index=False)
        print(f"Data for {ticker} saved to {output_file}.")


if __name__ == "__main__":
    # Define tickers and download data
    tickers = ["AAPL", "MSFT"]  # Add more tickers as needed
    download_stock_data(tickers)
