import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def get_tickers_data(ticker_list, start_date, end_date):
    """
    Download ticker data for a list of tickers.

    Args:
        ticker_list: List of ticker symbols
        start_date: Start date for data download
        end_date: End date for data download

    Returns:
        dict: Dictionary with ticker symbols as keys and dataframes as values
    """
    all_dict = {}

    for ticker in ticker_list:
        try:
            ticker_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)

            if ticker_data.empty:
                print(f"No data for ticker {ticker}")
                continue  # Skip this ticker

            all_dict[ticker] = ticker_data

        except Exception as e:
            print(f"Error downloading {ticker}: {e}")

    return all_dict
