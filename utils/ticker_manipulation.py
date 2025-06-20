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


def plot_tickers_values(all_dict, figsize=(30, 10)):
    """
    Plot the closing prices for all tickers.

    Args:
        all_dict: Dictionary with ticker data (from get_tickers_data function)
        figsize: Tuple for figure size (width, height)
    """
    plt.figure(figsize=figsize)

    for ticker, ticker_data in all_dict.items():
        plt.plot(ticker_data.index, ticker_data['Close'], label=f"Close_{ticker}")

    plt.grid(True)
    plt.show()

def plot_tickers_return(all_dict, figsize=(30, 10)):
    plt.figure(figsize=figsize)

    returns_list = []
    for name in all_dict:
        returns = np.log(all_dict[name]['Close'] / all_dict[name]['Close'].shift(1))
        plt.plot(returns, label=name)
        returns.name = name  # name the series so we can concat later
        returns_list.append(returns)

    # Build dataframe all at once
    df_returns = pd.concat(returns_list, axis=1)

    plt.grid(True)
    plt.show()
    plt.show()
    return df_returns