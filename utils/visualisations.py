import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

def heatmap_correlations(correlation_matrix):
    plt.figure(figsize=(round(len(correlation_matrix) / 4), round(len(correlation_matrix) / 4)))
    sns.heatmap(correlation_matrix, annot=False, cmap='RdYlBu_r', center=0, square=True, linewidths=0.1)
    plt.title('Stock Correlation Matrix')
    plt.show()