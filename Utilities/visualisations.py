import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
import itertools
import yfinance as yf

def plot_tickers_values(all_dict, figsize=(30, 10)):
    """
    Plot the closing prices for all tickers in the given dictionary.

    Parameters:
    ----------
    all_dict : dict
        Dictionary of DataFrames, keyed by ticker symbol, each containing at least a 'Close' column.

    figsize : tuple, optional
        Size of the matplotlib figure (width, height). Default is (30, 10).
    """
    plt.figure(figsize=figsize)
    for ticker, ticker_data in all_dict.items():
        plt.plot(ticker_data.index, ticker_data['Close'], label=f"Close_{ticker}")
    plt.legend()
    plt.grid(True)
    plt.title("Ticker Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()


def plot_tickers_return(all_dict, figsize=(30, 10)):
    """
    Plot the log returns for each ticker and return a DataFrame of the combined returns.

    Parameters:
    ----------
    all_dict : dict
        Dictionary of DataFrames, keyed by ticker symbol, each containing at least a 'Close' column.

    figsize : tuple, optional
        Size of the matplotlib figure (width, height). Default is (30, 10).

    Returns:
    -------
    pd.DataFrame
        DataFrame where each column contains log returns for one ticker.
    """
    plt.figure(figsize=figsize)
    returns_list = []

    for name in all_dict:
        returns = np.log(all_dict[name]['Close'] / all_dict[name]['Close'].shift(1))
        returns.name = name
        plt.plot(returns, label=name)
        returns_list.append(returns)

    df_returns = pd.concat(returns_list, axis=1)
    plt.legend()
    plt.grid(True)
    plt.title("Log Returns of Tickers")
    plt.xlabel("Date")
    plt.ylabel("Log Return")
    plt.show()

    return df_returns


def heatmap_correlations(correlation_matrix):
    """
    Plot a heatmap of a stock correlation matrix.

    Parameters:
    ----------
    correlation_matrix : pd.DataFrame
        A square DataFrame of correlation values between stock returns.
    """
    size = round(len(correlation_matrix) / 4)
    plt.figure(figsize=(size, size))
    sns.heatmap(correlation_matrix, annot=False, cmap='RdYlBu_r',
                center=0, square=True, linewidths=0.1)
    plt.title('Stock Correlation Matrix')
    plt.show()


def plot_stock_distributions(final_groups, df_returns, bins_size=100):
    """
    Plot side-by-side histograms and normal distribution curves for each group of stocks.

    Parameters:
    ----------
    final_groups : list of lists
        Each sublist contains ticker symbols to group and compare together.

    df_returns : pd.DataFrame
        DataFrame where each column is the log return for a ticker.

    bins_size : int, optional
        Number of bins for the histogram. Default is 100.
    """
    for group in final_groups:
        data = df_returns[group].dropna()
        plt.figure(figsize=(15, 5))

        all_data = [data[stock].dropna() for stock in group]

        # Define common bin edges
        combined = np.concatenate(all_data)
        bins = np.histogram_bin_edges(combined, bins=bins_size)
        bar_width = (bins[1] - bins[0]) / len(group)

        for i, stock in enumerate(group):
            stock_data = all_data[i]
            hist, _ = np.histogram(stock_data, bins=bins, density=True)
            bin_centers = bins[:-1] + bar_width * i + bar_width / 2

            # Plot histogram
            plt.bar(bin_centers, hist, width=bar_width, alpha=0.6,
                    label=f'{stock} Histogram')

            # Plot normal distribution
            mean = stock_data.mean()
            std = stock_data.std()
            x = np.linspace(stock_data.min(), stock_data.max(), 1000)
            plt.plot(x, norm.pdf(x, mean, std), lw=1,
                     label=f'{stock} Normal PDF')

            print(f'{stock} returns: mean: {mean * 100:.3f}% +/- std: {std * 100:.3f}%')

        plt.title(f'Histograms, Means and Normal Curves: {group}')
        plt.xlabel('Return')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

def print_ewma_correlations(group, group_closes, ewma_span):
    """
    Print the correlation of each stock’s EWM-smoothed prices with the group average
    (excluding itself), and the average within-group correlation.

    Parameters:
    ----------
    group : list
        List of tickers in the group.

    group_closes : pd.DataFrame
        Closing prices for all tickers in the group.

    ewma_span : int
        Span for the exponential moving average.
    """
    print_stats = ""

    for stock in group:
        stock_ewm = group_closes[stock].ewm(span=ewma_span).mean()
        other_stocks = [s for s in group if s != stock]
        other_ewm = group_closes[other_stocks].ewm(span=ewma_span).mean()
        group_avg_excl = other_ewm.mean(axis=1)

        corr_with_avg = stock_ewm.corr(group_avg_excl)
        try:
            stock_name = yf.Ticker(stock).info.get('longName', stock)
        except Exception:
            stock_name = stock
        print_stats = (f"{stock} vs GROUP_EWM_AVG (excluding {stock}): {corr_with_avg:.3f},  "
                       f"{stock_name}\n{print_stats}")

    print("returns correlations:")
    print(print_stats)


def plot_group_prices_with_avg(group, all_dict, figsize=(15, 5)):
    """
    Plot dashed lines for individual stocks and a solid line for group average closing price.

    Parameters:
    ----------
    group : list
        List of tickers in the group.

    all_dict : dict
        Dictionary of ticker DataFrames containing 'Close' price.

    figsize : tuple
        Size of the matplotlib figure.

    Returns:
    -------
    pd.DataFrame
        DataFrame of closing prices for all tickers in the group.
    """
    plt.figure(figsize=figsize)
    group_closes = pd.DataFrame()

    for stock in group:
        group_closes[stock] = all_dict[stock]['Close']
        plt.plot(group_closes[stock], label=stock, linestyle='dashed', linewidth=0.2)

    group_avg_prices = group_closes.mean(axis=1)
    plt.plot(group_avg_prices, label='GROUP_AVG_PRICES', linewidth=0.7, color='red')

    plt.title(f'Group: {group}', size=10)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    return group_closes
