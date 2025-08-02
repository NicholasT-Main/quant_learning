import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

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

def plot_stock_distributions(final_groups, df_returns,bins_size=100):
    for group in final_groups:
        data = df_returns[group].dropna()

        plt.figure(figsize=(15, 5))

        all_data = [data[stock].dropna() for stock in group]
        labels = group
        colors = ['red', 'blue', 'magenta', 'skyblue', 'yellow']  # Customize as needed

        # Define common bins
        # This is the amount of bars the values will be split between
        combined = np.concatenate(all_data)
        bins = np.histogram_bin_edges(combined, bins=bins_size)

        bar_width = (bins[1] - bins[0]) / len(group)  # width of each bar
        for i, stock in enumerate(group):
            stock_data = all_data[i]
            hist, _ = np.histogram(stock_data, bins=bins, density=True)

            # Shift bars for side-by-side placement
            bin_centers = bins[:-1] + bar_width * i + bar_width / 2

            plt.bar(bin_centers, hist, width=bar_width, alpha=0.6,
                    label=f'{stock} Histogram', color=colors[i % len(colors)])

            # Plot normal distribution
            mean = stock_data.mean()
            # plt.axvline(mean,lw=1,label=f'{stock} Mean', alpha=1,color=colors[i % len(colors)])
            std = stock_data.std()
            x = np.linspace(stock_data.min(), stock_data.max(), 1000)
            plt.plot(x, norm.pdf(x, mean, std), lw=1, label=f'{stock} Normal PDF', color=colors[i % len(colors)])

            print(f'{stock} returns: mean:{mean * 100:.3f}% +/- std:{std * 100:.3f}%')

        plt.title(f'Histograms, Means and Normal Curves: {group}')

        plt.xlabel('Return')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()