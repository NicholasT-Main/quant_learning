import pandas as pd
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt


class Data:
    def __init__(self, data):
        # Convert dictionary or DataFrame into a proper pandas DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame(data)
            # Try to set a 'Date' column as the index if it exists
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            self.data = df
        elif isinstance(data, pd.DataFrame):
            self.data = data
        else:
            raise TypeError("Data must be a pandas DataFrame or dict.")

    def __len__(self):
        return len(self.data)

    def empty(self) -> bool:
        return self.data.empty

    def __getattr__(self, name):
        # Delegate attribute access (like .index, .columns, etc.) to the internal DataFrame
        return getattr(self.data, name)

    def __getitem__(self, key):
        return self.data[key]

    def __repr__(self):
        return f"Data({len(self.data)} rows)"


class Stock:
    """
    A simple stock object for quantitative research.
    Stores price data and provides basic access methods.
    """

    def __init__(self, ticker: str, data: Optional[pd.DataFrame] = None):
        """
        Initialize a Stock object.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        data : pd.DataFrame or dict, optional
            DataFrame or dict with OHLCV data or similar.
            Expected columns: ['open', 'high', 'low', 'close', 'volume']
            Index should be datetime
        """
        self.ticker = ticker.upper()
        if data is not None:
            self.data = Data(data)
        else:
            self.data = Data(pd.DataFrame())

    def __repr__(self) -> str:
        if not self.data.empty():
            return (f"Stock('{self.ticker}', "
                    f"{len(self.data)} rows, "
                    f"{self.data.index[0].date()} to {self.data.index[-1].date()})")
        return f"Stock('{self.ticker}', empty)"

    def __len__(self) -> int:
        return len(self.data)

    def visualise(self, key: str) -> None:
        """
        Plots a single data column against the date index.

        Parameters:
        -----------
        key : str
            The column name (e.g., 'close', 'volume', 'RSI') to plot.
        """
        if self.data.empty():
            print(f"Cannot plot for {self.ticker}: Data is empty.")
            return

        if key not in self.data.columns:
            print(f"Error: Column '{key}' not found in stock data.")
            return

        plt.figure(figsize=(10, 6))
        self.data[key].plot(title=f"{self.ticker} - {key.capitalize()}")
        plt.ylabel(key.capitalize())
        plt.xlabel("Date")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    stock = Stock(
        'AAPL',
        {'test': (1, 2, 3, 4, 5, 6, 7,),
         'Date': ('2025-10-08', '2025-10-09', '2025-10-10', '2025-10-11', '2025-10-12', '2025-10-13', '2025-10-14',)}
    )

    stock.plot("test")
    print(stock)
