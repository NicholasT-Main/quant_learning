import yfinance as yf
import Mean_reversion

# Download NVDA daily data (last 5 years)
nvda = yf.download("NVDA", period="5y")

# Generate signals
signals = Mean_reversion.mean_reversion_strategy(nvda, lookback=20)

print(signals.tail(10))
print(signals.head(25))