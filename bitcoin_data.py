import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download Bitcoin price data from Yahoo Finance
data = yf.download("BTC-USD", start="2024-01-01", end="2025-01-01", interval="1d")

# Show the first few rows
print(data.head())

plt.figure(figsize=(10, 5))
plt.plot(data.index, data["Close"], label="BTC Close Price", color='blue')
plt.title("Bitcoin Price (2024)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show()