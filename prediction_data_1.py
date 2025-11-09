import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Download 2 years of data for Reliance (you can change the ticker)
data = yf.download("RELIANCE.NS", period="2y", interval="1d")

# Show first few rows
print(data.head())

# Plot closing price
plt.figure(figsize=(10,4))
plt.plot(data['Close'], label='Closing Price')
plt.title("Reliance Stock Price (2 Years)")
plt.legend()
plt.show()

data['Return'] = data['Close'].pct_change()
data['MA5'] = data['Close'].rolling(5).mean()
data['MA20'] = data['Close'].rolling(20).mean()
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

data.dropna(inplace=True)

corr = data[['Return', 'MA5', 'MA20', 'RSI']].corr()
print(corr)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()


data['MA_Cross'] = data['MA5'] - data['MA20']

corr = data[['Return', 'MA5', 'MA20', 'MA_Cross', 'RSI']].corr()
print(corr['Return'].sort_values(ascending=False))
