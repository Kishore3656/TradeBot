# ==========================================================
# 🎯 DAY 10: Improve Precision & Recall with Extra Features
# ==========================================================

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load data
data = yf.download("RELIANCE.NS", period="2y", interval="1d")

# Step 2: Base features
data['Return'] = data['Close'].pct_change()
data['MA5'] = data['Close'].rolling(5).mean()
data['MA20'] = data['Close'].rolling(20).mean()
data['MA_Cross'] = data['MA5'] - data['MA20']

# Proper RSI
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# NEW FEATURE 1: Volume (scaled)
data['Vol_Change'] = data['Volume'].pct_change()

# NEW FEATURE 2: Volatility (rolling std of returns)
data['Volatility'] = data['Return'].rolling(10).std()

# NEW FEATURE 3: Lag returns (previous 3 days)
data['Lag1'] = data['Return'].shift(1)
data['Lag2'] = data['Return'].shift(2)
data['Lag3'] = data['Return'].shift(3)

# Drop missing values
data.dropna(inplace=True)


data['Target'] = np.where(data['Return'].shift(-1) > 0, 1, 0)


features = ['RSI','MA5','MA20','MA_Cross','Vol_Change','Volatility','Lag1','Lag2','Lag3']

X = data[features]

# Clean invalid values before training
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("✅ Model Accuracy:", round(accuracy_score(y_test, y_pred),2))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Improved Model - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


importance = pd.Series(model.feature_importances_, index=features)
importance.sort_values().plot(kind='barh', figsize=(6,4), color='green')
plt.title("Feature Importance After Enhancement")
plt.show()

# Make predictions on the test set
data_test = data.iloc[-len(y_test):].copy()  # align last test period
data_test['Predicted'] = y_pred

# Strategy returns = next day's actual return * prediction
data_test['Strategy_Return'] = data_test['Return'].shift(-1) * data_test['Predicted']

# Cumulative returns
data_test['Cumulative_Market'] = (1 + data_test['Return']).cumprod()
data_test['Cumulative_Strategy'] = (1 + data_test['Strategy_Return']).cumprod()

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(data_test['Cumulative_Market'], label='Market (Buy & Hold)')
plt.plot(data_test['Cumulative_Strategy'], label='Model Strategy')
plt.title('Backtest: Model vs Market')
plt.legend()
plt.show()

data_test['Strategy_Return'] = data_test['Strategy_Return'].fillna(0)

data_test['Cumulative_Market'] = (1 + data_test['Return']).cumprod()
data_test['Cumulative_Strategy'] = (1 + data_test['Strategy_Return']).cumprod()

# Total Return
total_market = data_test['Cumulative_Market'].iloc[-1] - 1
total_strategy = data_test['Cumulative_Strategy'].iloc[-1] - 1

# Annualized Return (approx for daily data)
trading_days = len(data_test)
annual_market = (1 + total_market) ** (252/trading_days) - 1
annual_strategy = (1 + total_strategy) ** (252/trading_days) - 1

print(f"📊 Total Market Return: {total_market*100:.2f}%")
print(f"📈 Total Strategy Return: {total_strategy*100:.2f}%")
print(f"🏦 Annualized Market Return: {annual_market*100:.2f}%")
print(f"🚀 Annualized Strategy Return: {annual_strategy*100:.2f}%")

# Daily strategy returns standard deviation
volatility = data_test['Strategy_Return'].std() * np.sqrt(252)

# Sharpe Ratio (return-to-risk measure)
sharpe = (annual_strategy - 0.05) / volatility  # assuming 5% risk-free rate

print(f"⚙️ Strategy Volatility: {volatility:.2f}")
print(f"⭐ Sharpe Ratio: {sharpe:.2f}")



# Copy your last data_test with predictions
data_test = data.iloc[-len(y_test):].copy()
data_test['Predicted'] = y_pred

# Step 1: Basic strategy return
data_test['Strategy_Return'] = data_test['Return'].shift(-1) * data_test['Predicted']

# Step 2: Detect trade changes (signal flips)
data_test['Trade'] = data_test['Predicted'].diff().fillna(0).abs()  # 1 if a trade occurred

# Step 3: Apply transaction cost (0.1% per trade)
transaction_cost = 0.001
data_test['Cost'] = data_test['Trade'] * transaction_cost

# Step 4: Adjust strategy return after cost
data_test['Net_Strategy_Return'] = data_test['Strategy_Return'] - data_test['Cost']

initial_capital = 100000  # start with ₹1,00,000

data_test['Cumulative_Market'] = (1 + data_test['Return']).cumprod() * initial_capital
data_test['Cumulative_Strategy'] = (1 + data_test['Net_Strategy_Return']).cumprod() * initial_capital

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(data_test['Cumulative_Market'], label='Buy & Hold', color='blue')
plt.plot(data_test['Cumulative_Strategy'], label='Strategy (After Costs)', color='orange')
plt.title('Realistic Backtest: Market vs Model Strategy')
plt.xlabel('Date')
plt.ylabel('Account Value (₹)')
plt.legend()
plt.show()


total_market = data_test['Cumulative_Market'].iloc[-1] - initial_capital
total_strategy = data_test['Cumulative_Strategy'].iloc[-1] - initial_capital

total_market_return = (total_market / initial_capital) * 100
total_strategy_return = (total_strategy / initial_capital) * 100

print(f"📊 Market Return: {total_market_return:.2f}%")
print(f"🚀 Strategy Return (after costs): {total_strategy_return:.2f}%")

# Risk-adjusted metrics
volatility = data_test['Net_Strategy_Return'].std() * np.sqrt(252)
annual_return = ((data_test['Cumulative_Strategy'].iloc[-1] / initial_capital) ** (252/len(data_test))) - 1
sharpe = (annual_return - 0.05) / volatility

print(f"⚙️ Strategy Volatility: {volatility:.2f}")
print(f"⭐ Sharpe Ratio: {sharpe:.2f}")

data_test = data.iloc[-len(y_test):].copy()
data_test['Predicted'] = y_pred
data_test['Return'] = data_test['Return']

# Convert predictions to +1 (long) and -1 (short)
data_test['Position'] = data_test['Predicted'].map({1: 1, 0: -1})

# Compute strategy return
data_test['Strategy_Return'] = data_test['Return'].shift(-1) * data_test['Position']

data_test['Trade'] = data_test['Position'].diff().fillna(0).abs()
transaction_cost = 0.001
data_test['Cost'] = data_test['Trade'] * transaction_cost
data_test['Net_Strategy_Return'] = data_test['Strategy_Return'] - data_test['Cost']
data_test['Net_Strategy_Return'] = data_test['Net_Strategy_Return'].fillna(0)

initial_capital = 100000

data_test['Cumulative_Market'] = (1 + data_test['Return']).cumprod() * initial_capital
data_test['Cumulative_Strategy'] = (1 + data_test['Net_Strategy_Return']).cumprod() * initial_capital


import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(data_test['Cumulative_Market'], label='Market (Buy & Hold)', color='blue')
plt.plot(data_test['Cumulative_Strategy'], label='Long/Short Strategy', color='orange')
plt.title('Backtest: Long/Short Model vs Market')
plt.xlabel('Date')
plt.ylabel('Account Value (₹)')
plt.legend()
plt.show()

total_market = data_test['Cumulative_Market'].iloc[-1] - initial_capital
total_strategy = data_test['Cumulative_Strategy'].iloc[-1] - initial_capital

total_market_return = (total_market / initial_capital) * 100
total_strategy_return = (total_strategy / initial_capital) * 100

print(f"📊 Market Return: {total_market_return:.2f}%")
print(f"🚀 Long/Short Strategy Return (after costs): {total_strategy_return:.2f}%")

# Sharpe Ratio
volatility = data_test['Net_Strategy_Return'].std() * np.sqrt(252)
annual_return = ((data_test['Cumulative_Strategy'].iloc[-1] / initial_capital) ** (252/len(data_test))) - 1
sharpe = (annual_return - 0.05) / volatility

print(f"⚙️ Strategy Volatility: {volatility:.2f}")
print(f"⭐ Sharpe Ratio: {sharpe:.2f}")
 