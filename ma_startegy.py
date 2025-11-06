import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download Bitcoin daily price data
data = yf.download("BTC-USD", start="2024-01-01", end="2025-01-01", interval="1d", auto_adjust=False)

# Calculate Moving Averages
data["MA_short"] = data["Close"].rolling(window=10).mean()   # 10-day moving average
data["MA_long"] = data["Close"].rolling(window=30).mean()    # 30-day moving average

# Generate Buy/Sell signals
data["Signal"] = 0
data.loc[data["MA_short"] > data["MA_long"], "Signal"] = 1   # buy zone
data.loc[data["MA_short"] < data["MA_long"], "Signal"] = -1  # sell zone

# Detect crossover points (when signal changes)
data["Crossover"] = data["Signal"].diff()

# Plot price and MAs
plt.figure(figsize=(12,6))
plt.plot(data.index, data["Close"], label="Close Price", color="blue", alpha=0.6)
plt.plot(data.index, data["MA_short"], label="10-Day MA", color="green")
plt.plot(data.index, data["MA_long"], label="30-Day MA", color="red")

# Mark Buy and Sell signals
plt.plot(data[data["Crossover"] == 2].index,  # -1 → +1
         data["MA_short"][data["Crossover"] == 2],
         "^", markersize=10, color="g", label="Buy Signal")

plt.plot(data[data["Crossover"] == -2].index,  # +1 → -1
         data["MA_short"][data["Crossover"] == -2],
         "v", markersize=10, color="r", label="Sell Signal")

plt.title("Bitcoin Moving Average Crossover Strategy (2024)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show()


# Simple simulation: Buy 1 unit on Buy signal, Sell on Sell signal
positions = []
cash = 0.0
holding = False

for i in range(len(data)):
    crossover = data["Crossover"].iloc[i]
    close_price = float(data["Close"].iloc[i].iloc[0])  # ensure it's a number

    if crossover == 2 and not holding:  # Buy signal
        buy_price = float(data["Close"].iloc[i].iloc[0])  # explicitly convert
        holding = True
        positions.append(("BUY", data.index[i], buy_price))

    elif crossover == -2 and holding:  # Sell signal
        sell_price = float(data["Close"].iloc[i].iloc[0])  # explicitly convert
        profit = float(sell_price - buy_price)     # ensure pure float
        cash += profit
        holding = False
        positions.append(("SELL", data.index[i], sell_price, profit))

print(f"\nTotal Profit (1 unit per trade): ${cash:.2f}")
print("Last 5 trades:")
for p in positions[-5:]:
    print(p)



# PHASE 1–2 code ...
# (download data, calculate MAs, signals, crossover, plotting)
# (simulate trades and print total profit)

# --------------- ADD PHASE 3 BELOW THIS LINE ---------------

import numpy as np

# Convert positions list to DataFrame
trades = pd.DataFrame(
    [p for p in positions if p[0] == "SELL"],
    columns=["Type", "Date", "Sell_Price", "Profit"]
)

# Performance metrics
total_profit = trades["Profit"].sum()
num_trades = len(trades)
win_trades = trades[trades["Profit"] > 0]
loss_trades = trades[trades["Profit"] <= 0]

win_rate = len(win_trades) / num_trades * 100 if num_trades > 0 else 0
avg_profit = trades["Profit"].mean() if num_trades > 0 else 0
max_drawdown = trades["Profit"].cumsum().cummax() - trades["Profit"].cumsum()
max_drawdown = max_drawdown.max()

print(f"\n📊 Strategy Performance Summary:")
print(f"Total Trades: {num_trades}")
print(f"Winning Trades: {len(win_trades)} ({win_rate:.1f}%)")
print(f"Total Profit: ${total_profit:.2f}")
print(f"Average Profit per Trade: ${avg_profit:.2f}")
print(f"Max Drawdown: ${max_drawdown:.2f}")

# Profit curve
plt.figure(figsize=(10,4))
plt.plot(trades["Date"], trades["Profit"].cumsum(), label="Cumulative Profit")
plt.title("Profit Curve Over Time")
plt.xlabel("Date")
plt.ylabel("Cumulative Profit ($)")
plt.grid(True)
plt.legend()
plt.show()


import os
from datetime import datetime

# --------------------- SAVE TRADES TO CSV ----------------------

# Ensure 'logs' folder exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Add cumulative profit to trades DataFrame
if not trades.empty:
    trades["Cumulative_Profit"] = trades["Profit"].cumsum()

    # Create a timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"trades_{timestamp}.csv")

    # Save trades to CSV
    trades.to_csv(log_file, index=False)
    print(f"\n💾 Trade log saved to: {log_file}")
else:
    print("\n⚠️ No trades to log (trades DataFrame is empty).")

# --------------------- SAVE PERFORMANCE SUMMARY ----------------------

summary_file = os.path.join(log_dir, "performance_history.csv")

# Create a DataFrame with the run summary
summary_data = pd.DataFrame([{
    "Run_Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Total_Trades": num_trades,
    "Winning_Trades": len(win_trades),
    "Win_Rate_%": round(win_rate, 2),
    "Total_Profit_$": round(total_profit, 2),
    "Average_Profit_$": round(avg_profit, 2),
    "Max_Drawdown_$": round(max_drawdown, 2)
}])

# Append to file if it exists, otherwise create it
if os.path.exists(summary_file):
    summary_data.to_csv(summary_file, mode="a", header=False, index=False)
else:
    summary_data.to_csv(summary_file, mode="w", header=True, index=False)

print(f"📈 Performance summary updated: {summary_file}")
