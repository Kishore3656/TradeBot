import os
import warnings
warnings.filterwarnings("ignore")  # ✅ suppress sklearn/yfinance warnings

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# =====================================================
# 🧠 Function to run full quant analysis for one stock
# =====================================================
def run_quant_analysis(ticker, plot_dir, report_dir):
    # ============= Fetch Data =============
    df = yf.download(ticker, period="2y", interval="1d", auto_adjust=True)
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["Signal"] = np.where(df["MA10"] > df["MA50"], 1, 0)
    df["Return"] = df["Close"].pct_change()
    df.dropna(inplace=True)

    # ============= Features & Labels =============
    X = df[["MA10", "MA50", "Return"]]
    y = df["Signal"]

    # ============= Train/Test Split =============
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # ============= Train Model =============
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ============= Metrics =============
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)

    # ============= Backtest PnL =============
    df["Pred_Signal"] = np.nan
    df.iloc[-len(y_test):, df.columns.get_loc("Pred_Signal")] = y_pred
    df["Strategy_Return"] = df["Pred_Signal"].shift(1) * df["Return"]
    total_return = df["Strategy_Return"].sum() * 100

    # ============= Price Chart =============
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["Close"], label="Close Price", linewidth=1)
    plt.plot(df.index, df["MA10"], label="MA10", linestyle="--")
    plt.plot(df.index, df["MA50"], label="MA50", linestyle="--")
    plt.title(f"{ticker} Price + Moving Averages")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f"{ticker}_plot.png")
    plt.savefig(plot_path)
    plt.close()

    # ============= Confusion Matrix Plot =============
    fig, ax = plt.subplots(figsize=(4, 4))
    cax = ax.matshow(cm, cmap="Blues")
    plt.title("Confusion Matrix", pad=15)
    plt.colorbar(cax)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val}", ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(plot_dir, f"{ticker}_cm.png")
    plt.savefig(cm_path)
    plt.close()

    # ============= PDF Report =============
    pdf = FPDF()

    # --- Page 1: Summary + Chart ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(200, 10, text=f"Quant ML Report: {ticker}",
              new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    pdf.set_font("Helvetica", size=12)
    pdf.cell(200, 10, text=f"Model Accuracy: {acc:.2f}",
              new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(200, 10, text=f"Total Strategy Return: {total_return:.2f}%",
              new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(200, 10, text=f"Data Period: Last 2 Years (Daily)",
              new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.image(plot_path, x=10, y=60, w=180)

    # --- Page 2: Performance Metrics ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(200, 10, text="Performance Metrics", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.image(cm_path, x=10, y=25, w=100)

    pdf.set_xy(10, 140)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 8, text=report_text)

    # Save PDF
    pdf_path = os.path.join(report_dir, f"{ticker}_report.pdf")
    pdf.output(pdf_path)

    print(f"✅ Report generated for {ticker} | Accuracy: {acc:.2f} | Return: {total_return:.2f}%")

# =====================================================
# 📁 Folder Setup
# =====================================================
base_dir = r"D:\trading bot"
plot_dir = os.path.join(base_dir, "plots")
report_dir = os.path.join(base_dir, "reports")

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

# =====================================================
# 📈 List of Stocks
# =====================================================
stocks = ["RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS", "SBIN.NS"]

# =====================================================
# 🚀 Run Analysis for All Stocks
# =====================================================
for stock in stocks:
    try:
        run_quant_analysis(stock, plot_dir, report_dir)
    except Exception as e:
        print(f"❌ Error processing {stock}: {e}")
