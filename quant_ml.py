# ==========================================================
# 🎯 DAY 9: Predict Next-Day Stock Direction (Up / Down)
# ==========================================================

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load data
data = yf.download("RELIANCE.NS", period="2y", interval="1d")
data = data[['Close']]

# Step 2: Create features
data['Return'] = data['Close'].pct_change()
data['MA5'] = data['Close'].rolling(5).mean()
data['MA20'] = data['Close'].rolling(20).mean()
data['MA_Cross'] = data['MA5'] - data['MA20']

# Proper RSI calculation
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# Drop missing rows
data.dropna(inplace=True)

data['Target'] = np.where(data['Return'].shift(-1) > 0, 1, 0)

# Features (X) and target (y)
X = data[['RSI', 'MA5', 'MA20', 'MA_Cross']]
y = data['Target']

# Split into training and testing (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # no random shuffle (to respect time order)
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)


acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Up=1 / Down=0)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

feat_importance = pd.Series(model.feature_importances_, index=X.columns)
feat_importance.sort_values().plot(kind='barh', color='green')
plt.title("Feature Importance in Prediction")
plt.show()
