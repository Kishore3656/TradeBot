# ==========================================================
# 🎯 DAY 5: Model Comparison – Linear Regression vs Random Forest
# ==========================================================

# ----------------------------------------------------------
# Step 1: Import the necessary libraries
# ----------------------------------------------------------
# numpy / pandas -> data handling
# matplotlib     -> plotting
# sklearn        -> machine learning tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ----------------------------------------------------------
# Step 2: Load and prepare the dataset
# ----------------------------------------------------------
# We'll reuse the diabetes dataset (small & clean for demos)
data = load_diabetes(as_frame=True)
df = data.frame.copy()        # make a safe copy

print("✅ Data loaded:", df.shape, "rows/columns")
print(df.head())              # peek at the first few rows


# ----------------------------------------------------------
# Step 3: Handle (simulated) missing values
# ----------------------------------------------------------
# Every 20th row in 'bmi' will be set to NaN to practise cleaning
df.loc[::20, 'bmi'] = np.nan
print("\nMissing values before fill:\n", df.isna().sum())

# Fill missing BMI values with the column mean
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
print("Missing values after fill:\n", df.isna().sum())


# ----------------------------------------------------------
# Step 4: Split data into features (X) and target (y)
# ----------------------------------------------------------
X = df.drop(columns=['target'])   # independent variables
y = df['target']                  # dependent variable we predict


# ----------------------------------------------------------
# Step 5: Train/test split
# ----------------------------------------------------------
# 80% for training, 20% for testing (standard choice)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples: {len(X_train)},  Testing samples: {len(X_test)}")


# ----------------------------------------------------------
# Step 6: Feature scaling (for Linear Regression)
# ----------------------------------------------------------
# Linear models are sensitive to feature scales,
# RandomForest is not (it splits by thresholds, not slopes).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ----------------------------------------------------------
# Step 7: Train Linear Regression model
# ----------------------------------------------------------
lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train)
print("\n✅ Linear Regression trained.")


# ----------------------------------------------------------
# Step 8: Train Random Forest model
# ----------------------------------------------------------
# RandomForest builds many small Decision Trees and averages them.
# n_estimators = number of trees (200 is a good start)
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)     # unscaled is fine
print("✅ Random Forest trained.")


# ----------------------------------------------------------
# Step 9: Make predictions
# ----------------------------------------------------------
y_pred_lin = lin_model.predict(X_test_scaled)
y_pred_rf  = rf_model.predict(X_test)

print("\n🔹 Sample predictions comparison (first 5):")
for i in range(5):
    print(f"Actual: {y_test.iloc[i]:.1f} | LinReg: {y_pred_lin[i]:.1f} | RandomForest: {y_pred_rf[i]:.1f}")


# ----------------------------------------------------------
# Step 10: Define a helper function for evaluation
# ----------------------------------------------------------
def evaluate_model(y_true, y_pred, name="Model"):
    """Print MAE, MSE, and R² neatly"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"\n📊 {name} Performance:")
    print(f"MAE (avg error): {mae:.2f}")
    print(f"MSE (squared error): {mse:.2f}")
    print(f"R² (fit quality): {r2:.3f}")
    return mae, mse, r2

# Evaluate both
lin_results = evaluate_model(y_test, y_pred_lin, "Linear Regression")
rf_results  = evaluate_model(y_test, y_pred_rf,  "Random Forest")


# ----------------------------------------------------------
# Step 11: Compare visually (bar chart of R²)
# ----------------------------------------------------------
models = ['Linear Regression', 'Random Forest']
r2_scores = [lin_results[2], rf_results[2]]

plt.figure(figsize=(5,4))
plt.bar(models, r2_scores, color=['orange','green'])
plt.title("Model Comparison by R² Score (Higher = Better)")
plt.ylabel("R² value")
plt.show()


# ----------------------------------------------------------
# Step 12: Summary / takeaways
# ----------------------------------------------------------
print("""
✅ RESULTS SUMMARY:
- Linear Regression tries to fit a straight line through data.
- Random Forest builds many trees and averages their results.
- Because the relationship in data is not perfectly linear,
  Random Forest usually gives smaller errors and higher R².
- In real projects, we compare multiple models like this and
  pick the one with the best balance of accuracy + simplicity.
""")
