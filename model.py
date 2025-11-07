# ==========================================================
# 🎯 DAY 4: Model Evaluation – Checking How Good Our Model Is
# ==========================================================

# We’ll continue using Linear Regression on the diabetes dataset.
# You’ll learn what MAE, MSE, and R² really mean by seeing them in action.

# ----------------------------------------------------------
# Step 1: Import all the tools we need
# ----------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------
# Step 2: Load the dataset
# ----------------------------------------------------------
# The diabetes dataset has 442 rows and 10 input features.
# Our goal is to predict 'target' (a continuous number).

data = load_diabetes(as_frame=True)
df = data.frame
print("✅ Data loaded successfully!")
print(df.head())  # Show first few rows

# ----------------------------------------------------------
# Step 3: Simulate and fix missing values
# ----------------------------------------------------------
# We’ll add some missing values to 'bmi' just to practice cleaning them.
import numpy as np
df.loc[::20, 'bmi'] = np.nan   # make every 20th row missing
print("\nMissing values before filling:\n", df.isna().sum())

# Fill missing BMI values with the column mean
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
print("\nMissing values after filling:\n", df.isna().sum())

# ----------------------------------------------------------
# Step 4: Split data into features (X) and target (y)
# ----------------------------------------------------------
X = df.drop(columns=['target'])  # input columns
y = df['target']                 # output column

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

# ----------------------------------------------------------
# Step 5: Train-test split
# ----------------------------------------------------------
# 80% of data is for training, 20% for testing.
# random_state=42 → reproducible split (same each time you run).
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ----------------------------------------------------------
# Step 6: Feature Scaling (Standardization)
# ----------------------------------------------------------
# Linear regression works best when all features are on similar scales.
# So we scale them to mean = 0 and std = 1.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\n✅ Data scaling complete!")

# ----------------------------------------------------------
# Step 7: Train Linear Regression model
# ----------------------------------------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("✅ Model training complete!")

# ----------------------------------------------------------
# Step 8: Make predictions
# ----------------------------------------------------------
y_pred = model.predict(X_test_scaled)
print("\n🔹 Sample predictions:")
for i in range(5):
    print(f"Actual: {y_test.iloc[i]:.2f} | Predicted: {y_pred[i]:.2f}")

# ----------------------------------------------------------
# Step 9: Evaluate the model
# ----------------------------------------------------------
# MAE → Mean Absolute Error (average difference)
# MSE → Mean Squared Error (squared difference, punishes large errors)
# R² → How much of the pattern the model explains (1 = perfect, 0 = random)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation Metrics:")
print(f"MAE (average error): {mae:.2f}")
print(f"MSE (squared error): {mse:.2f}")
print(f"R² (fit quality): {r2:.3f}")

# ----------------------------------------------------------
# Step 10: Visualize Actual vs Predicted
# ----------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Target Values")
plt.ylabel("Predicted Target Values")
plt.title("Actual vs Predicted (Red line = Perfect Prediction)")
plt.show()

# ----------------------------------------------------------
# Step 11: Visualize Errors (Residuals)
# ----------------------------------------------------------
residuals = y_test - y_pred  # difference between actual and predicted

plt.figure(figsize=(6, 4))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot (Ideally, dots should be around the line = 0)")
plt.show()

# ----------------------------------------------------------
# Step 12: Histogram of Errors
# ----------------------------------------------------------
# This shows if the errors are roughly centered around 0 (good)
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=20, color='skyblue', edgecolor='black')
plt.title("Distribution of Errors (Residuals)")
plt.xlabel("Error Value")
plt.ylabel("Frequency")
plt.show()

# ----------------------------------------------------------
# Step 13: Summary of what we learned
# ----------------------------------------------------------
print("""
✅ SUMMARY:
- MAE tells you the average size of your model’s mistakes.
- MSE is similar, but punishes big mistakes more.
- R² tells you how much of the pattern in the data your model captured.
- Residual plots help you see if your model is missing something.
""")

print("🎯 Day 4 complete! You now understand how to measure model performance.")
