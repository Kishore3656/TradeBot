# ==========================================================
# 🎯 DAY 6: Making Random Forest Smarter (Hyperparameter Tuning)
# ==========================================================

# ----------------------------------------------------------
# Step 1: Import the tools we need
# ----------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------
# Step 2: Load the dataset
# ----------------------------------------------------------
data = load_diabetes(as_frame=True)
df = data.frame.copy()

# Handle missing BMI values (same as before)
df.loc[::20, 'bmi'] = np.nan
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

# Split into features and target
X = df.drop(columns=['target'])
y = df['target']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Data ready! Train size:", X_train.shape, " Test size:", X_test.shape)

# ----------------------------------------------------------
# Step 3: Build a base Random Forest model (for comparison)
# ----------------------------------------------------------
rf_default = RandomForestRegressor(random_state=42)
rf_default.fit(X_train, y_train)

y_pred_default = rf_default.predict(X_test)

# Evaluate performance before tuning
def evaluate_model(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"\n📊 {label}")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R² : {r2:.3f}")
    return r2

r2_default = evaluate_model(y_test, y_pred_default, "Before Tuning (Default RF)")

# ----------------------------------------------------------
# Step 4: Set up GridSearchCV for tuning
# ----------------------------------------------------------
# We'll define a small grid of possible hyperparameters.
# GridSearch will test all combinations and pick the best.

param_grid = {
    'n_estimators': [50, 100, 200],       # number of trees
    'max_depth': [None, 5, 10, 20],       # how deep each tree can go
    'min_samples_split': [2, 5, 10],      # when to split a branch
    'min_samples_leaf': [1, 2, 4]         # minimum samples at leaf node
}

# Create the base model
rf = RandomForestRegressor(random_state=42)

# GridSearchCV setup:
# cv=3 means it will test each combination using 3-fold cross-validation.
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,             # use all CPU cores
    scoring='r2',          # we want best R² score
    verbose=1              # show progress
)

# ----------------------------------------------------------
# Step 5: Run the grid search
# ----------------------------------------------------------
print("\n⏳ Searching for best hyperparameters... (this may take 1–2 minutes)")
grid_search.fit(X_train, y_train)

print("\n✅ Grid search complete!")
print("Best hyperparameters found:")
print(grid_search.best_params_)

# ----------------------------------------------------------
# Step 6: Evaluate the tuned model
# ----------------------------------------------------------
best_rf = grid_search.best_estimator_  # best model found

y_pred_tuned = best_rf.predict(X_test)
r2_tuned = evaluate_model(y_test, y_pred_tuned, "After Tuning (Best RF)")

# ----------------------------------------------------------
# Step 7: Compare before vs after tuning
# ----------------------------------------------------------
models = ['Default RF', 'Tuned RF']
r2_scores = [r2_default, r2_tuned]

plt.figure(figsize=(6,4))
plt.bar(models, r2_scores, color=['gray', 'green'])
plt.title("Random Forest Performance: Before vs After Tuning")
plt.ylabel("R² Score (Higher = Better)")
plt.show()

print("""
✅ SUMMARY:
- GridSearchCV automatically tested multiple Random Forest settings.
- It picked the combination that gave the highest R² on validation data.
- Usually, the tuned model gives better accuracy and smaller errors.
""")
