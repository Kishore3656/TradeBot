import pandas as pd
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt

# Load sample data
data = load_diabetes(as_frame=True)
df = data.frame

# Show first few rows
print("🔹 First 5 rows of data:")
print(df.head())
# Summary statistics
print("\n🔹 Summary statistics:")
print(df.describe())

# Quick scatter plot: BMI vs Target
df.plot(kind='scatter', x='bmi', y='target', title='BMI vs Target')
plt.show()



import pandas as pd
from sklearn.datasets import load_diabetes

data = load_diabetes(as_frame=True)
df = data.frame
print("✅ Data loaded successfully!")
df.head()

import numpy as np

# Simulate missing values in BMI column
df.loc[::20, 'bmi'] = np.nan
print("Number of missing values before filling:")
print(df.isna().sum())

# Fill missing values with mean
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

print("\nNumber of missing values after filling:")
print(df.isna().sum())

X = df.drop(columns=['target'])
y = df['target']

print("Features shape:", X.shape)
print("Target shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Scaling complete!")
print("Before scaling mean of BMI:", X_train['bmi'].mean())
print("After scaling mean of BMI:", X_train_scaled[:, 2].mean())
