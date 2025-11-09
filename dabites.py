# ==========================================================
# 🎯 DAY 7 (Part 2): Find & Plot Top 5 Most Correlated Features
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes

# Step 1: Load the dataset
data = load_diabetes(as_frame=True)
df = data.frame.copy()

# Step 2: Compute correlation with the target
corr_with_target = df.corr()['target'].drop('target')  # exclude target itself

# Step 3: Sort by absolute correlation (so both + and – matter)
corr_sorted = corr_with_target.abs().sort_values(ascending=False)

# Step 4: Take the top 5 strongest correlations
top5 = corr_sorted.head(5)
print("🔹 Top 5 most correlated features with the target:\n")
print(top5)

# Step 5: Plot them
plt.figure(figsize=(6,4))
sns.barplot(x=top5.values, y=top5.index, palette='viridis')
plt.title("Top 5 Most Correlated Features with Target")
plt.xlabel("Correlation Strength (|value|)")
plt.ylabel("Feature Name")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()
