# viz.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- Load dataset ---
diabetes = load_diabetes(as_frame=True)
df = diabetes.frame.copy()
df["target"] = diabetes.target

# --- Create simple classification label (High vs Low progression) ---
median_y = df["target"].median()
df["label"] = (df["target"] >= median_y).astype(int)

# --- Ensure figure folder exists ---
os.makedirs("notebooks/figures", exist_ok=True)

# --- 1. Target distribution ---
plt.figure()
df["label"].value_counts().plot(kind="bar", color=["skyblue", "salmon"])
plt.title("Target Distribution (High vs Low Progression)")
plt.xlabel("Label (0 = Low, 1 = High)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("notebooks/figures/target_distribution.png")
plt.close()

# --- 2. Correlation heatmap ---
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("notebooks/figures/correlation_heatmap.png")
plt.close()

# --- 3. Confusion matrix (simple classification model) ---
X = df.drop(columns=["target", "label"])
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Classification Model)")
plt.tight_layout()
plt.savefig("notebooks/figures/confusion_matrix.png")
plt.close()

# --- 4. Residuals plot (simple regression model) ---
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
residuals = y_test - y_pred
plt.figure()
sns.histplot(residuals, kde=True, bins=20, color="purple")
plt.title("Residuals Distribution (Regression Model)")
plt.xlabel("Residuals (y_true - y_pred)")
plt.tight_layout()
plt.savefig("notebooks/figures/residuals_plot.png")
plt.close()

print("✅ All 4 plots saved in notebooks/figures/")
