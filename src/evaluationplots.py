# Generates the following plots: target distribution, correlation heatmap, confusion matrix, and residuals plot.

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

# Creating simple classification label (High vs Low progression)
median_y = df["target"].median()
df["label"] = (df["target"] >= median_y).astype(int)

# Making sure the figure folder exists in notebooks
os.makedirs("notebooks/figures", exist_ok=True)

# We're using light grey and dark grey colors for all our plots for better comparison
light_grey = "#d9d9d9"
dark_grey = "#595959"

# Target distribution plot
plt.figure()
df["label"].value_counts().plot(kind="bar", color=[light_grey, dark_grey])
plt.title("Target Distribution (High vs Low Progression)")
plt.xlabel("Label (0 = Low, 1 = High)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("notebooks/figures/target_distribution.png")
plt.close()

# Correlation heatmap plot
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), cmap="Greys")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("notebooks/figures/correlation_heatmap.png")
plt.close()

# Confusion matrix (simple classification model) plot
X = df.drop(columns=["target", "label"])
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Greys")
plt.title("Confusion Matrix (Classification Model)")
plt.tight_layout()
plt.savefig("notebooks/figures/confusion_matrix.png")
plt.close()

# Residuals plot (simple regression model)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
residuals = y_test - y_pred
plt.figure()
sns.histplot(residuals, kde=True, bins=20, color=dark_grey)
plt.title("Residuals Distribution (Regression Model)")
plt.xlabel("Residuals (y_true - y_pred)")
plt.tight_layout()
plt.savefig("notebooks/figures/residuals_plot.png")
plt.close()
