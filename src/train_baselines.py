# src/train_baselines.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

from data import load_diabetes_df, add_class_label

# Data loading
df = load_diabetes_df()
df, median_y = add_class_label(df)

# Separate features and targets
X = df.drop(columns=["target", "label"])
y_reg = df["target"]
y_clf = df["label"]

# We use (70% / 15% / 15%) Train–Validation–Test Split 

X_train, X_temp, y_train_reg, y_temp_reg = train_test_split(X, y_reg, test_size=0.3, random_state=42)
X_val, X_test, y_val_reg, y_test_reg = train_test_split(X_temp, y_temp_reg, test_size=0.5, random_state=42)

# For classification target
_, _, y_train_clf, y_temp_clf = train_test_split(X, y_clf, test_size=0.3, random_state=42)
_, _, y_val_clf, y_test_clf = train_test_split(X_temp, y_temp_clf, test_size=0.5, random_state=42)

print(f"Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}")

## Regression Models

regression_results = []

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train_reg)
y_pred_val = lin_reg.predict(X_val)
y_pred_test = lin_reg.predict(X_test)
mse_val_lr = mean_squared_error(y_val_reg, y_pred_val)
mse_test_lr = mean_squared_error(y_test_reg, y_pred_test)
regression_results.append(["Linear Regression", mse_val_lr, mse_test_lr])

# Decision Tree Regressor
tree_reg = DecisionTreeRegressor(random_state=42, max_depth=4)
tree_reg.fit(X_train, y_train_reg)
y_pred_val = tree_reg.predict(X_val)
y_pred_test = tree_reg.predict(X_test)
mse_val_tr = mean_squared_error(y_val_reg, y_pred_val)
mse_test_tr = mean_squared_error(y_test_reg, y_pred_test)
regression_results.append(["Decision Tree Regressor", mse_val_tr, mse_test_tr])

# Save regression results
reg_df = pd.DataFrame(regression_results, columns=["Model", "Val MSE", "Test MSE"])
reg_df.to_csv("notebooks/tables/regression_results.csv", index=False)

## Classification Models
classification_results = []

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train_clf)
y_pred_val = log_reg.predict(X_val)
y_pred_test = log_reg.predict(X_test)
acc_val_lr = accuracy_score(y_val_clf, y_pred_val)
acc_test_lr = accuracy_score(y_test_clf, y_pred_test)
classification_results.append(["Logistic Regression", acc_val_lr, acc_test_lr])

# Decision Tree Classifier
tree_clf = DecisionTreeClassifier(random_state=42, max_depth=4)
tree_clf.fit(X_train, y_train_clf)
y_pred_val = tree_clf.predict(X_val)
y_pred_test = tree_clf.predict(X_test)
acc_val_tr = accuracy_score(y_val_clf, y_pred_val)
acc_test_tr = accuracy_score(y_test_clf, y_pred_test)
classification_results.append(["Decision Tree Classifier", acc_val_tr, acc_test_tr])

# Classification Results
clf_df = pd.DataFrame(classification_results, columns=["Model", "Val Accuracy", "Test Accuracy"])
clf_df.to_csv("notebooks/tables/classification_results.csv", index=False)

print("All results saved to notebooks/tables/.")

