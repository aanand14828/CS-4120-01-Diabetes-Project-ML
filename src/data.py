import pandas as pd
from sklearn.datasets import load_diabetes

def load_diabetes_df():
    d = load_diabetes(as_frame=True)
    df = d.frame.copy()
    df["target"] = d.target
    return df

# Here we define a helper function that converts the target regression variable (continuous) into a binary classification label:
# It first finds the median value of the target column.
# And if a row’s target is above or equal to the median, then label = 1 (high progression). Otherwise, label = 0 (low progression).
# It returns both the updated DataFrame and the median value used for splitting.

def add_class_label(df):
    median_y = df["target"].median()
    df = df.assign(label=(df["target"] >= median_y).astype(int))
    return df, median_y

