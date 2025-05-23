import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# function to display missing values
def missing_values_visualization(df: pd.DataFrame, title: str, figsize: tuple = (12,6)) -> None:
    plt.figure(figsize=(12, 6))

    counts = []
    for col in df.columns:
        counts.append({'column': col, 'status': 'Existing', 'count': df[col].notnull().sum()})
        counts.append({'column': col, 'status': 'Missing', 'count': df[col].isna().sum()})
    counts_df = pd.DataFrame(counts)

    plt.figure(figsize=figsize)
    sns.barplot(data=counts_df, x='column', y='count', hue='status', edgecolor='black')
    plt.title(title)
    plt.xticks(rotation=90)

    plt.show()


# plotting the confusion matrix
def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray, title: str, figsize: tuple = (10, 7)) -> None:
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()