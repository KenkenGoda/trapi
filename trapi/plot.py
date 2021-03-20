from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_feature_importance(
    df: pd.DataFrame,
    importance_type: str = "split",
    n_plot: int = 20,
    figsize: Tuple[int] = (10, 10),
):
    """
    Function to plot feature importance.

    Args:
        df (pd.DataFrame): Dataframe with ["feature", "split", "gain", "fold"] columns.
        importance_type (str, optional): "split" or "gain". Defaults to "split".
        n_plot (int, optional): Number of features to plot. Defaults to 20.
        figsize (Tuple[int], optional): Figure size. Defaults to (10, 10).
    """
    cols = (
        df.groupby("feature")
        .mean()
        .sort_values(importance_type, ascending=False)
        .head(n_plot)
        .index
    )
    plot_df = df.loc[df["feature"].isin(cols)].sort_values(
        importance_type, ascending=False
    )
    plt.figure(figsize=figsize)
    sns.barplot(x=importance_type, y="feature", data=plot_df)
    plt.show()


def plot_prediction_distribution(
    valid_pred: np.array, test_pred: np.array, figsize: Tuple[int] = (7, 5)
):
    """
    Function to plot prediction distribution.

    Args:
        valid_pred (np.array): Predicted values of validation.
        test_pred (np.array): Predicted values of test.
        figsize (Tuple[int], optional): Figure size. Defaults to (7, 5).
    """
    plt.figure(figsize=figsize)
    sns.histplot(valid_pred, stat="density", color="blue", alpha=0.3, label="valid")
    sns.histplot(test_pred, stat="density", color="red", alpha=0.3, label="test")
    plt.legend()
    plt.grid()
    plt.show()
