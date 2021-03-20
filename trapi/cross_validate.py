from typing import Any, Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

from trapi.train import train_with_lightgbm


def cross_validate(
    cv: BaseCrossValidator,
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict[str, Any],
    groups: pd.Series = None,
    tune: bool = False,
    **kwargs,
) -> (List[lgb.Booster], pd.DataFrame, pd.DataFrame):
    """
    Function to run cross-validation.

    Args:
        cv (BaseCrossValidator): Cross-validation generator.
        X (pd.DataFrame): Training data.
        y (pd.Series): Target.
        params (Dict(str, Any)): LightGBM parameters.
        groups (pd.Series, optional): Group labels for the samples. Defaults to None.
        tune (bool, optional): If run tuning or not. Defaults to False.

    Returns:
        List(lgb.Booster): List of trained lightgbm boosters.
        pd.DataFrame: Dataframe with ["true", "pred"] columns, which use for model evaluation.
        pd.DataFrame: Dataframe with ["feature", "split", "gain", "fold"] columns, which use for feature importance plot.
    """
    models = []
    y_true = np.array([])
    y_pred = np.array([])
    imp_df = pd.DataFrame()

    for i, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups=groups)):
        fold = i + 1
        print("--------------------------------------------------")
        print(f"Fold: {fold}/{cv.get_n_splits()}")

        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = train_with_lightgbm(
            X_train, y_train, X_valid, y_valid, params, tune=tune, **kwargs
        )

        models.append(model)
        y_true = np.concatenate([y_true, y_valid])
        y_pred = np.concatenate([y_pred, model.predict(X_valid)])

        _df = pd.DataFrame()
        _df["feature"] = model.feature_name()
        _df["split"] = model.feature_importance("split")
        _df["gain"] = model.feature_importance("gain")
        _df["fold"] = fold
        imp_df = pd.concat([imp_df, _df])
    print("--------------------------------------------------")

    eval_df = pd.DataFrame({"true": y_true, "pred": y_pred})
    imp_df = imp_df.reset_index(drop=True)

    return models, eval_df, imp_df
