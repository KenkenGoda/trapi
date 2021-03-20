from typing import Any, Dict

import lightgbm as lgb
import pandas as pd
from optuna.integration import lightgbm as lgb_tuner


def train_with_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    params: Dict[str, Any],
    tune: bool = False,
    **kwargs
) -> lgb.Booster:
    """
    Function to train lightgbm model.

    Args:
        X_train (pd.DataFrame): Training Data.
        y_train (pd.Series): Target for train.
        X_valid (pd.DataFrame): Validation Data.
        y_valid (pd.Series): Target for validation.
        params (Dict[str, Any]): LightGBM parameters.
        tune (bool, optional): If run tuning or not. Defaults to False.

    Returns:
        [lgb.Booster]: Trained model.
    """
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid)

    if not tune:
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=["train", "valid"],
            **kwargs,
        )
    else:
        model = lgb_tuner.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=["train", "valid"],
            **kwargs,
        )
    return model
