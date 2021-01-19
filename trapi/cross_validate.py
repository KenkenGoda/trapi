import numpy as np
import pandas as pd

from trapi.train import train_with_lightgbm


def cross_validate(cv, X, y, params, tune=False, **kwargs):
    models = []
    y_true = np.array([])
    y_pred = np.array([])
    df_imp = pd.DataFrame()

    for i, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
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
        df_imp = pd.concat([df_imp, _df])
    print("--------------------------------------------------")

    return models, y_true, y_pred, df_imp
