import lightgbm as lgb
import optuna.integration.lightgbm as lgb_tuner


def train_with_lightgbm(
    X_train, y_train, X_valid, y_valid, params, tune=False, **kwargs
):
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
        model = lgb_tuner(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=["train", "valid"],
            **kwargs,
        )
    return model
