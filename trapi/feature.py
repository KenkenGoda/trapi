from typing import List

import pandas as pd
from sklearn.preprocess import LabelEncoder


class BaseBlock:
    """
    Base block class for feature generator.
    This has fit and transform functions, like machine learning flow, as default.
    You can make block class as below:
    ---------------------------------------------------------------------------
    AgeAggregatedBlock(BaseBlock):
        def __init__(self, column, aggs, fill_value=0):
            self.column = column
            self.aggs = aggs
            self.fill_value = fill_value

        def fit(self, input_df, y=None):
            self.agg_df_ = input_df.groupby(self.column)["age"].agg(self.aggs)
            return self.transform(input_df)

        def transform(self, input_df):
            out_df = (
                input_df[[self.column]]
                .merge(self.agg_df_, on=self.column, how="left")
                .fillna(self.fill_value)
                .drop(columns=self.column)
            )
            return out_df.add_prefix(f"{self.column}_age_")
    """

    def fit(self, input_df: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> NotImplementedError:
        raise NotImplementedError()


class RawValueBlock(BaseBlock):
    def __init__(self, column: str):
        self.column = column

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        return input_df[[self.column]]


class CountEncodingBlock(BaseBlock):
    def __init__(self, column: str, all_df: pd.DataFrame = None):
        self.column = column
        self.all_df = all_df

    def fit(self, input_df: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        if self.all_df is not None:
            _df = self.all_df
        else:
            _df = input_df
        self.vc_ = _df[self.column].value_counts()
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        out_df = pd.DataFrame()
        out_df[self.column] = input_df[self.column].map(self.vc_)
        return out_df.add_prefix("CE_")


class LabelEncodingBlock(BaseBlock):
    def __init__(self, column: str):
        self.column = column
        self.encoder = LabelEncoder()

    def fit(self, input_df: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        self.encoder.fit(input_df[self.column].values)
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        out_df = pd.DataFrame()
        out_df[self.column] = self.encoder.transform(input_df[self.column].values)
        return out_df.add_prefix("LE_")


class TargetEncodingBlock(BaseBlock):
    def __init__(
        self,
        column: str,
        target: str,
        aggs: List[str] = ["mean", "max", "min", "std", "sum"],
        fill_value: float = None,
    ):
        self.column = column
        self.target = target
        self.aggs = aggs
        self.fill_value = fill_value

    def fit(self, input_df: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        _df = pd.concat([input_df[self.column], y], axis=1)
        _agg_df = _df.groupby(self.column)[self.target].agg(self.aggs)
        if "max" in self.aggs and "min" in self.aggs:
            _agg_df["max-min"] = _agg_df["max"] - _agg_df["min"]
            _agg_df["max/min"] = _agg_df["max"] / (_agg_df["min"] + 1e-9)
        if "mean" in self.aggs and "std" in self.aggs:
            _agg_df["mean/std"] = _agg_df["mean"] / (_agg_df["std"].fillna(0) + 1e-9)
        self.agg_df_ = _agg_df
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        out_df = (
            input_df[[self.column]]
            .merge(self.agg_df_, on=self.column, how="left")
            .drop(columns=self.column)
        )
        if self.fill_value is not None:
            out_df = out_df.fillna(self.fill_value)
        return out_df.add_prefix(f"TE_{self.column}_")


class AggregationBlock(BaseBlock):
    def __init__(
        self,
        column: str,
        target: str,
        aggs: List[str] = ["mean", "max", "min", "std", "sum"],
        fill_value: float = None,
        all_df: pd.DataFrame = None,
    ):
        self.column = column
        self.target = target
        self.aggs = aggs
        self.fill_value = fill_value
        self.all_df = all_df

    def fit(self, input_df: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        if self.all_df is not None:
            _df = self.all_df
        else:
            _df = input_df
        _agg_df = _df.groupby(self.column)[self.target].agg(self.aggs)
        if "max" in self.aggs and "min" in self.aggs:
            _agg_df["max-min"] = _agg_df["max"] - _agg_df["min"]
            _agg_df["max/min"] = _agg_df["max"] / (_agg_df["min"] + 1e-9)
        if "mean" in self.aggs and "std" in self.aggs:
            _agg_df["mean/std"] = _agg_df["mean"] / (_agg_df["std"].fillna(0) + 1e-9)
        self.agg_df_ = _agg_df
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        out_df = (
            input_df[[self.column]]
            .merge(self.agg_df_, on=self.column, how="left")
            .drop(columns=self.column)
        )
        if self.fill_value is not None:
            out_df = out_df.fillna(self.fill_value)
        return out_df.add_prefix(f"{self.column}_{self.target}_")
