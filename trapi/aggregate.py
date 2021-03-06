def aggregate(df, by, target, methods, renamed_columns=None, as_index=False):
    """
    Function to aggregate columns by target column with target methods.
    You can select methods from next:
        "sum", "max", "min", "mean", "count", "std", "var", "median", "prod", "corr"

    Args:
        df (pd.DataFrame): Dataframe to aggregate.
        by (str or list(str)): Target columns aggregated by.
        target (str): Target column to aggregate.
        methods (list(str)): Methods for aggregation.
        renamed_columns (list(str), optional): Renamed columns after aggregated. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe with aggregated columns.
    """
    assert type(methods) == list, f"type of methods, {type(methods)}, is not list"
    if renamed_columns is not None:
        assert (
            type(renamed_columns) == list
        ), f"type of renamed_columns: {type(renamed_columns)} is not list"
        assert len(methods) == len(
            renamed_columns
        ), f"length of methods: {len(methods)} is different from that of renamed_columns: {len(renamed_columns)}"
    df_agg = df.groupby(by)[target].agg(methods)
    if renamed_columns is not None:
        df_agg.columns = renamed_columns
    if as_index:
        df_agg = df_agg.reset_index()
    return df_agg
