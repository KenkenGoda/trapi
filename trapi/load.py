import pandas as pd

from utils import logger, reduce_mem_usage


def load_to_df(path: str, **kwargs) -> pd.DataFrame:
    """
    Function to load file to pandas dataframe.

    Args:
        path (str): File path such as CSV.

    Returns:
        pd.DataFrame: Dataframe loaded specified file.
    """
    df = pd.read_csv(path, **kwargs)
    logger(f"Dataframe shape: {df.shape}")
    return reduce_mem_usage(df)
