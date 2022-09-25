import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int):
    """
    Function to set seed.

    Args
        seed (int): Seed number.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True


def logger(message: str):
    """
    Function to show log.

    Args:
        message (str): Message shown as log.
    """
    now_string = str(datetime.now().strftime("%H:%M:%S"))
    print(f"[{now_string}] - {message}")


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Function to reduce memory usage of numerical columns.

    Args:
        df (pd.DataFrame): Dataframe to reduce memory usage.
        verbose (bool, optional): If show how the memory usage decreased or not. Defaults to True.

    Returns:
        pd.DataFrame: Dataframe decreased memory usage.
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                int8_min = np.iinfo(np.int8).min
                int8_max = np.iinfo(np.int8).max
                int16_min = np.iinfo(np.int16).min
                int16_max = np.iinfo(np.int16).max
                int32_min = np.iinfo(np.int32).min
                int32_max = np.iinfo(np.int32).max
                int64_min = np.iinfo(np.int64).min
                int64_max = np.iinfo(np.int64).max
                if c_min > int8_min and c_max < int8_max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > int16_min and c_max < int16_max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > int32_min and c_max < int32_max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > int64_min and c_max < int64_max:
                    df[col] = df[col].astype(np.int64)
            else:
                float16_min = np.finfo(np.float16).min
                float16_max = np.finfo(np.float16).max
                float32_min = np.finfo(np.float32).min
                float32_max = np.finfo(np.float32).max
                if c_min > float16_min and c_max < float16_max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > float32_min and c_max < float32_max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        ratio = (start_mem - end_mem) / start_mem * 100
        logger(f"Mem. usage decreased to {end_mem:.2f} Mb ({ratio:.1f}% reduction)")
    return df
