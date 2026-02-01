"""
Data loading module for fraud detection dataset.
"""
import pandas as pd
from pathlib import Path
from typing import Tuple


def load_raw_data(data_path: str = "data/raw/creditcard.csv") -> pd.DataFrame:
    """
    Load raw credit card transaction data.

    Args:
        data_path: Path to the raw CSV file

    Returns:
        DataFrame containing the raw data
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(path)
    return df


def load_processed_data(
    train_path: str = "data/processed/train.csv",
    val_path: str = "data/processed/val.csv",
    test_path: str = "data/processed/test.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed train, validation, and test datasets.

    Args:
        train_path: Path to training data
        val_path: Path to validation data
        test_path: Path to test data

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df
