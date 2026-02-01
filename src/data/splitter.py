"""
Data splitting module for train/val/test splits.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify_col: str = 'Class'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets.

    Args:
        df: Input DataFrame
        test_size: Proportion of data for test set
        val_size: Proportion of remaining data for validation set
        random_state: Random seed for reproducibility
        stratify_col: Column name to use for stratified splitting

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # First split: separate test set
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[stratify_col]
    )

    # Second split: separate validation set from training
    val_proportion = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_proportion,
        random_state=random_state,
        stratify=train_val[stratify_col]
    )

    return train, val, test


def get_X_y(
    df: pd.DataFrame,
    target_col: str = 'Class',
    drop_cols: list = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target variable.

    Args:
        df: Input DataFrame
        target_col: Name of target column
        drop_cols: Additional columns to drop from features

    Returns:
        Tuple of (X, y)
    """
    if drop_cols is None:
        drop_cols = []

    y = df[target_col]
    X = df.drop(columns=[target_col] + drop_cols)

    return X, y
