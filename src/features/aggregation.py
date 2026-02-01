"""
Aggregation features for fraud detection.
"""
import pandas as pd
import numpy as np


def create_aggregation_features(df: pd.DataFrame, windows: list = [5, 10, 20]) -> pd.DataFrame:
    """
    Create rolling aggregation features.

    Args:
        df: DataFrame with transaction features
        windows: List of window sizes for rolling calculations

    Returns:
        DataFrame with additional aggregation features
    """
    df = df.copy()
    df = df.sort_values('Time')

    for window in windows:
        # Rolling statistics for Amount
        df[f'amount_rolling_mean_{window}'] = df['Amount'].rolling(window=window, min_periods=1).mean()
        df[f'amount_rolling_std_{window}'] = df['Amount'].rolling(window=window, min_periods=1).std()
        df[f'amount_rolling_max_{window}'] = df['Amount'].rolling(window=window, min_periods=1).max()
        df[f'amount_rolling_min_{window}'] = df['Amount'].rolling(window=window, min_periods=1).min()

        # Fill NaN values
        df[f'amount_rolling_std_{window}'] = df[f'amount_rolling_std_{window}'].fillna(0)

    # Ratio of current amount to rolling mean
    df['amount_to_mean_ratio'] = df['Amount'] / (df['amount_rolling_mean_5'] + 1e-5)

    return df
