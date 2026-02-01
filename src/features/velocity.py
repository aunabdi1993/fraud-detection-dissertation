"""
Transaction velocity features for fraud detection.
"""
import pandas as pd
import numpy as np


def create_velocity_features(df: pd.DataFrame, windows: list = [3600, 7200, 86400]) -> pd.DataFrame:
    """
    Create transaction velocity features.

    Velocity features capture the rate of transactions and spending patterns.

    Args:
        df: DataFrame with Time and Amount columns
        windows: List of time windows (in seconds) for velocity calculation

    Returns:
        DataFrame with additional velocity features
    """
    df = df.copy()
    df = df.sort_values('Time')

    for window in windows:
        window_hours = window / 3600
        col_suffix = f"_{int(window_hours)}h"

        # Transaction count velocity
        df[f'tx_count{col_suffix}'] = (
            df.groupby(pd.cut(df['Time'], bins=int(df['Time'].max() / window)))['Time']
            .transform('count')
        )

        # Amount velocity (total spending in window)
        df[f'amount_velocity{col_suffix}'] = (
            df.groupby(pd.cut(df['Time'], bins=int(df['Time'].max() / window)))['Amount']
            .transform('sum')
        )

        # Average transaction size in window
        df[f'avg_tx_size{col_suffix}'] = (
            df.groupby(pd.cut(df['Time'], bins=int(df['Time'].max() / window)))['Amount']
            .transform('mean')
        )

    # Time between transactions
    df['time_diff'] = df['Time'].diff()
    df['time_diff'] = df['time_diff'].fillna(df['time_diff'].median())

    return df
