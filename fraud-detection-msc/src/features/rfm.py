"""
RFM (Recency, Frequency, Monetary) feature engineering.
"""
import pandas as pd
import numpy as np


def create_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create RFM features for fraud detection.

    Args:
        df: DataFrame with Time, Amount, and customer identifier columns

    Returns:
        DataFrame with additional RFM features
    """
    df = df.copy()

    # Sort by time
    df = df.sort_values('Time')

    # Recency: Time since last transaction (if customer ID available)
    # For anonymized data, use time-based windows
    df['time_hour'] = df['Time'] / 3600

    # Frequency: Count of transactions in rolling window
    # Using time-based grouping
    df['freq_last_hour'] = df.groupby(pd.cut(df['Time'], bins=100))['Time'].transform('count')

    # Monetary: Amount statistics
    df['amount_mean'] = df.groupby(pd.cut(df['Time'], bins=100))['Amount'].transform('mean')
    df['amount_std'] = df.groupby(pd.cut(df['Time'], bins=100))['Amount'].transform('std')

    # Amount deviation from mean
    df['amount_deviation'] = np.abs(df['Amount'] - df['amount_mean'])

    return df
