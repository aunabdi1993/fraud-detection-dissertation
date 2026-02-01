"""
Data preprocessing and validation module.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple


class DataPreprocessor:
    """Handles data cleaning, validation, and preprocessing."""

    def __init__(self):
        self.scaler = StandardScaler()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and duplicates.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        # Remove duplicates
        df = df.drop_duplicates()

        # Handle missing values
        df = df.dropna()

        return df

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate dataset structure and content.

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, raises ValueError otherwise
        """
        # Check for required columns
        required_cols = ['Time', 'Amount', 'Class']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Need: {required_cols}")

        # Check for valid class labels
        valid_classes = {0, 1}
        if not set(df['Class'].unique()).issubset(valid_classes):
            raise ValueError("Class column must contain only 0 and 1")

        return True

    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame = None,
        X_test: pd.DataFrame = None
    ) -> Tuple[np.ndarray, ...]:
        """
        Scale features using StandardScaler fitted on training data.

        Args:
            X_train: Training features
            X_val: Validation features (optional)
            X_test: Test features (optional)

        Returns:
            Tuple of scaled arrays
        """
        X_train_scaled = self.scaler.fit_transform(X_train)

        results = [X_train_scaled]

        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            results.append(X_val_scaled)

        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            results.append(X_test_scaled)

        return tuple(results)
