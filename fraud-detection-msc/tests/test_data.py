"""
Tests for data loading and preprocessing modules.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.preprocessor import DataPreprocessor
from data.splitter import split_data, get_X_y


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""

    def test_clean_data_removes_duplicates(self):
        """Test that clean_data removes duplicate rows."""
        preprocessor = DataPreprocessor()

        # Create sample data with duplicates
        df = pd.DataFrame({
            'Time': [1, 2, 1, 3],
            'Amount': [100, 200, 100, 300],
            'Class': [0, 1, 0, 0]
        })

        cleaned = preprocessor.clean_data(df)

        assert len(cleaned) == 3  # One duplicate removed

    def test_validate_data_checks_required_columns(self):
        """Test that validate_data checks for required columns."""
        preprocessor = DataPreprocessor()

        # Missing required column
        df_invalid = pd.DataFrame({
            'Time': [1, 2],
            'Amount': [100, 200]
            # Missing 'Class' column
        })

        with pytest.raises(ValueError):
            preprocessor.validate_data(df_invalid)

    def test_scale_features(self):
        """Test feature scaling."""
        preprocessor = DataPreprocessor()

        X_train = pd.DataFrame(np.random.randn(100, 5))
        X_test = pd.DataFrame(np.random.randn(20, 5))

        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test=X_test)

        # Check scaled arrays have same shape
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape


class TestDataSplitter:
    """Tests for data splitting functions."""

    def test_split_data_proportions(self):
        """Test that split_data creates correct proportions."""
        df = pd.DataFrame({
            'Feature1': np.random.randn(1000),
            'Feature2': np.random.randn(1000),
            'Class': np.random.choice([0, 1], 1000)
        })

        train, val, test = split_data(df, test_size=0.2, val_size=0.1, random_state=42)

        total = len(train) + len(val) + len(test)
        assert total == len(df)
        assert len(test) / total == pytest.approx(0.2, abs=0.01)

    def test_get_X_y_separates_features_and_target(self):
        """Test that get_X_y correctly separates features and target."""
        df = pd.DataFrame({
            'Feature1': [1, 2, 3],
            'Feature2': [4, 5, 6],
            'Class': [0, 1, 0]
        })

        X, y = get_X_y(df, target_col='Class')

        assert list(X.columns) == ['Feature1', 'Feature2']
        assert y.name == 'Class'
        assert len(X) == len(y) == 3
