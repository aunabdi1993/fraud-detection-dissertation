"""
Sampling techniques for handling class imbalance.
"""
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from typing import Tuple, Any
import numpy as np


class SamplingHandler:
    """Handles various sampling techniques for class imbalance."""

    def __init__(self, sampling_strategy: str = 'auto', random_state: int = 42):
        """
        Initialize sampling handler.

        Args:
            sampling_strategy: Strategy for resampling ('auto', 'minority', or ratio)
            random_state: Random seed for reproducibility
        """
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.sampler = None

    def get_sampler(self, method: str) -> Any:
        """
        Get sampler instance for specified method.

        Args:
            method: Sampling method name

        Returns:
            Sampler instance
        """
        samplers = {
            'random_oversample': RandomOverSampler(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state
            ),
            'random_undersample': RandomUnderSampler(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state
            ),
            'smote': SMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                k_neighbors=5
            ),
            'adasyn': ADASYN(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                n_neighbors=5
            ),
            'smote_tomek': SMOTETomek(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state
            )
        }

        if method not in samplers:
            raise ValueError(f"Unknown sampling method: {method}")

        return samplers[method]

    def apply_sampling(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply sampling technique to dataset.

        Args:
            X: Feature matrix
            y: Target labels
            method: Sampling method to apply

        Returns:
            Tuple of (resampled X, resampled y)
        """
        self.sampler = self.get_sampler(method)
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)

        return X_resampled, y_resampled
