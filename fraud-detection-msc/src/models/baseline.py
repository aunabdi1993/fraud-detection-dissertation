"""
Baseline model implementations.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any


def get_baseline_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Get dictionary of baseline models with default hyperparameters.

    Args:
        random_state: Random seed for reproducibility

    Returns:
        Dictionary mapping model names to model instances
    """
    models = {
        'logistic_regression': LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver='lbfgs'
        ),
        'decision_tree': DecisionTreeClassifier(
            random_state=random_state,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10
        ),
        'random_forest': RandomForestClassifier(
            random_state=random_state,
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            n_jobs=-1
        )
    }

    return models


class BaselineModelTrainer:
    """Trainer for baseline models."""

    def __init__(self, model_name: str, random_state: int = 42):
        """
        Initialize trainer with specified model.

        Args:
            model_name: Name of baseline model to use
            random_state: Random seed
        """
        self.model_name = model_name
        self.random_state = random_state
        self.models = get_baseline_models(random_state)

        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        self.model = self.models[model_name]

    def train(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict_proba(X)
