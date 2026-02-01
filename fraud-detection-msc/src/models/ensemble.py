"""
Ensemble models for fraud detection.
"""
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier, RUSBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Any


def get_ensemble_models(random_state: int = 42, scale_pos_weight: float = None) -> Dict[str, Any]:
    """
    Get dictionary of ensemble models.

    Args:
        random_state: Random seed for reproducibility
        scale_pos_weight: Weight for positive class (for XGBoost/LightGBM)

    Returns:
        Dictionary mapping model names to model instances
    """
    # Calculate scale_pos_weight if not provided
    # Typically: (# negative samples) / (# positive samples)
    if scale_pos_weight is None:
        scale_pos_weight = 1.0

    models = {
        'balanced_random_forest': BalancedRandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            max_depth=10,
            n_jobs=-1
        ),
        'easy_ensemble': EasyEnsembleClassifier(
            n_estimators=10,
            random_state=random_state,
            n_jobs=-1
        ),
        'rus_boost': RUSBoostClassifier(
            n_estimators=100,
            random_state=random_state
        ),
        'xgboost': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        ),
        'lightgbm': lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            is_unbalance=True,
            random_state=random_state,
            verbose=-1
        )
    }

    return models


class EnsembleModelTrainer:
    """Trainer for ensemble models."""

    def __init__(self, model_name: str, random_state: int = 42, scale_pos_weight: float = None):
        """
        Initialize trainer with specified ensemble model.

        Args:
            model_name: Name of ensemble model to use
            random_state: Random seed
            scale_pos_weight: Weight for positive class
        """
        self.model_name = model_name
        self.random_state = random_state
        self.scale_pos_weight = scale_pos_weight
        self.models = get_ensemble_models(random_state, scale_pos_weight)

        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        self.model = self.models[model_name]

    def train(self, X_train, y_train):
        """Train the ensemble model."""
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict_proba(X)
