"""
Evaluation metrics for fraud detection models.
"""
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import numpy as np
from typing import Dict, Tuple


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate comprehensive set of evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities for positive class

    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }

    # Add probability-based metrics if available
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    })

    # Additional metrics
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['fall_out'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return metrics


def find_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray, metric: str = 'f1') -> Tuple[float, float]:
    """
    Find optimal classification threshold based on specified metric.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall')

    Returns:
        Tuple of (optimal_threshold, metric_value)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    if metric == 'f1':
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_value = f1_scores[optimal_idx]
    elif metric == 'precision':
        optimal_idx = np.argmax(precisions)
        optimal_value = precisions[optimal_idx]
    elif metric == 'recall':
        optimal_idx = np.argmax(recalls)
        optimal_value = recalls[optimal_idx]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

    return optimal_threshold, optimal_value


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Generate classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Classification report as string
    """
    return classification_report(y_true, y_pred, target_names=['Legitimate', 'Fraud'])
