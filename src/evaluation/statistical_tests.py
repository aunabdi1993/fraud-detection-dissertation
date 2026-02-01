"""
Statistical significance testing for model comparison.
"""
from scipy import stats
import numpy as np
from typing import List, Dict, Tuple


def paired_t_test(scores1: List[float], scores2: List[float], alpha: float = 0.05) -> Dict[str, any]:
    """
    Perform paired t-test to compare two models.

    Args:
        scores1: Performance scores for model 1 (e.g., from cross-validation)
        scores2: Performance scores for model 2
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    if len(scores1) != len(scores2):
        raise ValueError("Score arrays must have same length")

    t_stat, p_value = stats.ttest_rel(scores1, scores2)

    result = {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'is_significant': p_value < alpha,
        'alpha': alpha,
        'mean_diff': float(np.mean(scores1) - np.mean(scores2))
    }

    return result


def wilcoxon_test(scores1: List[float], scores2: List[float], alpha: float = 0.05) -> Dict[str, any]:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Args:
        scores1: Performance scores for model 1
        scores2: Performance scores for model 2
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    if len(scores1) != len(scores2):
        raise ValueError("Score arrays must have same length")

    statistic, p_value = stats.wilcoxon(scores1, scores2)

    result = {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'is_significant': p_value < alpha,
        'alpha': alpha,
        'median_diff': float(np.median(scores1) - np.median(scores2))
    }

    return result


def mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Dict[str, any]:
    """
    Perform McNemar's test for comparing classifier predictions.

    Args:
        y_true: True labels
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2

    Returns:
        Dictionary with test results
    """
    # Create contingency table
    # n01: model1 wrong, model2 correct
    # n10: model1 correct, model2 wrong
    n01 = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
    n10 = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))

    # McNemar's test statistic (with continuity correction)
    statistic = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10) if (n01 + n10) > 0 else 0

    # p-value from chi-square distribution with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    result = {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'n01': int(n01),
        'n10': int(n10),
        'is_significant': p_value < 0.05
    }

    return result


def compare_multiple_models(
    model_scores: Dict[str, List[float]],
    alpha: float = 0.05
) -> Dict[str, Dict[str, any]]:
    """
    Compare multiple models using Friedman test followed by post-hoc analysis.

    Args:
        model_scores: Dictionary mapping model names to score lists
        alpha: Significance level

    Returns:
        Dictionary with comparison results
    """
    model_names = list(model_scores.keys())
    scores_array = np.array([model_scores[name] for name in model_names])

    # Friedman test
    statistic, p_value = stats.friedmanchisquare(*scores_array)

    result = {
        'friedman_statistic': float(statistic),
        'friedman_p_value': float(p_value),
        'is_significant': p_value < alpha,
        'num_models': len(model_names),
        'model_names': model_names
    }

    # If significant, perform pairwise comparisons
    if p_value < alpha:
        pairwise_results = {}
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                name1, name2 = model_names[i], model_names[j]
                test_result = wilcoxon_test(scores_array[i], scores_array[j], alpha)
                pairwise_results[f"{name1}_vs_{name2}"] = test_result

        result['pairwise_comparisons'] = pairwise_results

    return result
