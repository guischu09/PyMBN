from typing import Tuple

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .base_classes import Setup


def multiple_comparison_correction(weights, pvalue, setup) -> Tuple[np.ndarray, np.ndarray]:
    """Correct the edge p-values for multiple comparisons and zero non-significant edges.

    Allowed corrections are described in:
    https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
    """
    rows, cols = np.shape(weights)
    iu = np.triu_indices(rows, k=1)

    _, qvalue_tri, _, _ = multipletests(pvalue[iu], method=setup.correction_type)

    qvalue_corrected = np.zeros((rows, cols))
    qvalue_corrected[iu] = qvalue_tri
    qvalue_corrected = qvalue_corrected + qvalue_corrected.T

    significant = qvalue_corrected < setup.alpha
    np.fill_diagonal(significant, False)

    weights_corrected = np.where(significant, weights, 0.0) + np.eye(rows)

    return weights_corrected, qvalue_corrected


def threshold_correction(
    weights_corrected: np.ndarray, pvalue_corrected: np.ndarray, setup: Setup
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove edges whose absolute weight is below setup.threshold.
    """
    weights_corrected = np.array(weights_corrected, copy=True)
    pvalue_corrected = np.array(pvalue_corrected, copy=True)

    logic_mat = np.abs(weights_corrected) < setup.threshold
    np.fill_diagonal(logic_mat, False)

    weights_corrected[logic_mat] = 0
    pvalue_corrected[logic_mat] = 1
    return weights_corrected, pvalue_corrected


def compute_network_weights(data: np.ndarray, setup: Setup) -> Tuple[np.ndarray, np.ndarray]:
    if setup.weights == "pearson_correlation":
        coeff, pvalue = pearson_correlation(data)
    else:
        raise ValueError("Invalid weight type")

    return coeff, pvalue


def pearson_correlation(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Pearson correlation between columns, with two-sided t-test p-values."""
    n_rows, n_cols = array.shape

    # Standardize the data
    standardized = (array - array.mean(axis=0)) / array.std(axis=0, ddof=1)

    # Compute correlation matrix
    correlations = (standardized.T @ standardized) / (n_rows - 1)
    correlations = np.clip(correlations, -1.0, 1.0)

    # Compute t-statistic and p-values on the off-diagonal entries only.
    off_diag = ~np.eye(n_cols, dtype=bool)
    r = correlations[off_diag]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = r * np.sqrt((n_rows - 2) / (1 - r**2))
    pvalues = np.ones((n_cols, n_cols))
    pvalues[off_diag] = 2 * stats.t.sf(np.abs(t_stat), n_rows - 2)

    # Fix diagonal values
    np.fill_diagonal(correlations, 1.0)
    np.fill_diagonal(pvalues, 1.0)

    return correlations, pvalues
