"""
Diagnostic tools for assessing population adjustment quality.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def calculate_balance_metrics(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    covariates: List[str],
    weights1: Optional[np.ndarray] = None,
    weights2: Optional[np.ndarray] = None,
    categorical_vars: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate comprehensive balance metrics between two datasets.

    Parameters
    ----------
    data1 : pd.DataFrame
        First dataset
    data2 : pd.DataFrame
        Second dataset
    covariates : List[str]
        Variables to assess balance on
    weights1 : Optional[np.ndarray]
        Weights for first dataset
    weights2 : Optional[np.ndarray]
        Weights for second dataset
    categorical_vars : Optional[List[str]]
        List of categorical variables

    Returns
    -------
    pd.DataFrame
        Balance metrics including SMD, variance ratios, KS statistics
    """
    if categorical_vars is None:
        categorical_vars = []

    balance_results = []

    for var in covariates:
        x1 = data1[var].values
        x2 = data2[var].values

        # Calculate means
        if weights1 is not None:
            mean1 = np.average(x1, weights=weights1)
            var1 = np.average((x1 - mean1)**2, weights=weights1)
        else:
            mean1 = np.mean(x1)
            var1 = np.var(x1, ddof=1)

        if weights2 is not None:
            mean2 = np.average(x2, weights=weights2)
            var2 = np.average((x2 - mean2)**2, weights=weights2)
        else:
            mean2 = np.mean(x2)
            var2 = np.var(x2, ddof=1)

        # Standardized mean difference
        pooled_sd = np.sqrt((var1 + var2) / 2)
        if pooled_sd > 0:
            smd = (mean1 - mean2) / pooled_sd
        else:
            smd = 0.0

        # Variance ratio
        if var2 > 0:
            var_ratio = var1 / var2
        else:
            var_ratio = np.nan

        # Kolmogorov-Smirnov statistic (for distributional balance)
        if var not in categorical_vars:
            ks_stat, ks_pval = ks_2samp(x1, x2)
        else:
            ks_stat, ks_pval = np.nan, np.nan

        balance_results.append({
            'variable': var,
            'mean_group1': mean1,
            'mean_group2': mean2,
            'smd': smd,
            'var_ratio': var_ratio,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'balanced': abs(smd) < 0.1  # Common threshold
        })

    return pd.DataFrame(balance_results)


def assess_overlap(
    propensity_scores_1: np.ndarray,
    propensity_scores_2: np.ndarray,
    n_bins: int = 20
) -> Dict:
    """
    Assess propensity score overlap between two groups.

    Parameters
    ----------
    propensity_scores_1 : np.ndarray
        Propensity scores for group 1
    propensity_scores_2 : np.ndarray
        Propensity scores for group 2
    n_bins : int
        Number of bins for histogram

    Returns
    -------
    Dict
        Overlap metrics and histograms
    """
    # Calculate overlap coefficient
    # Bins
    bins = np.linspace(0, 1, n_bins + 1)
    hist1, _ = np.histogram(propensity_scores_1, bins=bins, density=True)
    hist2, _ = np.histogram(propensity_scores_2, bins=bins, density=True)

    # Overlap coefficient (area of intersection)
    overlap = np.sum(np.minimum(hist1, hist2)) / n_bins

    # Common support (proportion in overlapping range)
    min_ps1, max_ps1 = np.min(propensity_scores_1), np.max(propensity_scores_1)
    min_ps2, max_ps2 = np.min(propensity_scores_2), np.max(propensity_scores_2)

    common_min = max(min_ps1, min_ps2)
    common_max = min(max_ps1, max_ps2)

    if common_max > common_min:
        prop_overlap_1 = np.mean(
            (propensity_scores_1 >= common_min) & (propensity_scores_1 <= common_max)
        )
        prop_overlap_2 = np.mean(
            (propensity_scores_2 >= common_min) & (propensity_scores_2 <= common_max)
        )
    else:
        prop_overlap_1 = 0.0
        prop_overlap_2 = 0.0

    return {
        'overlap_coefficient': float(overlap),
        'common_support_range': (common_min, common_max),
        'proportion_in_support_group1': float(prop_overlap_1),
        'proportion_in_support_group2': float(prop_overlap_2),
        'ps_range_group1': (min_ps1, max_ps1),
        'ps_range_group2': (min_ps2, max_ps2)
    }


def sensitivity_analysis(
    adjustment_method,
    ipd_data: pd.DataFrame,
    aggregate_data: pd.DataFrame,
    covariates: List[str],
    outcome: str,
    treatment: str,
    unmeasured_confounder_effect: np.ndarray = np.array([0.5, 1.0, 1.5, 2.0]),
    unmeasured_confounder_prevalence: np.ndarray = np.array([0.1, 0.2, 0.3])
) -> pd.DataFrame:
    """
    Perform sensitivity analysis for unmeasured confounding.

    Implements methods similar to those in VanderWeele & Ding (2017).

    Parameters
    ----------
    adjustment_method : PopulationAdjustmentMethod
        Fitted adjustment method
    ipd_data : pd.DataFrame
        IPD data
    aggregate_data : pd.DataFrame
        Aggregate/target data
    covariates : List[str]
        Covariates
    outcome : str
        Outcome variable
    treatment : str
        Treatment variable
    unmeasured_confounder_effect : np.ndarray
        Range of hypothetical odds ratios for unmeasured confounder
    unmeasured_confounder_prevalence : np.ndarray
        Range of prevalence values for unmeasured confounder

    Returns
    -------
    pd.DataFrame
        Sensitivity analysis results showing adjusted effects under
        different unmeasured confounding scenarios
    """
    # Get baseline result
    base_result = adjustment_method.result

    if base_result is None:
        raise ValueError("Method must be fitted before sensitivity analysis")

    results = []

    for or_u in unmeasured_confounder_effect:
        for prev_u in unmeasured_confounder_prevalence:
            # Approximate bias due to unmeasured confounder
            # Simplified approach based on confounding bias formulas

            # Bias factor (simplified)
            # This is a rough approximation; more sophisticated methods exist
            bias = np.log(or_u) * prev_u * (1 - prev_u)

            # Adjusted effect
            adjusted_effect = base_result.effect_estimate - bias

            results.append({
                'unmeasured_or': or_u,
                'unmeasured_prevalence': prev_u,
                'bias': bias,
                'adjusted_effect': adjusted_effect,
                'original_effect': base_result.effect_estimate,
                'ci_lower_adjusted': base_result.ci_lower - bias,
                'ci_upper_adjusted': base_result.ci_upper - bias
            })

    return pd.DataFrame(results)


def effective_sample_size_loss(
    weights: np.ndarray,
    original_n: Optional[int] = None
) -> Dict:
    """
    Calculate effective sample size and information loss.

    Parameters
    ----------
    weights : np.ndarray
        Weight vector
    original_n : Optional[int]
        Original sample size (if None, uses len(weights))

    Returns
    -------
    Dict
        ESS metrics and percent loss
    """
    if original_n is None:
        original_n = len(weights)

    # Effective sample size
    ess = np.sum(weights)**2 / np.sum(weights**2)

    # Percent loss
    percent_loss = (1 - ess / original_n) * 100

    # Entropy of weights (measure of uniformity)
    w_norm = weights / np.sum(weights)
    entropy = -np.sum(w_norm * np.log(w_norm + 1e-10))
    max_entropy = np.log(len(weights))
    relative_entropy = entropy / max_entropy

    return {
        'effective_sample_size': float(ess),
        'original_sample_size': original_n,
        'percent_loss': float(percent_loss),
        'entropy': float(entropy),
        'relative_entropy': float(relative_entropy)
    }


def check_positivity(
    data: pd.DataFrame,
    treatment: str,
    covariates: List[str],
    min_cell_size: int = 5
) -> Dict:
    """
    Check positivity assumption (overlap in covariate distributions).

    Parameters
    ----------
    data : pd.DataFrame
        Data
    treatment : str
        Treatment variable
    covariates : List[str]
        Covariates to check
    min_cell_size : int
        Minimum cell size for cross-tabulation

    Returns
    -------
    Dict
        Positivity violations and warnings
    """
    violations = []
    warnings_list = []

    trt = data[treatment].values

    # Check each covariate
    for cov in covariates:
        x = data[cov].values

        # Check if categorical (small number of unique values)
        unique_vals = np.unique(x)

        if len(unique_vals) <= 10:  # Treat as categorical
            for val in unique_vals:
                n_trt = np.sum((x == val) & (trt == 1))
                n_ctrl = np.sum((x == val) & (trt == 0))

                if n_trt < min_cell_size or n_ctrl < min_cell_size:
                    violations.append({
                        'covariate': cov,
                        'value': val,
                        'n_treated': int(n_trt),
                        'n_control': int(n_ctrl),
                        'violation_type': 'sparse_cell'
                    })

        else:  # Continuous - check for extreme values
            # Check if any extreme values lack overlap
            q_low, q_high = np.percentile(x, [5, 95])

            # Check extremes in treatment group
            extreme_low_trt = x[(trt == 1) & (x < q_low)]
            extreme_high_trt = x[(trt == 1) & (x > q_high)]

            if len(extreme_low_trt) > 0:
                warnings_list.append({
                    'covariate': cov,
                    'warning': 'Extreme low values in treatment group',
                    'n_extreme': len(extreme_low_trt)
                })

            if len(extreme_high_trt) > 0:
                warnings_list.append({
                    'covariate': cov,
                    'warning': 'Extreme high values in treatment group',
                    'n_extreme': len(extreme_high_trt)
                })

    return {
        'violations': violations,
        'warnings': warnings_list,
        'positivity_satisfied': len(violations) == 0
    }
