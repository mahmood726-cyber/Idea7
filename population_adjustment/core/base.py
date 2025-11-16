"""
Base classes for population adjustment methods.

This module provides abstract base classes and result containers for all
population adjustment methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class AdjustmentResult:
    """
    Container for population adjustment results.

    Attributes
    ----------
    effect_estimate : float
        The adjusted treatment effect estimate
    standard_error : float
        Standard error of the effect estimate
    ci_lower : float
        Lower bound of confidence/credible interval
    ci_upper : float
        Upper bound of confidence/credible interval
    method_name : str
        Name of the adjustment method used
    diagnostics : Dict
        Dictionary containing diagnostic information:
        - effective_sample_size: Effective sample size after weighting
        - max_weight: Maximum weight value
        - weight_cv: Coefficient of variation of weights
        - balance_metrics: Standardized mean differences before/after
    weights : Optional[np.ndarray]
        Individual weights (if applicable)
    predictions : Optional[np.ndarray]
        Predicted outcomes (for STC)
    posterior_samples : Optional[np.ndarray]
        Posterior samples (for Bayesian methods)
    ipd_data : Optional[pd.DataFrame]
        Original IPD data with weights attached
    """

    effect_estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    method_name: str
    diagnostics: Dict = field(default_factory=dict)
    weights: Optional[np.ndarray] = None
    predictions: Optional[np.ndarray] = None
    posterior_samples: Optional[np.ndarray] = None
    ipd_data: Optional[pd.DataFrame] = None

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """Return the confidence/credible interval as a tuple."""
        return (self.ci_lower, self.ci_upper)

    @property
    def effective_sample_size(self) -> Optional[float]:
        """Return the effective sample size from diagnostics."""
        return self.diagnostics.get('effective_sample_size')

    def summary(self) -> str:
        """
        Return a formatted summary of the adjustment results.

        Returns
        -------
        str
            Formatted summary string
        """
        summary_lines = [
            f"\n{'='*60}",
            f"Population Adjustment Results: {self.method_name}",
            f"{'='*60}",
            f"Adjusted Effect Estimate: {self.effect_estimate:.4f}",
            f"Standard Error:          {self.standard_error:.4f}",
            f"95% CI:                  [{self.ci_lower:.4f}, {self.ci_upper:.4f}]",
            f"\nDiagnostics:",
        ]

        if 'effective_sample_size' in self.diagnostics:
            summary_lines.append(
                f"  Effective Sample Size: {self.diagnostics['effective_sample_size']:.1f}"
            )

        if 'max_weight' in self.diagnostics:
            summary_lines.append(f"  Maximum Weight:        {self.diagnostics['max_weight']:.4f}")

        if 'weight_cv' in self.diagnostics:
            summary_lines.append(
                f"  Weight CV:             {self.diagnostics['weight_cv']:.4f}"
            )

        summary_lines.append(f"{'='*60}\n")

        return "\n".join(summary_lines)

    def plot_weights(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot the distribution of weights.

        Parameters
        ----------
        figsize : Tuple[int, int]
            Figure size (width, height)

        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        if self.weights is None:
            raise ValueError("No weights available to plot")

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Histogram
        axes[0].hist(self.weights, bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(self.weights), color='red',
                       linestyle='--', label='Mean')
        axes[0].axvline(np.median(self.weights), color='blue',
                       linestyle='--', label='Median')
        axes[0].set_xlabel('Weight')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Weights')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Box plot
        axes[1].boxplot(self.weights, vert=True)
        axes[1].set_ylabel('Weight')
        axes[1].set_title('Weight Distribution (Box Plot)')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_balance(self,
                    covariates: Optional[List[str]] = None,
                    figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot covariate balance before and after adjustment.

        Parameters
        ----------
        covariates : Optional[List[str]]
            List of covariate names to plot. If None, uses all from diagnostics.
        figsize : Tuple[int, int]
            Figure size (width, height)

        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        if 'balance_metrics' not in self.diagnostics:
            raise ValueError("No balance metrics available")

        balance = self.diagnostics['balance_metrics']

        if covariates is None:
            covariates = list(balance.keys())

        # Prepare data for plotting
        smd_before = [balance[cov]['before'] for cov in covariates]
        smd_after = [balance[cov]['after'] for cov in covariates]

        fig, ax = plt.subplots(figsize=figsize)

        y_pos = np.arange(len(covariates))
        width = 0.35

        ax.barh(y_pos - width/2, smd_before, width,
               label='Before Adjustment', alpha=0.7, color='coral')
        ax.barh(y_pos + width/2, smd_after, width,
               label='After Adjustment', alpha=0.7, color='steelblue')

        # Add threshold lines at ±0.1 (common balance criterion)
        ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=-0.1, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(covariates)
        ax.set_xlabel('Standardized Mean Difference')
        ax.set_title('Covariate Balance Before and After Adjustment')
        ax.legend()
        ax.grid(alpha=0.3, axis='x')

        plt.tight_layout()
        return fig


class PopulationAdjustmentMethod(ABC):
    """
    Abstract base class for all population adjustment methods.

    Parameters
    ----------
    method : str
        Estimation method ('frequentist' or 'bayesian')
    ci_level : float
        Confidence/credible interval level (default: 0.95)
    random_state : Optional[int]
        Random seed for reproducibility
    """

    def __init__(
        self,
        method: str = 'frequentist',
        ci_level: float = 0.95,
        random_state: Optional[int] = None
    ):
        if method not in ['frequentist', 'bayesian']:
            raise ValueError("method must be 'frequentist' or 'bayesian'")

        if not 0 < ci_level < 1:
            raise ValueError("ci_level must be between 0 and 1")

        self.method = method
        self.ci_level = ci_level
        self.random_state = random_state
        self._result: Optional[AdjustmentResult] = None

        if random_state is not None:
            np.random.seed(random_state)

    @abstractmethod
    def fit(
        self,
        ipd_data: pd.DataFrame,
        aggregate_data: pd.DataFrame,
        covariates: List[str],
        outcome: str,
        treatment: str,
        **kwargs
    ) -> AdjustmentResult:
        """
        Fit the population adjustment method.

        Parameters
        ----------
        ipd_data : pd.DataFrame
            Individual patient data from index trial
        aggregate_data : pd.DataFrame
            Aggregate data from target population or comparator trial
        covariates : List[str]
            Names of covariates to adjust for
        outcome : str
            Name of outcome variable
        treatment : str
            Name of treatment variable
        **kwargs
            Additional method-specific parameters

        Returns
        -------
        AdjustmentResult
            Results object containing adjusted estimates and diagnostics
        """
        pass

    @abstractmethod
    def _validate_data(
        self,
        ipd_data: pd.DataFrame,
        aggregate_data: pd.DataFrame,
        covariates: List[str],
        outcome: str,
        treatment: str
    ) -> None:
        """
        Validate input data.

        Parameters
        ----------
        ipd_data : pd.DataFrame
            Individual patient data
        aggregate_data : pd.DataFrame
            Aggregate data
        covariates : List[str]
            Covariate names
        outcome : str
            Outcome variable name
        treatment : str
            Treatment variable name

        Raises
        ------
        ValueError
            If data validation fails
        """
        pass

    def calculate_smd(
        self,
        data1: Union[pd.DataFrame, np.ndarray],
        data2: Union[pd.DataFrame, np.ndarray],
        weights1: Optional[np.ndarray] = None,
        weights2: Optional[np.ndarray] = None
    ) -> Union[float, np.ndarray]:
        """
        Calculate standardized mean difference (SMD) between two groups.

        Parameters
        ----------
        data1 : Union[pd.DataFrame, np.ndarray]
            First dataset
        data2 : Union[pd.DataFrame, np.ndarray]
            Second dataset
        weights1 : Optional[np.ndarray]
            Weights for first dataset
        weights2 : Optional[np.ndarray]
            Weights for second dataset

        Returns
        -------
        Union[float, np.ndarray]
            Standardized mean difference(s)
        """
        if isinstance(data1, pd.DataFrame):
            data1 = data1.values
        if isinstance(data2, pd.DataFrame):
            data2 = data2.values

        # Calculate weighted means
        if weights1 is not None:
            mean1 = np.average(data1, weights=weights1, axis=0)
            var1 = np.average((data1 - mean1)**2, weights=weights1, axis=0)
        else:
            mean1 = np.mean(data1, axis=0)
            var1 = np.var(data1, axis=0, ddof=1)

        if weights2 is not None:
            mean2 = np.average(data2, weights=weights2, axis=0)
            var2 = np.average((data2 - mean2)**2, weights=weights2, axis=0)
        else:
            mean2 = np.mean(data2, axis=0)
            var2 = np.var(data2, axis=0, ddof=1)

        # Pooled standard deviation
        pooled_sd = np.sqrt((var1 + var2) / 2)

        # Avoid division by zero
        pooled_sd = np.where(pooled_sd == 0, 1, pooled_sd)

        smd = (mean1 - mean2) / pooled_sd

        return smd if smd.size > 1 else float(smd)

    def calculate_effective_sample_size(self, weights: np.ndarray) -> float:
        """
        Calculate effective sample size from weights.

        ESS = (sum of weights)^2 / sum of squared weights

        Parameters
        ----------
        weights : np.ndarray
            Array of weights

        Returns
        -------
        float
            Effective sample size
        """
        return np.sum(weights)**2 / np.sum(weights**2)

    @property
    def result(self) -> Optional[AdjustmentResult]:
        """Return the most recent adjustment result."""
        return self._result
