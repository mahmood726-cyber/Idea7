"""
Plotting functions for visualizing population adjustment results.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_balance(
    balance_metrics: pd.DataFrame,
    threshold: float = 0.1,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create a love plot showing covariate balance.

    Parameters
    ----------
    balance_metrics : pd.DataFrame
        Balance metrics from calculate_balance_metrics()
    threshold : float
        SMD threshold for adequate balance
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    variables = balance_metrics['variable'].values
    smd = balance_metrics['smd'].values

    y_pos = np.arange(len(variables))

    colors = ['green' if abs(s) < threshold else 'red' for s in smd]

    ax.barh(y_pos, smd, color=colors, alpha=0.6)
    ax.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5, label=f'±{threshold} threshold')
    ax.axvline(x=-threshold, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables)
    ax.set_xlabel('Standardized Mean Difference (SMD)')
    ax.set_title('Covariate Balance (Love Plot)')
    ax.legend()
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    return fig


def plot_weights(
    weights: np.ndarray,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot weight distribution.

    Parameters
    ----------
    weights : np.ndarray
        Weight vector
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Histogram
    axes[0].hist(weights, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(np.mean(weights), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {np.mean(weights):.2f}')
    axes[0].axvline(np.median(weights), color='orange', linestyle='--',
                    linewidth=2, label=f'Median: {np.median(weights):.2f}')
    axes[0].set_xlabel('Weight')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Weight Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Box plot
    axes[1].boxplot(weights, vert=True)
    axes[1].set_ylabel('Weight')
    axes[1].set_title('Weight Distribution (Box Plot)')
    axes[1].grid(alpha=0.3)

    # Q-Q plot
    from scipy import stats
    stats.probplot(weights, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_propensity_scores(
    ps_trial: np.ndarray,
    ps_target: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot propensity score distributions.

    Parameters
    ----------
    ps_trial : np.ndarray
        Propensity scores for trial participants
    ps_target : Optional[np.ndarray]
        Propensity scores for target population
    weights : Optional[np.ndarray]
        Weights for trial participants
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    if ps_target is not None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(6, 5))
        axes = [axes]

    # Unweighted distribution
    axes[0].hist(ps_trial, bins=30, alpha=0.5, label='Trial',
                 color='steelblue', edgecolor='black')
    if ps_target is not None:
        axes[0].hist(ps_target, bins=30, alpha=0.5, label='Target',
                     color='coral', edgecolor='black')
    axes[0].set_xlabel('Propensity Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Propensity Score Distribution (Unweighted)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Weighted distribution (if weights provided)
    if ps_target is not None and weights is not None:
        axes[1].hist(ps_trial, bins=30, weights=weights, alpha=0.5,
                     label='Trial (weighted)', color='steelblue', edgecolor='black')
        axes[1].hist(ps_target, bins=30, alpha=0.5, label='Target',
                     color='coral', edgecolor='black')
        axes[1].set_xlabel('Propensity Score')
        axes[1].set_ylabel('Weighted Frequency')
        axes[1].set_title('Propensity Score Distribution (Weighted)')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_forest(
    results: List[Dict],
    labels: List[str],
    xlabel: str = 'Effect Estimate',
    title: str = 'Forest Plot',
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create forest plot for comparing multiple estimates.

    Parameters
    ----------
    results : List[Dict]
        List of result dictionaries with 'estimate', 'ci_lower', 'ci_upper'
    labels : List[str]
        Labels for each estimate
    xlabel : str
        X-axis label
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(results))

    # Extract estimates and CIs
    estimates = [r['estimate'] for r in results]
    ci_lower = [r['ci_lower'] for r in results]
    ci_upper = [r['ci_upper'] for r in results]

    # Calculate error bars
    errors = np.array([
        [estimates[i] - ci_lower[i], ci_upper[i] - estimates[i]]
        for i in range(len(results))
    ]).T

    # Plot
    ax.errorbar(estimates, y_pos, xerr=errors, fmt='o', markersize=8,
                capsize=5, capthick=2, linewidth=2, color='steelblue')

    # Reference line at 0 or 1 depending on scale
    if all(e > 0 for e in estimates):
        ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    else:
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    return fig


def plot_sensitivity(
    sensitivity_results: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot sensitivity analysis results.

    Parameters
    ----------
    sensitivity_results : pd.DataFrame
        Results from sensitivity_analysis()
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Heatmap of adjusted effects
    pivot_data = sensitivity_results.pivot(
        index='unmeasured_prevalence',
        columns='unmeasured_or',
        values='adjusted_effect'
    )

    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn',
                center=0, ax=axes[0], cbar_kws={'label': 'Adjusted Effect'})
    axes[0].set_xlabel('Unmeasured Confounder OR')
    axes[0].set_ylabel('Unmeasured Confounder Prevalence')
    axes[0].set_title('Sensitivity to Unmeasured Confounding')

    # Contour plot
    pivot_bias = sensitivity_results.pivot(
        index='unmeasured_prevalence',
        columns='unmeasured_or',
        values='bias'
    )

    X = pivot_bias.columns.values
    Y = pivot_bias.index.values
    Z = pivot_bias.values

    contour = axes[1].contourf(X, Y, Z, levels=10, cmap='RdYlBu_r')
    axes[1].contour(X, Y, Z, levels=10, colors='black', linewidths=0.5)
    fig.colorbar(contour, ax=axes[1], label='Bias')
    axes[1].set_xlabel('Unmeasured Confounder OR')
    axes[1].set_ylabel('Unmeasured Confounder Prevalence')
    axes[1].set_title('Bias Due to Unmeasured Confounding')

    plt.tight_layout()
    return fig


def plot_covariate_distributions(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    covariates: List[str],
    weights1: Optional[np.ndarray] = None,
    labels: Tuple[str, str] = ('Trial', 'Target'),
    ncols: int = 3,
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Plot covariate distributions for two datasets.

    Parameters
    ----------
    data1 : pd.DataFrame
        First dataset
    data2 : pd.DataFrame
        Second dataset
    covariates : List[str]
        Covariates to plot
    weights1 : Optional[np.ndarray]
        Weights for first dataset
    labels : Tuple[str, str]
        Labels for datasets
    ncols : int
        Number of columns in subplot grid
    figsize : Optional[Tuple[int, int]]
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    n_vars = len(covariates)
    nrows = int(np.ceil(n_vars / ncols))

    if figsize is None:
        figsize = (ncols * 4, nrows * 3)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_vars > 1 else [axes]

    for i, var in enumerate(covariates):
        ax = axes[i]

        x1 = data1[var].values
        x2 = data2[var].values

        # Determine if categorical
        if len(np.unique(x1)) <= 10:
            # Categorical - bar plot
            unique_vals = np.unique(np.concatenate([x1, x2]))

            if weights1 is not None:
                freq1 = np.array([
                    np.sum(weights1[x1 == v]) / np.sum(weights1)
                    for v in unique_vals
                ])
            else:
                freq1 = np.array([np.mean(x1 == v) for v in unique_vals])

            freq2 = np.array([np.mean(x2 == v) for v in unique_vals])

            x_pos = np.arange(len(unique_vals))
            width = 0.35

            ax.bar(x_pos - width/2, freq1, width, label=labels[0], alpha=0.7)
            ax.bar(x_pos + width/2, freq2, width, label=labels[1], alpha=0.7)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(unique_vals)
            ax.set_ylabel('Proportion')

        else:
            # Continuous - histogram
            if weights1 is not None:
                ax.hist(x1, bins=20, alpha=0.5, label=labels[0],
                        weights=weights1, density=True)
            else:
                ax.hist(x1, bins=20, alpha=0.5, label=labels[0], density=True)

            ax.hist(x2, bins=20, alpha=0.5, label=labels[1], density=True)
            ax.set_ylabel('Density')

        ax.set_xlabel(var)
        ax.set_title(f'Distribution of {var}')
        ax.legend()
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for i in range(n_vars, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig
