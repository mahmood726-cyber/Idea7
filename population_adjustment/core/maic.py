"""
Matching-Adjusted Indirect Comparison (MAIC)

Implementation of MAIC methods for population adjustment with both
frequentist and Bayesian approaches.

References
----------
Signorovitch et al. (2012). Comparative effectiveness without head-to-head trials.
Pharmacoeconomics, 30(5), 351-363.

Phillippo et al. (2020). NICE DSU Technical Support Document 18.
"""

import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

from population_adjustment.core.base import PopulationAdjustmentMethod, AdjustmentResult


class MAIC(PopulationAdjustmentMethod):
    """
    Matching-Adjusted Indirect Comparison (MAIC)

    MAIC reweights individual patient data to match summary statistics from
    a target population using propensity score weighting.

    Parameters
    ----------
    method : str
        'frequentist' or 'bayesian' (default: 'frequentist')
    weighting_method : str
        Type of weighting: 'standard', 'entropy_balancing', 'calibration'
        (default: 'standard')
    ci_level : float
        Confidence/credible interval level (default: 0.95)
    random_state : Optional[int]
        Random seed for reproducibility
    max_iter : int
        Maximum iterations for optimization (default: 1000)
    tol : float
        Convergence tolerance (default: 1e-6)

    Attributes
    ----------
    weights_ : np.ndarray
        Estimated weights for each individual
    beta_ : np.ndarray
        Estimated balancing coefficients
    result_ : AdjustmentResult
        Results from most recent fit

    Examples
    --------
    >>> from population_adjustment import MAIC
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Simulate IPD
    >>> ipd = pd.DataFrame({
    ...     'age': np.random.normal(60, 10, 200),
    ...     'sex': np.random.binomial(1, 0.5, 200),
    ...     'treatment': np.random.binomial(1, 0.5, 200),
    ...     'response': np.random.binomial(1, 0.4, 200)
    ... })
    >>>
    >>> # Target population statistics
    >>> target = pd.DataFrame({
    ...     'age': [65],
    ...     'sex': [0.6]
    ... })
    >>>
    >>> # Fit MAIC
    >>> maic = MAIC(method='frequentist')
    >>> result = maic.fit(
    ...     ipd_data=ipd,
    ...     aggregate_data=target,
    ...     covariates=['age', 'sex'],
    ...     outcome='response',
    ...     treatment='treatment'
    ... )
    >>>
    >>> print(result.summary())
    """

    def __init__(
        self,
        method: str = 'frequentist',
        weighting_method: str = 'standard',
        ci_level: float = 0.95,
        random_state: Optional[int] = None,
        max_iter: int = 1000,
        tol: float = 1e-6
    ):
        super().__init__(method=method, ci_level=ci_level, random_state=random_state)

        if weighting_method not in ['standard', 'entropy_balancing', 'calibration']:
            raise ValueError(
                "weighting_method must be 'standard', 'entropy_balancing', or 'calibration'"
            )

        self.weighting_method = weighting_method
        self.max_iter = max_iter
        self.tol = tol

        self.weights_: Optional[np.ndarray] = None
        self.beta_: Optional[np.ndarray] = None
        self.result_: Optional[AdjustmentResult] = None

    def fit(
        self,
        ipd_data: pd.DataFrame,
        aggregate_data: pd.DataFrame,
        covariates: List[str],
        outcome: str,
        treatment: str,
        outcome_type: str = 'binary',
        **kwargs
    ) -> AdjustmentResult:
        """
        Fit MAIC to estimate population-adjusted treatment effect.

        Parameters
        ----------
        ipd_data : pd.DataFrame
            Individual patient data with columns for covariates, outcome, treatment
        aggregate_data : pd.DataFrame
            Target population statistics (single row with mean values)
        covariates : List[str]
            Names of covariates to balance
        outcome : str
            Name of outcome variable in IPD
        treatment : str
            Name of treatment variable in IPD
        outcome_type : str
            Type of outcome: 'binary', 'continuous', 'time_to_event'
        **kwargs
            Additional parameters

        Returns
        -------
        AdjustmentResult
            Results object with adjusted estimates
        """
        # Validate inputs
        self._validate_data(ipd_data, aggregate_data, covariates, outcome, treatment)

        # Extract covariate matrices
        X_ipd = ipd_data[covariates].values
        X_target = aggregate_data[covariates].values.flatten()

        # Standardize covariates for numerical stability
        scaler = StandardScaler()
        X_ipd_scaled = scaler.fit_transform(X_ipd)
        X_target_scaled = scaler.transform(X_target.reshape(1, -1)).flatten()

        # Estimate weights
        if self.method == 'frequentist':
            self.weights_, self.beta_ = self._estimate_weights_frequentist(
                X_ipd_scaled, X_target_scaled
            )
        else:  # Bayesian
            self.weights_, self.beta_ = self._estimate_weights_bayesian(
                X_ipd_scaled, X_target_scaled
            )

        # Calculate adjusted treatment effect
        effect_estimate, std_error, ci = self._estimate_treatment_effect(
            ipd_data, outcome, treatment, self.weights_, outcome_type
        )

        # Calculate diagnostics
        diagnostics = self._calculate_diagnostics(
            X_ipd, X_target, self.weights_, covariates
        )

        # Create result object
        self.result_ = AdjustmentResult(
            effect_estimate=effect_estimate,
            standard_error=std_error,
            ci_lower=ci[0],
            ci_upper=ci[1],
            method_name=f"MAIC ({self.method}, {self.weighting_method})",
            diagnostics=diagnostics,
            weights=self.weights_,
            ipd_data=ipd_data.copy()
        )

        self._result = self.result_

        return self.result_

    def _estimate_weights_frequentist(
        self,
        X_ipd: np.ndarray,
        X_target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate weights using frequentist method of moments.

        The weights are chosen to minimize the entropy subject to
        moment constraints.

        Parameters
        ----------
        X_ipd : np.ndarray
            Standardized IPD covariate matrix (n x p)
        X_target : np.ndarray
            Target population means (p,)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            weights, beta coefficients
        """
        n, p = X_ipd.shape

        def objective(beta):
            """Dual objective function to minimize."""
            log_weights = -X_ipd @ beta
            # Normalize to prevent overflow
            log_weights = log_weights - np.max(log_weights)
            weights = np.exp(log_weights)
            weights = weights / np.sum(weights)

            # Entropy (negative since we're minimizing)
            entropy = np.sum(weights * np.log(weights * n))

            return entropy

        def constraint_eq(beta):
            """Moment matching constraints."""
            log_weights = -X_ipd @ beta
            log_weights = log_weights - np.max(log_weights)
            weights = np.exp(log_weights)
            weights = weights / np.sum(weights)

            # Weighted means should equal target means
            weighted_means = np.sum(X_ipd * weights[:, np.newaxis], axis=0)

            return weighted_means - X_target

        # Initial guess
        beta_init = np.zeros(p)

        # Optimization with constraints
        if self.weighting_method == 'standard':
            result = optimize.minimize(
                objective,
                beta_init,
                method='SLSQP',
                constraints={'type': 'eq', 'fun': constraint_eq},
                options={'maxiter': self.max_iter, 'ftol': self.tol}
            )

            if not result.success:
                warnings.warn(f"Optimization did not converge: {result.message}")

            beta = result.x

        elif self.weighting_method == 'entropy_balancing':
            # Entropy balancing with additional variance constraint
            beta = self._entropy_balancing(X_ipd, X_target)

        else:  # calibration
            # Calibration weighting
            beta = self._calibration_weighting(X_ipd, X_target)

        # Calculate final weights
        log_weights = -X_ipd @ beta
        log_weights = log_weights - np.max(log_weights)
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights) * n

        return weights, beta

    def _entropy_balancing(
        self,
        X_ipd: np.ndarray,
        X_target: np.ndarray
    ) -> np.ndarray:
        """
        Entropy balancing with stable weights.

        Adds soft constraints to prevent extreme weights.
        """
        n, p = X_ipd.shape

        def loss(beta):
            """Combined objective: entropy + moment mismatch."""
            log_weights = -X_ipd @ beta
            log_weights = log_weights - np.max(log_weights)
            weights = np.exp(log_weights)
            weights = weights / np.sum(weights)

            # Entropy
            entropy = np.sum(weights * np.log(weights * n + 1e-10))

            # Moment mismatch (soft constraint)
            weighted_means = np.sum(X_ipd * weights[:, np.newaxis], axis=0)
            moment_penalty = 1000 * np.sum((weighted_means - X_target)**2)

            return entropy + moment_penalty

        beta_init = np.zeros(p)
        result = optimize.minimize(
            loss,
            beta_init,
            method='L-BFGS-B',
            options={'maxiter': self.max_iter}
        )

        return result.x

    def _calibration_weighting(
        self,
        X_ipd: np.ndarray,
        X_target: np.ndarray
    ) -> np.ndarray:
        """
        Calibration weighting (linear regression approach).

        Uses linear calibration to match moments.
        """
        n = X_ipd.shape[0]

        # Start with equal weights
        base_weights = np.ones(n) / n

        # Calculate initial weighted means
        initial_means = X_ipd.T @ base_weights

        # Calibration adjustment
        diff = X_target - initial_means

        # Use generalized regression estimator
        Q = (X_ipd.T @ X_ipd) / n
        Q_inv = np.linalg.pinv(Q)

        # Calculate adjustment factor
        g = np.ones(n) + X_ipd @ Q_inv @ diff

        # Ensure positive weights
        g = np.maximum(g, 0.1)

        # Calculate beta from g-weights
        weights = g / np.sum(g) * n
        beta = -np.log(weights / n)

        return beta

    def _estimate_weights_bayesian(
        self,
        X_ipd: np.ndarray,
        X_target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate weights using Bayesian approach with PyMC.

        Places a prior on beta and samples from posterior.

        Parameters
        ----------
        X_ipd : np.ndarray
            IPD covariates
        X_target : np.ndarray
            Target population means

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Mean posterior weights and beta
        """
        try:
            import pymc as pm
            import pytensor.tensor as pt
        except ImportError:
            raise ImportError(
                "PyMC is required for Bayesian methods. "
                "Install with: pip install pymc>=5.0"
            )

        n, p = X_ipd.shape

        with pm.Model() as model:
            # Prior on beta (weakly informative)
            beta = pm.Normal('beta', mu=0, sigma=2, shape=p)

            # Calculate weights
            log_weights = -pm.math.dot(X_ipd, beta)
            log_weights = log_weights - pm.math.max(log_weights)
            unnorm_weights = pm.math.exp(log_weights)
            weights = unnorm_weights / pm.math.sum(unnorm_weights)

            # Moment matching likelihood
            weighted_means = pm.math.dot(X_ipd.T, weights)

            # Likelihood: weighted means should match target
            _ = pm.Normal(
                'obs',
                mu=weighted_means,
                sigma=0.1,  # Small sigma for tight matching
                observed=X_target
            )

            # Sample from posterior
            trace = pm.sample(
                1000,
                tune=1000,
                random_seed=self.random_state,
                progressbar=False,
                return_inferencedata=False
            )

        # Extract posterior means
        beta_mean = trace['beta'].mean(axis=0)

        # Calculate mean weights
        log_weights = -X_ipd @ beta_mean
        log_weights = log_weights - np.max(log_weights)
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights) * n

        return weights, beta_mean

    def _estimate_treatment_effect(
        self,
        ipd_data: pd.DataFrame,
        outcome: str,
        treatment: str,
        weights: np.ndarray,
        outcome_type: str
    ) -> Tuple[float, float, Tuple[float, float]]:
        """
        Estimate weighted treatment effect.

        Parameters
        ----------
        ipd_data : pd.DataFrame
            IPD data
        outcome : str
            Outcome variable
        treatment : str
            Treatment variable
        weights : np.ndarray
            Estimated weights
        outcome_type : str
            Type of outcome

        Returns
        -------
        Tuple[float, float, Tuple[float, float]]
            Effect estimate, standard error, confidence interval
        """
        y = ipd_data[outcome].values
        trt = ipd_data[treatment].values

        # Normalize weights to sum to n for variance calculation
        n = len(y)
        w_norm = weights / np.sum(weights) * n

        if outcome_type == 'binary':
            # Risk difference or log odds ratio
            # Calculate weighted proportions
            p1 = np.sum(y * trt * w_norm) / np.sum(trt * w_norm)
            p0 = np.sum(y * (1 - trt) * w_norm) / np.sum((1 - trt) * w_norm)

            effect = p1 - p0  # Risk difference

            # Variance using weighted formula
            n1 = np.sum(trt * w_norm)
            n0 = np.sum((1 - trt) * w_norm)

            var1 = p1 * (1 - p1) / n1
            var0 = p0 * (1 - p0) / n0

            std_error = np.sqrt(var1 + var0)

        elif outcome_type == 'continuous':
            # Mean difference
            mean1 = np.sum(y * trt * w_norm) / np.sum(trt * w_norm)
            mean0 = np.sum(y * (1 - trt) * w_norm) / np.sum((1 - trt) * w_norm)

            effect = mean1 - mean0

            # Weighted variance
            var1 = np.sum(((y - mean1)**2) * trt * w_norm) / np.sum(trt * w_norm)
            var0 = np.sum(((y - mean0)**2) * (1 - trt) * w_norm) / np.sum((1 - trt) * w_norm)

            n1 = np.sum(trt * w_norm)
            n0 = np.sum((1 - trt) * w_norm)

            std_error = np.sqrt(var1 / n1 + var0 / n0)

        else:
            raise ValueError(f"Outcome type '{outcome_type}' not yet implemented")

        # Confidence interval
        z_alpha = norm.ppf(1 - (1 - self.ci_level) / 2)
        ci_lower = effect - z_alpha * std_error
        ci_upper = effect + z_alpha * std_error

        return effect, std_error, (ci_lower, ci_upper)

    def _calculate_diagnostics(
        self,
        X_ipd: np.ndarray,
        X_target: np.ndarray,
        weights: np.ndarray,
        covariates: List[str]
    ) -> Dict:
        """
        Calculate diagnostic metrics for weight quality.

        Parameters
        ----------
        X_ipd : np.ndarray
            IPD covariates
        X_target : np.ndarray
            Target means
        weights : np.ndarray
            Estimated weights
        covariates : List[str]
            Covariate names

        Returns
        -------
        Dict
            Diagnostic metrics
        """
        # Effective sample size
        ess = self.calculate_effective_sample_size(weights)

        # Maximum weight
        max_weight = np.max(weights)

        # Coefficient of variation of weights
        weight_cv = np.std(weights) / np.mean(weights)

        # Balance metrics (SMD before and after)
        balance_metrics = {}

        # Before adjustment
        smd_before = (np.mean(X_ipd, axis=0) - X_target) / np.std(X_ipd, axis=0)

        # After adjustment
        w_norm = weights / np.sum(weights)
        weighted_means = np.sum(X_ipd * w_norm[:, np.newaxis], axis=0)
        weighted_std = np.sqrt(
            np.sum(((X_ipd - weighted_means)**2) * w_norm[:, np.newaxis], axis=0)
        )
        smd_after = (weighted_means - X_target) / weighted_std

        for i, cov in enumerate(covariates):
            balance_metrics[cov] = {
                'before': float(smd_before[i]),
                'after': float(smd_after[i])
            }

        diagnostics = {
            'effective_sample_size': float(ess),
            'max_weight': float(max_weight),
            'weight_cv': float(weight_cv),
            'balance_metrics': balance_metrics,
            'n_observations': len(weights)
        }

        return diagnostics

    def _validate_data(
        self,
        ipd_data: pd.DataFrame,
        aggregate_data: pd.DataFrame,
        covariates: List[str],
        outcome: str,
        treatment: str
    ) -> None:
        """Validate input data for MAIC."""
        # Check IPD has required columns
        required_cols = covariates + [outcome, treatment]
        missing = set(required_cols) - set(ipd_data.columns)
        if missing:
            raise ValueError(f"IPD data missing columns: {missing}")

        # Check aggregate data has covariate columns
        missing_agg = set(covariates) - set(aggregate_data.columns)
        if missing_agg:
            raise ValueError(f"Aggregate data missing columns: {missing_agg}")

        # Check for missing values
        if ipd_data[required_cols].isnull().any().any():
            raise ValueError("IPD data contains missing values")

        # Check treatment is binary
        unique_trt = ipd_data[treatment].unique()
        if not set(unique_trt).issubset({0, 1}):
            raise ValueError("Treatment variable must be binary (0/1)")

        # Check aggregate data is single row
        if len(aggregate_data) != 1:
            warnings.warn(
                f"Aggregate data has {len(aggregate_data)} rows. "
                "Using first row as target."
            )
