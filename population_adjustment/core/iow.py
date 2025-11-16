"""
Inverse Odds Weighting (IOW) for Population Adjustment

Implementation of inverse odds weighting methods, combining propensity scores
with inverse odds of trial participation for doubly-robust estimation.

References
----------
Phillippo et al. (2020). NICE DSU Technical Support Document 18.

Remiro-Azócar et al. (2020). Methods for population adjustment with limited
access to individual patient data. Research Synthesis Methods, 12(6), 750-775.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from population_adjustment.core.base import PopulationAdjustmentMethod, AdjustmentResult


class InverseOddsWeighting(PopulationAdjustmentMethod):
    """
    Inverse Odds Weighting (IOW) for Population Adjustment

    IOW combines propensity score weighting with inverse odds of trial
    participation to adjust for population differences. It can be used
    in both single-arm and comparative settings.

    Parameters
    ----------
    method : str
        'frequentist' or 'bayesian' (default: 'frequentist')
    propensity_model : str
        Model for propensity scores: 'logistic', 'boosting', 'random_forest'
        (default: 'logistic')
    doubly_robust : bool
        Use doubly-robust estimation (default: True)
    stabilize_weights : bool
        Use stabilized weights (default: True)
    trim_weights : Optional[float]
        Trim weights at specified quantile (e.g., 0.95)
    ci_level : float
        Confidence/credible interval level (default: 0.95)
    random_state : Optional[int]
        Random seed for reproducibility

    Attributes
    ----------
    propensity_model_ : object
        Fitted propensity score model
    weights_ : np.ndarray
        Estimated weights
    propensity_scores_ : np.ndarray
        Estimated propensity scores
    result_ : AdjustmentResult
        Results from most recent fit

    Examples
    --------
    >>> from population_adjustment import InverseOddsWeighting
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Combine IPD from trial and target population
    >>> trial_data = pd.DataFrame({
    ...     'age': np.random.normal(60, 10, 200),
    ...     'sex': np.random.binomial(1, 0.5, 200),
    ...     'treatment': np.random.binomial(1, 0.5, 200),
    ...     'response': np.random.binomial(1, 0.4, 200),
    ...     'trial': 1  # Indicator for trial membership
    ... })
    >>>
    >>> target_data = pd.DataFrame({
    ...     'age': np.random.normal(65, 10, 300),
    ...     'sex': np.random.binomial(1, 0.6, 300),
    ...     'trial': 0
    ... })
    >>>
    >>> # Combine datasets
    >>> combined = pd.concat([trial_data, target_data], ignore_index=True)
    >>>
    >>> # Fit IOW
    >>> iow = InverseOddsWeighting(propensity_model='logistic', doubly_robust=True)
    >>> result = iow.fit(
    ...     combined_data=combined,
    ...     trial_indicator='trial',
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
        propensity_model: str = 'logistic',
        doubly_robust: bool = True,
        stabilize_weights: bool = True,
        trim_weights: Optional[float] = None,
        ci_level: float = 0.95,
        random_state: Optional[int] = None
    ):
        super().__init__(method=method, ci_level=ci_level, random_state=random_state)

        if propensity_model not in ['logistic', 'boosting', 'random_forest']:
            raise ValueError(
                "propensity_model must be 'logistic', 'boosting', or 'random_forest'"
            )

        self.propensity_model_type = propensity_model
        self.doubly_robust = doubly_robust
        self.stabilize_weights = stabilize_weights
        self.trim_weights = trim_weights

        self.propensity_model_: Optional[object] = None
        self.outcome_model_: Optional[object] = None
        self.weights_: Optional[np.ndarray] = None
        self.propensity_scores_: Optional[np.ndarray] = None
        self.result_: Optional[AdjustmentResult] = None

    def fit(
        self,
        combined_data: pd.DataFrame,
        trial_indicator: str,
        covariates: List[str],
        outcome: str,
        treatment: str,
        outcome_type: str = 'binary',
        **kwargs
    ) -> AdjustmentResult:
        """
        Fit IOW to estimate population-adjusted treatment effect.

        Parameters
        ----------
        combined_data : pd.DataFrame
            Combined data from trial (S=1) and target population (S=0)
        trial_indicator : str
            Name of trial membership indicator (1=trial, 0=target)
        covariates : List[str]
            Names of covariates for propensity score model
        outcome : str
            Name of outcome variable (only in trial data)
        treatment : str
            Name of treatment variable (only in trial data)
        outcome_type : str
            Type of outcome: 'binary' or 'continuous'
        **kwargs
            Additional parameters

        Returns
        -------
        AdjustmentResult
            Results object with adjusted estimates
        """
        # Validate inputs
        self._validate_data_iow(
            combined_data, trial_indicator, covariates, outcome, treatment
        )

        # Separate trial and target data
        trial_data = combined_data[combined_data[trial_indicator] == 1].copy()
        target_data = combined_data[combined_data[trial_indicator] == 0].copy()

        # Estimate propensity scores
        self.propensity_scores_ = self._estimate_propensity_scores(
            combined_data, trial_indicator, covariates
        )

        # Calculate inverse odds weights
        ps_trial = self.propensity_scores_[combined_data[trial_indicator] == 1]
        self.weights_ = self._calculate_iow_weights(ps_trial)

        # Estimate treatment effect
        if self.doubly_robust:
            effect_estimate, std_error, ci = self._doubly_robust_estimation(
                trial_data, target_data, covariates, outcome, treatment,
                self.weights_, outcome_type
            )
        else:
            effect_estimate, std_error, ci = self._weighted_estimation(
                trial_data, outcome, treatment, self.weights_, outcome_type
            )

        # Calculate diagnostics
        diagnostics = self._calculate_diagnostics(
            trial_data, target_data, covariates, self.weights_
        )

        # Create result object
        self.result_ = AdjustmentResult(
            effect_estimate=effect_estimate,
            standard_error=std_error,
            ci_lower=ci[0],
            ci_upper=ci[1],
            method_name=f"IOW ({'DR' if self.doubly_robust else 'IPW'}, {self.propensity_model_type})",
            diagnostics=diagnostics,
            weights=self.weights_,
            ipd_data=trial_data
        )

        self._result = self.result_

        return self.result_

    def _estimate_propensity_scores(
        self,
        combined_data: pd.DataFrame,
        trial_indicator: str,
        covariates: List[str]
    ) -> np.ndarray:
        """
        Estimate propensity scores for trial membership.

        Parameters
        ----------
        combined_data : pd.DataFrame
            Combined trial and target data
        trial_indicator : str
            Trial membership indicator
        covariates : List[str]
            Covariate names

        Returns
        -------
        np.ndarray
            Propensity scores (probability of being in trial)
        """
        X = combined_data[covariates].values
        S = combined_data[trial_indicator].values

        # Standardize covariates
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit propensity score model
        if self.propensity_model_type == 'logistic':
            self.propensity_model_ = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='lbfgs'
            )
        elif self.propensity_model_type == 'boosting':
            self.propensity_model_ = GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=3
            )
        elif self.propensity_model_type == 'random_forest':
            self.propensity_model_ = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=10
            )

        self.propensity_model_.fit(X_scaled, S)

        # Predict propensity scores
        ps = self.propensity_model_.predict_proba(X_scaled)[:, 1]

        # Clip to avoid extreme weights
        ps = np.clip(ps, 0.01, 0.99)

        return ps

    def _calculate_iow_weights(
        self,
        propensity_scores: np.ndarray
    ) -> np.ndarray:
        """
        Calculate inverse odds weights from propensity scores.

        IOW weight = (1 - PS) / PS = odds of being in target population

        Parameters
        ----------
        propensity_scores : np.ndarray
            Propensity scores for trial participants

        Returns
        -------
        np.ndarray
            Inverse odds weights
        """
        # Inverse odds weight
        weights = (1 - propensity_scores) / propensity_scores

        # Stabilize weights if requested
        if self.stabilize_weights:
            # Stabilized weight = IOW / mean(IOW)
            weights = weights / np.mean(weights)

        # Trim extreme weights if requested
        if self.trim_weights is not None:
            upper_bound = np.quantile(weights, self.trim_weights)
            weights = np.minimum(weights, upper_bound)

        return weights

    def _weighted_estimation(
        self,
        trial_data: pd.DataFrame,
        outcome: str,
        treatment: str,
        weights: np.ndarray,
        outcome_type: str
    ) -> Tuple[float, float, Tuple[float, float]]:
        """
        Simple weighted estimation (IPW only).

        Parameters
        ----------
        trial_data : pd.DataFrame
            Trial data
        outcome : str
            Outcome variable
        treatment : str
            Treatment variable
        weights : np.ndarray
            Weights
        outcome_type : str
            Type of outcome

        Returns
        -------
        Tuple[float, float, Tuple[float, float]]
            Effect estimate, SE, CI
        """
        y = trial_data[outcome].values
        trt = trial_data[treatment].values
        n = len(y)

        # Normalize weights
        w_norm = weights / np.sum(weights) * n

        if outcome_type == 'binary':
            # Weighted proportions
            p1 = np.sum(y * trt * w_norm) / np.sum(trt * w_norm)
            p0 = np.sum(y * (1 - trt) * w_norm) / np.sum((1 - trt) * w_norm)

            effect = p1 - p0

            # Variance
            n1_eff = self.calculate_effective_sample_size(w_norm[trt == 1])
            n0_eff = self.calculate_effective_sample_size(w_norm[trt == 0])

            var1 = p1 * (1 - p1) / n1_eff
            var0 = p0 * (1 - p0) / n0_eff

            std_error = np.sqrt(var1 + var0)

        else:  # continuous
            # Weighted means
            mean1 = np.sum(y * trt * w_norm) / np.sum(trt * w_norm)
            mean0 = np.sum(y * (1 - trt) * w_norm) / np.sum((1 - trt) * w_norm)

            effect = mean1 - mean0

            # Weighted variance
            var1 = np.sum(((y - mean1)**2) * trt * w_norm) / np.sum(trt * w_norm)
            var0 = np.sum(((y - mean0)**2) * (1 - trt) * w_norm) / np.sum((1 - trt) * w_norm)

            n1_eff = self.calculate_effective_sample_size(w_norm[trt == 1])
            n0_eff = self.calculate_effective_sample_size(w_norm[trt == 0])

            std_error = np.sqrt(var1 / n1_eff + var0 / n0_eff)

        # Confidence interval
        z_alpha = norm.ppf(1 - (1 - self.ci_level) / 2)
        ci_lower = effect - z_alpha * std_error
        ci_upper = effect + z_alpha * std_error

        return effect, std_error, (ci_lower, ci_upper)

    def _doubly_robust_estimation(
        self,
        trial_data: pd.DataFrame,
        target_data: pd.DataFrame,
        covariates: List[str],
        outcome: str,
        treatment: str,
        weights: np.ndarray,
        outcome_type: str
    ) -> Tuple[float, float, Tuple[float, float]]:
        """
        Doubly-robust estimation combining weighting and outcome regression.

        Parameters
        ----------
        trial_data : pd.DataFrame
            Trial data
        target_data : pd.DataFrame
            Target population data
        covariates : List[str]
            Covariate names
        outcome : str
            Outcome variable
        treatment : str
            Treatment variable
        weights : np.ndarray
            IOW weights
        outcome_type : str
            Type of outcome

        Returns
        -------
        Tuple[float, float, Tuple[float, float]]
            Effect estimate, SE, CI
        """
        # Step 1: Fit outcome models in trial data
        X_trial = trial_data[covariates].values
        y_trial = trial_data[outcome].values
        trt_trial = trial_data[treatment].values

        scaler = StandardScaler()
        X_trial_scaled = scaler.fit_transform(X_trial)

        # Fit separate models for treated and control
        if outcome_type == 'binary':
            model_1 = LogisticRegression(max_iter=1000)
            model_0 = LogisticRegression(max_iter=1000)
        else:
            from sklearn.linear_model import LinearRegression
            model_1 = LinearRegression()
            model_0 = LinearRegression()

        # Fit models
        idx_1 = trt_trial == 1
        idx_0 = trt_trial == 0

        model_1.fit(X_trial_scaled[idx_1], y_trial[idx_1])
        model_0.fit(X_trial_scaled[idx_0], y_trial[idx_0])

        self.outcome_model_ = {'model_1': model_1, 'model_0': model_0, 'scaler': scaler}

        # Step 2: Predict in target population
        if len(target_data) > 0:
            X_target = target_data[covariates].values
            X_target_scaled = scaler.transform(X_target)

            if outcome_type == 'binary':
                y_target_1 = model_1.predict_proba(X_target_scaled)[:, 1]
                y_target_0 = model_0.predict_proba(X_target_scaled)[:, 1]
            else:
                y_target_1 = model_1.predict(X_target_scaled)
                y_target_0 = model_0.predict(X_target_scaled)

            # Average treatment effect in target population
            effect_outcome_model = np.mean(y_target_1 - y_target_0)
        else:
            # No target population data, use predictions on trial data
            if outcome_type == 'binary':
                y_pred_1 = model_1.predict_proba(X_trial_scaled)[:, 1]
                y_pred_0 = model_0.predict_proba(X_trial_scaled)[:, 1]
            else:
                y_pred_1 = model_1.predict(X_trial_scaled)
                y_pred_0 = model_0.predict(X_trial_scaled)

            effect_outcome_model = np.mean(y_pred_1 - y_pred_0)

        # Step 3: Doubly-robust estimator
        # Combines weighted estimation with outcome model predictions
        n = len(y_trial)
        w_norm = weights / np.sum(weights) * n

        # Predict outcomes for trial participants
        if outcome_type == 'binary':
            mu_1 = model_1.predict_proba(X_trial_scaled)[:, 1]
            mu_0 = model_0.predict_proba(X_trial_scaled)[:, 1]
        else:
            mu_1 = model_1.predict(X_trial_scaled)
            mu_0 = model_0.predict(X_trial_scaled)

        # DR estimator
        dr_component_1 = w_norm * trt_trial * (y_trial - mu_1)
        dr_component_0 = w_norm * (1 - trt_trial) * (y_trial - mu_0)

        effect = (
            effect_outcome_model +
            np.mean(dr_component_1) -
            np.mean(dr_component_0)
        )

        # Variance estimation (conservative using bootstrap)
        # For simplicity, use IPW variance as conservative estimate
        _, std_error, _ = self._weighted_estimation(
            trial_data, outcome, treatment, weights, outcome_type
        )

        # Confidence interval
        z_alpha = norm.ppf(1 - (1 - self.ci_level) / 2)
        ci_lower = effect - z_alpha * std_error
        ci_upper = effect + z_alpha * std_error

        return effect, std_error, (ci_lower, ci_upper)

    def _calculate_diagnostics(
        self,
        trial_data: pd.DataFrame,
        target_data: pd.DataFrame,
        covariates: List[str],
        weights: np.ndarray
    ) -> Dict:
        """
        Calculate diagnostic metrics.

        Parameters
        ----------
        trial_data : pd.DataFrame
            Trial data
        target_data : pd.DataFrame
            Target population data
        covariates : List[str]
            Covariate names
        weights : np.ndarray
            Weights

        Returns
        -------
        Dict
            Diagnostic metrics
        """
        # Effective sample size
        ess = self.calculate_effective_sample_size(weights)

        # Weight statistics
        max_weight = np.max(weights)
        weight_cv = np.std(weights) / np.mean(weights)

        # Propensity score overlap
        ps_trial = self.propensity_scores_[
            len(self.propensity_scores_) - len(trial_data):
        ]
        ps_overlap = {
            'ps_mean_trial': float(np.mean(ps_trial)),
            'ps_min_trial': float(np.min(ps_trial)),
            'ps_max_trial': float(np.max(ps_trial))
        }

        # Covariate balance
        X_trial = trial_data[covariates].values
        X_target = target_data[covariates].values if len(target_data) > 0 else None

        balance_metrics = {}
        w_norm = weights / np.sum(weights)

        for i, cov in enumerate(covariates):
            # Before weighting
            if X_target is not None:
                smd_before = self.calculate_smd(
                    X_trial[:, i], X_target[:, i]
                )
            else:
                smd_before = 0.0

            # After weighting
            if X_target is not None:
                smd_after = self.calculate_smd(
                    X_trial[:, i], X_target[:, i],
                    weights1=w_norm
                )
            else:
                smd_after = 0.0

            balance_metrics[cov] = {
                'before': float(smd_before),
                'after': float(smd_after)
            }

        diagnostics = {
            'effective_sample_size': float(ess),
            'max_weight': float(max_weight),
            'weight_cv': float(weight_cv),
            'propensity_score_overlap': ps_overlap,
            'balance_metrics': balance_metrics,
            'n_trial': len(trial_data),
            'n_target': len(target_data),
            'doubly_robust': self.doubly_robust
        }

        return diagnostics

    def _validate_data_iow(
        self,
        combined_data: pd.DataFrame,
        trial_indicator: str,
        covariates: List[str],
        outcome: str,
        treatment: str
    ) -> None:
        """Validate input data for IOW."""
        # Check trial indicator exists
        if trial_indicator not in combined_data.columns:
            raise ValueError(f"Trial indicator '{trial_indicator}' not in data")

        # Check trial indicator is binary
        unique_s = combined_data[trial_indicator].unique()
        if not set(unique_s).issubset({0, 1}):
            raise ValueError("Trial indicator must be binary (0/1)")

        # Check covariates exist
        missing = set(covariates) - set(combined_data.columns)
        if missing:
            raise ValueError(f"Data missing columns: {missing}")

        # Check trial data has outcome and treatment
        trial_data = combined_data[combined_data[trial_indicator] == 1]

        if outcome not in trial_data.columns:
            raise ValueError(f"Outcome '{outcome}' not in trial data")

        if treatment not in trial_data.columns:
            raise ValueError(f"Treatment '{treatment}' not in trial data")

        # Check for missing values
        if combined_data[covariates].isnull().any().any():
            raise ValueError("Covariate data contains missing values")

        if trial_data[[outcome, treatment]].isnull().any().any():
            raise ValueError("Trial outcome/treatment data contains missing values")

    def _validate_data(
        self,
        ipd_data: pd.DataFrame,
        aggregate_data: pd.DataFrame,
        covariates: List[str],
        outcome: str,
        treatment: str
    ) -> None:
        """Placeholder - IOW uses different validation."""
        pass
