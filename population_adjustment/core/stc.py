"""
Simulated Treatment Comparison (STC)

Implementation of STC methods for population adjustment using outcome regression.
Includes frequentist and Bayesian approaches with G-computation.

References
----------
Caro & Ishak (2010). No head-to-head trial? Simulate the missing arms.
Pharmacoeconomics, 28(10), 957-967.

Phillippo et al. (2020). NICE DSU Technical Support Document 18.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from population_adjustment.core.base import PopulationAdjustmentMethod, AdjustmentResult


class STC(PopulationAdjustmentMethod):
    """
    Simulated Treatment Comparison (STC)

    STC uses outcome regression to predict counterfactual outcomes in the
    target population. It models the relationship between covariates, treatment,
    and outcome in the IPD, then uses this model to predict outcomes in the
    target population.

    Parameters
    ----------
    method : str
        'frequentist' or 'bayesian' (default: 'frequentist')
    outcome_model : str
        Type of outcome model: 'linear', 'logistic', 'random_forest'
        (default: 'logistic')
    include_interaction : bool
        Include treatment-covariate interactions (default: True)
    ci_level : float
        Confidence/credible interval level (default: 0.95)
    random_state : Optional[int]
        Random seed for reproducibility
    bootstrap_samples : int
        Number of bootstrap samples for variance estimation (default: 1000)

    Attributes
    ----------
    outcome_model_ : object
        Fitted outcome model
    scaler_ : StandardScaler
        Fitted scaler for covariates
    result_ : AdjustmentResult
        Results from most recent fit

    Examples
    --------
    >>> from population_adjustment import STC
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
    >>> # Target population IPD or aggregate data
    >>> target = pd.DataFrame({
    ...     'age': np.random.normal(65, 10, 150),
    ...     'sex': np.random.binomial(1, 0.6, 150),
    ... })
    >>>
    >>> # Fit STC
    >>> stc = STC(outcome_model='logistic', include_interaction=True)
    >>> result = stc.fit(
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
        outcome_model: str = 'logistic',
        include_interaction: bool = True,
        ci_level: float = 0.95,
        random_state: Optional[int] = None,
        bootstrap_samples: int = 1000
    ):
        super().__init__(method=method, ci_level=ci_level, random_state=random_state)

        if outcome_model not in ['linear', 'logistic', 'random_forest']:
            raise ValueError(
                "outcome_model must be 'linear', 'logistic', or 'random_forest'"
            )

        self.outcome_model_type = outcome_model
        self.include_interaction = include_interaction
        self.bootstrap_samples = bootstrap_samples

        self.outcome_model_: Optional[object] = None
        self.scaler_: Optional[StandardScaler] = None
        self.result_: Optional[AdjustmentResult] = None

    def fit(
        self,
        ipd_data: pd.DataFrame,
        aggregate_data: pd.DataFrame,
        covariates: List[str],
        outcome: str,
        treatment: str,
        outcome_type: str = 'binary',
        target_is_ipd: bool = True,
        **kwargs
    ) -> AdjustmentResult:
        """
        Fit STC to estimate population-adjusted treatment effect.

        Parameters
        ----------
        ipd_data : pd.DataFrame
            Individual patient data from index trial
        aggregate_data : pd.DataFrame
            Either IPD from target population or aggregate statistics
        covariates : List[str]
            Names of covariates for outcome model
        outcome : str
            Name of outcome variable
        treatment : str
            Name of treatment variable
        outcome_type : str
            Type of outcome: 'binary' or 'continuous'
        target_is_ipd : bool
            Whether aggregate_data contains IPD (True) or summary stats (False)
        **kwargs
            Additional parameters

        Returns
        -------
        AdjustmentResult
            Results object with adjusted estimates
        """
        # Validate inputs
        self._validate_data(ipd_data, aggregate_data, covariates, outcome, treatment)

        # Store IPD data for bootstrap (fixes variance estimation)
        self._ipd_data_for_bootstrap = ipd_data.copy()
        self._outcome_for_bootstrap = outcome
        self._treatment_for_bootstrap = treatment

        # Fit outcome model on IPD
        if self.method == 'frequentist':
            self.outcome_model_ = self._fit_outcome_model_frequentist(
                ipd_data, covariates, outcome, treatment, outcome_type
            )
        else:  # Bayesian
            self.outcome_model_ = self._fit_outcome_model_bayesian(
                ipd_data, covariates, outcome, treatment, outcome_type
            )

        # Predict counterfactual outcomes in target population
        if target_is_ipd:
            effect_estimate, std_error, ci, predictions = self._predict_from_ipd(
                aggregate_data, covariates, treatment, outcome_type
            )
        else:
            effect_estimate, std_error, ci, predictions = self._predict_from_aggregate(
                aggregate_data, covariates, treatment, outcome_type
            )

        # Calculate diagnostics
        diagnostics = self._calculate_diagnostics(
            ipd_data, aggregate_data, covariates, target_is_ipd
        )

        # Create result object
        self.result_ = AdjustmentResult(
            effect_estimate=effect_estimate,
            standard_error=std_error,
            ci_lower=ci[0],
            ci_upper=ci[1],
            method_name=f"STC ({self.method}, {self.outcome_model_type})",
            diagnostics=diagnostics,
            predictions=predictions,
            ipd_data=ipd_data.copy()
        )

        self._result = self.result_

        return self.result_

    def _fit_outcome_model_frequentist(
        self,
        ipd_data: pd.DataFrame,
        covariates: List[str],
        outcome: str,
        treatment: str,
        outcome_type: str
    ) -> object:
        """
        Fit outcome regression model using frequentist approach.

        Parameters
        ----------
        ipd_data : pd.DataFrame
            IPD data
        covariates : List[str]
            Covariate names
        outcome : str
            Outcome variable
        treatment : str
            Treatment variable
        outcome_type : str
            Type of outcome

        Returns
        -------
        object
            Fitted model
        """
        # Prepare features
        X = ipd_data[covariates].values
        trt = ipd_data[treatment].values.reshape(-1, 1)
        y = ipd_data[outcome].values

        # Standardize covariates
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Create feature matrix with treatment
        if self.include_interaction:
            # Include main effects + treatment + interactions
            X_full = np.hstack([
                X_scaled,
                trt,
                X_scaled * trt  # Interactions
            ])
        else:
            # Only main effects + treatment
            X_full = np.hstack([X_scaled, trt])

        # Fit appropriate model
        if self.outcome_model_type == 'logistic' or outcome_type == 'binary':
            model = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='lbfgs'
            )
        elif self.outcome_model_type == 'linear' or outcome_type == 'continuous':
            model = LinearRegression()
        elif self.outcome_model_type == 'random_forest':
            if outcome_type == 'binary':
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    max_depth=10
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.random_state,
                    max_depth=10
                )
        else:
            raise ValueError(f"Unknown outcome model type: {self.outcome_model_type}")

        model.fit(X_full, y)

        return model

    def _fit_outcome_model_bayesian(
        self,
        ipd_data: pd.DataFrame,
        covariates: List[str],
        outcome: str,
        treatment: str,
        outcome_type: str
    ) -> object:
        """
        Fit outcome regression model using Bayesian approach with PyMC.

        Parameters
        ----------
        ipd_data : pd.DataFrame
            IPD data
        covariates : List[str]
            Covariate names
        outcome : str
            Outcome variable
        treatment : str
            Treatment variable
        outcome_type : str
            Type of outcome

        Returns
        -------
        object
            Dictionary with model and trace
        """
        try:
            import pymc as pm
            import pytensor.tensor as pt
        except ImportError:
            raise ImportError(
                "PyMC is required for Bayesian methods. "
                "Install with: pip install pymc>=5.0"
            )

        # Prepare features
        X = ipd_data[covariates].values
        trt = ipd_data[treatment].values
        y = ipd_data[outcome].values

        # Standardize
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        n_covariates = X_scaled.shape[1]

        with pm.Model() as model:
            # Priors for coefficients
            beta_cov = pm.Normal('beta_cov', mu=0, sigma=2, shape=n_covariates)
            beta_trt = pm.Normal('beta_trt', mu=0, sigma=2)

            if self.include_interaction:
                beta_int = pm.Normal('beta_int', mu=0, sigma=1, shape=n_covariates)
                linear_pred = (
                    pm.math.dot(X_scaled, beta_cov) +
                    beta_trt * trt +
                    pm.math.dot(X_scaled * trt[:, None], beta_int)
                )
            else:
                linear_pred = pm.math.dot(X_scaled, beta_cov) + beta_trt * trt

            if outcome_type == 'binary':
                # Logistic regression
                p = pm.math.sigmoid(linear_pred)
                _ = pm.Bernoulli('obs', p=p, observed=y)
            else:
                # Linear regression
                sigma = pm.HalfNormal('sigma', sigma=1)
                _ = pm.Normal('obs', mu=linear_pred, sigma=sigma, observed=y)

            # Sample
            trace = pm.sample(
                1000,
                tune=1000,
                random_seed=self.random_state,
                progressbar=False,
                return_inferencedata=False
            )

        return {'model': model, 'trace': trace, 'n_covariates': n_covariates}

    def _predict_from_ipd(
        self,
        target_data: pd.DataFrame,
        covariates: List[str],
        treatment: str,
        outcome_type: str
    ) -> Tuple[float, float, Tuple[float, float], np.ndarray]:
        """
        Predict outcomes and estimate effect when target is IPD.

        Uses G-computation: predict outcomes under treatment and control
        for each individual in target population.

        Parameters
        ----------
        target_data : pd.DataFrame
            Target population IPD
        covariates : List[str]
            Covariate names
        treatment : str
            Treatment variable
        outcome_type : str
            Type of outcome

        Returns
        -------
        Tuple[float, float, Tuple[float, float], np.ndarray]
            Effect estimate, SE, CI, predictions
        """
        X_target = target_data[covariates].values
        X_target_scaled = self.scaler_.transform(X_target)
        n_target = len(target_data)

        if self.method == 'frequentist':
            # Predict under treatment
            if self.include_interaction:
                X_trt1 = np.hstack([
                    X_target_scaled,
                    np.ones((n_target, 1)),
                    X_target_scaled  # Interactions with treatment=1
                ])
                X_trt0 = np.hstack([
                    X_target_scaled,
                    np.zeros((n_target, 1)),
                    np.zeros_like(X_target_scaled)  # Interactions with treatment=0
                ])
            else:
                X_trt1 = np.hstack([X_target_scaled, np.ones((n_target, 1))])
                X_trt0 = np.hstack([X_target_scaled, np.zeros((n_target, 1))])

            # Get predictions
            if outcome_type == 'binary':
                y_pred_1 = self.outcome_model_.predict_proba(X_trt1)[:, 1]
                y_pred_0 = self.outcome_model_.predict_proba(X_trt0)[:, 1]
            else:
                y_pred_1 = self.outcome_model_.predict(X_trt1)
                y_pred_0 = self.outcome_model_.predict(X_trt0)

            # Marginal effect
            effect = np.mean(y_pred_1 - y_pred_0)

            # Bootstrap for variance
            std_error = self._bootstrap_variance(
                target_data, covariates, outcome_type
            )

            predictions = np.column_stack([y_pred_0, y_pred_1])

        else:  # Bayesian
            # Sample from posterior predictive
            y_pred_1_samples = []
            y_pred_0_samples = []

            trace = self.outcome_model_['trace']
            n_samples = len(trace['beta_trt'])

            for i in range(min(n_samples, 1000)):
                beta_cov = trace['beta_cov'][i]
                beta_trt = trace['beta_trt'][i]

                if self.include_interaction:
                    beta_int = trace['beta_int'][i]
                    linear_pred_1 = (
                        X_target_scaled @ beta_cov +
                        beta_trt +
                        (X_target_scaled * 1) @ beta_int
                    )
                    linear_pred_0 = X_target_scaled @ beta_cov
                else:
                    linear_pred_1 = X_target_scaled @ beta_cov + beta_trt
                    linear_pred_0 = X_target_scaled @ beta_cov

                if outcome_type == 'binary':
                    y_pred_1_samples.append(1 / (1 + np.exp(-linear_pred_1)))
                    y_pred_0_samples.append(1 / (1 + np.exp(-linear_pred_0)))
                else:
                    y_pred_1_samples.append(linear_pred_1)
                    y_pred_0_samples.append(linear_pred_0)

            y_pred_1_samples = np.array(y_pred_1_samples)
            y_pred_0_samples = np.array(y_pred_0_samples)

            # Marginal effects from each posterior sample
            effects = np.mean(y_pred_1_samples - y_pred_0_samples, axis=1)

            effect = np.mean(effects)
            std_error = np.std(effects)

            predictions = np.column_stack([
                np.mean(y_pred_0_samples, axis=0),
                np.mean(y_pred_1_samples, axis=0)
            ])

        # Confidence interval
        z_alpha = norm.ppf(1 - (1 - self.ci_level) / 2)
        ci_lower = effect - z_alpha * std_error
        ci_upper = effect + z_alpha * std_error

        return effect, std_error, (ci_lower, ci_upper), predictions

    def _predict_from_aggregate(
        self,
        aggregate_data: pd.DataFrame,
        covariates: List[str],
        treatment: str,
        outcome_type: str
    ) -> Tuple[float, float, Tuple[float, float], np.ndarray]:
        """
        Predict outcomes when only aggregate statistics available.

        Simulates a population matching the aggregate statistics,
        then applies G-computation.

        Parameters
        ----------
        aggregate_data : pd.DataFrame
            Aggregate statistics (means, SDs)
        covariates : List[str]
            Covariate names
        treatment : str
            Treatment variable
        outcome_type : str
            Type of outcome

        Returns
        -------
        Tuple[float, float, Tuple[float, float], np.ndarray]
            Effect estimate, SE, CI, predictions
        """
        # Simulate population from aggregate statistics
        # Assume multivariate normal for continuous, binomial for binary
        n_sim = 10000

        X_sim = []
        for cov in covariates:
            if cov in aggregate_data.columns:
                mean = aggregate_data[cov].iloc[0]
                # Try to get SD if available
                sd_col = f"{cov}_sd"
                if sd_col in aggregate_data.columns:
                    sd = aggregate_data[sd_col].iloc[0]
                else:
                    sd = mean * 0.3  # Assume CV = 30% if SD not provided

                X_sim.append(np.random.normal(mean, sd, n_sim))

        X_sim = np.column_stack(X_sim)

        # Create simulated IPD
        sim_data = pd.DataFrame(X_sim, columns=covariates)

        # Use IPD prediction method
        return self._predict_from_ipd(sim_data, covariates, treatment, outcome_type)

    def _bootstrap_variance(
        self,
        target_data: pd.DataFrame,
        covariates: List[str],
        outcome_type: str
    ) -> float:
        """
        Estimate variance using bootstrap by resampling IPD and refitting model.

        CORRECTED: Now resamples IPD trial data and refits outcome model each time,
        rather than just resampling target population. This properly accounts for
        uncertainty in model parameter estimation.

        Parameters
        ----------
        target_data : pd.DataFrame
            Target population data
        covariates : List[str]
            Covariate names
        outcome_type : str
            Type of outcome

        Returns
        -------
        float
            Standard error
        """
        # Need access to original IPD data - store during fit
        if not hasattr(self, '_ipd_data_for_bootstrap'):
            # Fallback: use simpler variance estimate
            import warnings
            warnings.warn(
                "IPD data not available for bootstrap. Using approximation. "
                "Results may underestimate uncertainty."
            )
            return self._bootstrap_variance_target_only(target_data, covariates, outcome_type)

        ipd_data = self._ipd_data_for_bootstrap
        outcome = self._outcome_for_bootstrap
        treatment = self._treatment_for_bootstrap

        n_ipd = len(ipd_data)
        effects = []

        for _ in range(self.bootstrap_samples):
            try:
                # Resample IPD trial data
                boot_idx = np.random.choice(n_ipd, size=n_ipd, replace=True)
                ipd_boot = ipd_data.iloc[boot_idx].reset_index(drop=True)

                # Refit outcome model on bootstrap IPD sample
                boot_model = self._fit_outcome_model_frequentist(
                    ipd_boot, covariates, outcome, treatment, outcome_type
                )

                # Predict on target population using bootstrap model
                X_target = target_data[covariates].values
                X_target_scaled = self.scaler_.transform(X_target)
                n_target = len(target_data)

                if self.include_interaction:
                    X_trt1 = np.hstack([
                        X_target_scaled,
                        np.ones((n_target, 1)),
                        X_target_scaled
                    ])
                    X_trt0 = np.hstack([
                        X_target_scaled,
                        np.zeros((n_target, 1)),
                        np.zeros_like(X_target_scaled)
                    ])
                else:
                    X_trt1 = np.hstack([X_target_scaled, np.ones((n_target, 1))])
                    X_trt0 = np.hstack([X_target_scaled, np.zeros((n_target, 1))])

                if outcome_type == 'binary':
                    y_pred_1 = boot_model.predict_proba(X_trt1)[:, 1]
                    y_pred_0 = boot_model.predict_proba(X_trt0)[:, 1]
                else:
                    y_pred_1 = boot_model.predict(X_trt1)
                    y_pred_0 = boot_model.predict(X_trt0)

                effects.append(np.mean(y_pred_1 - y_pred_0))

            except:
                # Skip failed bootstrap samples
                continue

        if len(effects) < self.bootstrap_samples * 0.5:
            import warnings
            warnings.warn(
                f"Only {len(effects)}/{self.bootstrap_samples} bootstrap samples succeeded. "
                "Variance estimate may be unreliable."
            )

        return np.std(effects)

    def _bootstrap_variance_target_only(
        self,
        target_data: pd.DataFrame,
        covariates: List[str],
        outcome_type: str
    ) -> float:
        """
        Fallback bootstrap that only resamples target (underestimates variance).

        This is the OLD INCORRECT method. Only used if IPD not available.
        """
        n = len(target_data)
        effects = []

        for _ in range(min(self.bootstrap_samples, 200)):  # Reduced for speed
            idx = np.random.choice(n, size=n, replace=True)
            boot_data = target_data.iloc[idx]

            X_boot = boot_data[covariates].values
            X_boot_scaled = self.scaler_.transform(X_boot)

            if self.include_interaction:
                X_trt1 = np.hstack([
                    X_boot_scaled,
                    np.ones((n, 1)),
                    X_boot_scaled
                ])
                X_trt0 = np.hstack([
                    X_boot_scaled,
                    np.zeros((n, 1)),
                    np.zeros_like(X_boot_scaled)
                ])
            else:
                X_trt1 = np.hstack([X_boot_scaled, np.ones((n, 1))])
                X_trt0 = np.hstack([X_boot_scaled, np.zeros((n, 1))])

            if outcome_type == 'binary':
                y_pred_1 = self.outcome_model_.predict_proba(X_trt1)[:, 1]
                y_pred_0 = self.outcome_model_.predict_proba(X_trt0)[:, 1]
            else:
                y_pred_1 = self.outcome_model_.predict(X_trt1)
                y_pred_0 = self.outcome_model_.predict(X_trt0)

            effects.append(np.mean(y_pred_1 - y_pred_0))

        return np.std(effects)

    def _calculate_diagnostics(
        self,
        ipd_data: pd.DataFrame,
        target_data: pd.DataFrame,
        covariates: List[str],
        target_is_ipd: bool
    ) -> Dict:
        """
        Calculate diagnostic metrics.

        Parameters
        ----------
        ipd_data : pd.DataFrame
            Index trial IPD
        target_data : pd.DataFrame
            Target population data
        covariates : List[str]
            Covariate names
        target_is_ipd : bool
            Whether target is IPD

        Returns
        -------
        Dict
            Diagnostic metrics
        """
        diagnostics = {
            'n_ipd': len(ipd_data),
            'n_covariates': len(covariates),
            'include_interaction': self.include_interaction
        }

        if target_is_ipd:
            diagnostics['n_target'] = len(target_data)

            # Calculate covariate balance
            X_ipd = ipd_data[covariates].values
            X_target = target_data[covariates].values

            balance_metrics = {}
            for i, cov in enumerate(covariates):
                smd = self.calculate_smd(
                    X_ipd[:, i], X_target[:, i]
                )
                balance_metrics[cov] = {
                    'ipd_mean': float(np.mean(X_ipd[:, i])),
                    'target_mean': float(np.mean(X_target[:, i])),
                    'smd': float(smd)
                }

            diagnostics['balance_metrics'] = balance_metrics

        # Model diagnostics
        if hasattr(self.outcome_model_, 'score'):
            # For sklearn models
            diagnostics['model_type'] = self.outcome_model_type
        elif isinstance(self.outcome_model_, dict):
            # For Bayesian models
            diagnostics['model_type'] = f'bayesian_{self.outcome_model_type}'

        return diagnostics

    def _validate_data(
        self,
        ipd_data: pd.DataFrame,
        aggregate_data: pd.DataFrame,
        covariates: List[str],
        outcome: str,
        treatment: str
    ) -> None:
        """Validate input data for STC."""
        # Check IPD has required columns
        required_cols = covariates + [outcome, treatment]
        missing = set(required_cols) - set(ipd_data.columns)
        if missing:
            raise ValueError(f"IPD data missing columns: {missing}")

        # Check aggregate/target data has covariate columns
        missing_agg = set(covariates) - set(aggregate_data.columns)
        if missing_agg:
            raise ValueError(f"Target data missing columns: {missing_agg}")

        # Check for missing values in IPD
        if ipd_data[required_cols].isnull().any().any():
            raise ValueError("IPD data contains missing values")

        # Check treatment is binary
        unique_trt = ipd_data[treatment].unique()
        if not set(unique_trt).issubset({0, 1}):
            raise ValueError("Treatment variable must be binary (0/1)")
