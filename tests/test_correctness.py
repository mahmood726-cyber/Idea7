"""
Correctness tests for population adjustment methods.

These tests verify that methods:
1. Recover known treatment effects in controlled simulations
2. Achieve nominal coverage (95% CIs contain true effect ~95% of time)
3. Are unbiased (mean estimate ≈ true effect)
4. Handle edge cases correctly

These are the most important tests - they prove the methods actually work.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from population_adjustment import MAIC, STC, IOW


class TestGroundTruthRecovery:
    """Test that methods recover known true effects."""

    def generate_data_known_effect(
        self,
        n_trial=300,
        n_target=500,
        true_effect=0.15,
        imbalance='moderate',
        seed=42
    ):
        """
        Generate data with known ground truth.

        Returns trial data, target data, and true effect in target population.
        """
        np.random.seed(seed)

        # Target population
        age_target = np.random.normal(65, 10, n_target)
        sex_target = np.random.binomial(1, 0.5, n_target)

        target_data = pd.DataFrame({
            'age': age_target,
            'sex': sex_target
        })

        # Trial population (with imbalance)
        if imbalance == 'none':
            age_trial = np.random.normal(65, 10, n_trial)
            sex_trial = np.random.binomial(1, 0.5, n_trial)
        elif imbalance == 'moderate':
            age_trial = np.random.normal(58, 10, n_trial)  # Younger
            sex_trial = np.random.binomial(1, 0.4, n_trial)  # More male
        else:  # severe
            age_trial = np.random.normal(52, 8, n_trial)  # Much younger
            sex_trial = np.random.binomial(1, 0.3, n_trial)  # Mostly male

        treatment = np.random.binomial(1, 0.5, n_trial)

        # Outcome with NO effect modification (constant treatment effect)
        baseline_risk = 0.3 + 0.01 * (age_trial - 65) + 0.05 * sex_trial
        outcome = np.random.binomial(
            1,
            baseline_risk + true_effect * treatment,
            n_trial
        )

        trial_data = pd.DataFrame({
            'age': age_trial,
            'sex': sex_trial,
            'treatment': treatment,
            'outcome': outcome
        })

        # True effect in target is same (no effect modification)
        true_effect_target = true_effect

        return trial_data, target_data, true_effect_target

    def test_maic_recovers_true_effect(self):
        """Test that MAIC recovers true effect on average."""
        np.random.seed(123)
        n_sims = 50  # Reduced for speed, but still meaningful
        true_effect = 0.15
        estimates = []

        for i in range(n_sims):
            trial_data, target_data, _ = self.generate_data_known_effect(
                seed=i,
                true_effect=true_effect,
                imbalance='moderate'
            )

            maic = MAIC()
            result = maic.fit(
                trial_data=trial_data,
                target_data=target_data,
                covariates=['age', 'sex'],
                outcome='outcome',
                treatment='treatment',
                outcome_type='binary'
            )

            estimates.append(result.effect_estimate)

        mean_estimate = np.mean(estimates)
        bias = mean_estimate - true_effect

        # Check bias is small (< 0.02)
        assert abs(bias) < 0.02, f"MAIC is biased: bias = {bias:.4f}"

        # Check estimates are reasonable
        assert 0.10 < mean_estimate < 0.20, f"Mean estimate {mean_estimate:.3f} far from true {true_effect:.3f}"

    def test_stc_recovers_true_effect(self):
        """Test that STC recovers true effect on average."""
        np.random.seed(456)
        n_sims = 50
        true_effect = 0.15
        estimates = []

        for i in range(n_sims):
            trial_data, target_data, _ = self.generate_data_known_effect(
                seed=100 + i,
                true_effect=true_effect,
                imbalance='moderate'
            )

            stc = STC(bootstrap_samples=200)  # Reduced for speed
            result = stc.fit(
                trial_data=trial_data,
                target_data=target_data,
                covariates=['age', 'sex'],
                outcome='outcome',
                treatment='treatment',
                outcome_type='binary'
            )

            estimates.append(result.effect_estimate)

        mean_estimate = np.mean(estimates)
        bias = mean_estimate - true_effect

        # Check bias is small
        assert abs(bias) < 0.02, f"STC is biased: bias = {bias:.4f}"
        assert 0.10 < mean_estimate < 0.20, f"Mean estimate {mean_estimate:.3f} far from true {true_effect:.3f}"

    def test_iow_recovers_true_effect(self):
        """Test that IOW recovers true effect on average."""
        np.random.seed(789)
        n_sims = 50
        true_effect = 0.15
        estimates = []

        for i in range(n_sims):
            trial_data, target_data, _ = self.generate_data_known_effect(
                seed=200 + i,
                true_effect=true_effect,
                imbalance='moderate'
            )

            iow = IOW(method='propensity')
            result = iow.fit(
                trial_data=trial_data,
                target_data=target_data,
                covariates=['age', 'sex'],
                outcome='outcome',
                treatment='treatment',
                outcome_type='binary'
            )

            estimates.append(result.effect_estimate)

        mean_estimate = np.mean(estimates)
        bias = mean_estimate - true_effect

        # Check bias is small
        assert abs(bias) < 0.02, f"IOW is biased: bias = {bias:.4f}"
        assert 0.10 < mean_estimate < 0.20, f"Mean estimate {mean_estimate:.3f} far from true {true_effect:.3f}"


class TestCoverageProperties:
    """Test that confidence intervals have correct coverage."""

    def generate_data_for_coverage(self, n_trial=400, n_target=600, seed=42):
        """Generate single dataset for coverage testing."""
        np.random.seed(seed)

        # Target
        age_target = np.random.normal(65, 10, n_target)
        sex_target = np.random.binomial(1, 0.5, n_target)
        target_data = pd.DataFrame({'age': age_target, 'sex': sex_target})

        # Trial (imbalanced)
        age_trial = np.random.normal(58, 10, n_trial)
        sex_trial = np.random.binomial(1, 0.4, n_trial)
        treatment = np.random.binomial(1, 0.5, n_trial)

        # Outcome
        baseline_risk = 0.3 + 0.01 * (age_trial - 65) + 0.05 * sex_trial
        outcome = np.random.binomial(1, baseline_risk + 0.15 * treatment, n_trial)

        trial_data = pd.DataFrame({
            'age': age_trial,
            'sex': sex_trial,
            'treatment': treatment,
            'outcome': outcome
        })

        return trial_data, target_data

    def test_maic_coverage(self):
        """Test that MAIC 95% CIs contain true effect ~95% of time."""
        np.random.seed(111)
        true_effect = 0.15
        n_sims = 100
        coverage_count = 0

        for i in range(n_sims):
            trial_data, target_data = self.generate_data_for_coverage(seed=300 + i)

            maic = MAIC()
            result = maic.fit(
                trial_data=trial_data,
                target_data=target_data,
                covariates=['age', 'sex'],
                outcome='outcome',
                treatment='treatment',
                outcome_type='binary'
            )

            # Check if CI contains true effect
            if result.ci_lower <= true_effect <= result.ci_upper:
                coverage_count += 1

        coverage = coverage_count / n_sims

        # Coverage should be 0.95 ± 0.04 (allowing for Monte Carlo error)
        assert 0.91 < coverage < 0.99, f"MAIC coverage {coverage:.3f} is not near 0.95"

    def test_stc_coverage(self):
        """Test that STC 95% CIs contain true effect ~95% of time."""
        np.random.seed(222)
        true_effect = 0.15
        n_sims = 100
        coverage_count = 0

        for i in range(n_sims):
            trial_data, target_data = self.generate_data_for_coverage(seed=400 + i)

            stc = STC(bootstrap_samples=200)
            result = stc.fit(
                trial_data=trial_data,
                target_data=target_data,
                covariates=['age', 'sex'],
                outcome='outcome',
                treatment='treatment',
                outcome_type='binary'
            )

            if result.ci_lower <= true_effect <= result.ci_upper:
                coverage_count += 1

        coverage = coverage_count / n_sims

        # Allow slightly wider tolerance for bootstrap methods
        assert 0.90 < coverage < 0.99, f"STC coverage {coverage:.3f} is not near 0.95"

    def test_iow_coverage(self):
        """Test that IOW 95% CIs contain true effect ~95% of time."""
        np.random.seed(333)
        true_effect = 0.15
        n_sims = 100
        coverage_count = 0

        for i in range(n_sims):
            trial_data, target_data = self.generate_data_for_coverage(seed=500 + i)

            iow = IOW(method='propensity')
            result = iow.fit(
                trial_data=trial_data,
                target_data=target_data,
                covariates=['age', 'sex'],
                outcome='outcome',
                treatment='treatment',
                outcome_type='binary'
            )

            if result.ci_lower <= true_effect <= result.ci_upper:
                coverage_count += 1

        coverage = coverage_count / n_sims

        assert 0.91 < coverage < 0.99, f"IOW coverage {coverage:.3f} is not near 0.95"


class TestBiasReduction:
    """Test that methods reduce bias compared to naive analysis."""

    def generate_imbalanced_data(self, n_trial=300, n_target=500, seed=42):
        """Generate data with severe imbalance."""
        np.random.seed(seed)

        # Target: older population
        age_target = np.random.normal(70, 10, n_target)
        sex_target = np.random.binomial(1, 0.5, n_target)
        target_data = pd.DataFrame({'age': age_target, 'sex': sex_target})

        # Trial: younger, healthier
        age_trial = np.random.normal(50, 8, n_trial)
        sex_trial = np.random.binomial(1, 0.3, n_trial)
        treatment = np.random.binomial(1, 0.5, n_trial)

        # Outcome with age effect (older = higher risk)
        baseline_risk = 0.2 + 0.015 * (age_trial - 50)
        outcome = np.random.binomial(1, baseline_risk + 0.15 * treatment, n_trial)

        trial_data = pd.DataFrame({
            'age': age_trial,
            'sex': sex_trial,
            'treatment': treatment,
            'outcome': outcome
        })

        # True effect in target (same treatment effect, but different baseline)
        # Since treatment effect is constant (0.15), true effect is 0.15
        return trial_data, target_data, 0.15

    def test_adjustment_reduces_bias(self):
        """Test that adjustment reduces bias vs naive."""
        np.random.seed(444)
        n_sims = 50
        true_effect = 0.15

        naive_bias_list = []
        maic_bias_list = []

        for i in range(n_sims):
            trial_data, target_data, _ = self.generate_imbalanced_data(seed=600 + i)

            # Naive estimate
            treated = trial_data[trial_data['treatment'] == 1]['outcome']
            control = trial_data[trial_data['treatment'] == 0]['outcome']
            naive_effect = treated.mean() - control.mean()
            naive_bias_list.append(abs(naive_effect - true_effect))

            # MAIC estimate
            maic = MAIC()
            result = maic.fit(
                trial_data=trial_data,
                target_data=target_data,
                covariates=['age', 'sex'],
                outcome='outcome',
                treatment='treatment',
                outcome_type='binary'
            )
            maic_bias_list.append(abs(result.effect_estimate - true_effect))

        mean_naive_bias = np.mean(naive_bias_list)
        mean_maic_bias = np.mean(maic_bias_list)

        # MAIC should reduce bias
        assert mean_maic_bias < mean_naive_bias, \
            f"MAIC ({mean_maic_bias:.4f}) did not reduce bias vs naive ({mean_naive_bias:.4f})"

        # MAIC bias should be small
        assert mean_maic_bias < 0.05, f"MAIC still has substantial bias: {mean_maic_bias:.4f}"


class TestEdgeCases:
    """Test behavior in edge cases."""

    def test_no_imbalance_equals_naive(self):
        """When populations are identical, adjustment should ≈ naive."""
        np.random.seed(555)

        # Identical distributions
        n = 500
        age = np.random.normal(60, 10, n)
        sex = np.random.binomial(1, 0.5, n)

        # Split into trial and target
        trial_data = pd.DataFrame({
            'age': age[:300],
            'sex': sex[:300],
            'treatment': np.random.binomial(1, 0.5, 300),
            'outcome': np.random.binomial(1, 0.4, 300)
        })

        target_data = pd.DataFrame({
            'age': age[300:],
            'sex': sex[300:]
        })

        # Naive
        treated = trial_data[trial_data['treatment'] == 1]['outcome']
        control = trial_data[trial_data['treatment'] == 0]['outcome']
        naive_effect = treated.mean() - control.mean()

        # MAIC
        maic = MAIC()
        result = maic.fit(
            trial_data=trial_data,
            target_data=target_data,
            covariates=['age', 'sex'],
            outcome='outcome',
            treatment='treatment',
            outcome_type='binary'
        )

        # Should be similar (within 0.05)
        assert abs(result.effect_estimate - naive_effect) < 0.05, \
            "MAIC differs from naive when no imbalance exists"

        # ESS should be high (> 90% of sample)
        assert result.diagnostics['ess'] > 0.9 * len(trial_data), \
            "ESS should be high when no reweighting needed"

    def test_handles_small_samples(self):
        """Test that methods handle small samples without crashing."""
        np.random.seed(666)

        # Small trial
        trial_data = pd.DataFrame({
            'age': np.random.normal(60, 10, 50),
            'sex': np.random.binomial(1, 0.5, 50),
            'treatment': np.random.binomial(1, 0.5, 50),
            'outcome': np.random.binomial(1, 0.4, 50)
        })

        target_data = pd.DataFrame({
            'age': np.random.normal(65, 10, 100),
            'sex': np.random.binomial(1, 0.5, 100)
        })

        # Should not crash
        maic = MAIC()
        result = maic.fit(
            trial_data=trial_data,
            target_data=target_data,
            covariates=['age', 'sex'],
            outcome='outcome',
            treatment='treatment',
            outcome_type='binary'
        )

        # Should return valid result
        assert result.effect_estimate is not None
        assert not np.isnan(result.effect_estimate)
        assert result.se > 0

    def test_handles_continuous_outcomes(self):
        """Test methods work with continuous outcomes."""
        np.random.seed(777)

        trial_data = pd.DataFrame({
            'age': np.random.normal(60, 10, 200),
            'sex': np.random.binomial(1, 0.5, 200),
            'treatment': np.random.binomial(1, 0.5, 200),
            'outcome': np.random.normal(50, 10, 200)  # Continuous
        })

        target_data = pd.DataFrame({
            'age': np.random.normal(65, 10, 300),
            'sex': np.random.binomial(1, 0.5, 300)
        })

        # All methods should handle continuous
        for method_class in [MAIC, STC, IOW]:
            if method_class == IOW:
                method = method_class(method='propensity')
            else:
                method = method_class()

            result = method.fit(
                trial_data=trial_data,
                target_data=target_data,
                covariates=['age', 'sex'],
                outcome='outcome',
                treatment='treatment',
                outcome_type='continuous'
            )

            assert result.effect_estimate is not None
            assert not np.isnan(result.effect_estimate)


class TestPropertyBased:
    """Property-based tests - things that should always be true."""

    def test_ci_width_increases_with_smaller_sample(self):
        """Smaller samples should have wider confidence intervals."""
        np.random.seed(888)

        # Generate large dataset
        large_trial = pd.DataFrame({
            'age': np.random.normal(60, 10, 500),
            'sex': np.random.binomial(1, 0.5, 500),
            'treatment': np.random.binomial(1, 0.5, 500),
            'outcome': np.random.binomial(1, 0.4, 500)
        })

        target_data = pd.DataFrame({
            'age': np.random.normal(65, 10, 600),
            'sex': np.random.binomial(1, 0.5, 600)
        })

        # Small sample
        small_trial = large_trial.sample(n=100, random_state=42)

        # Fit both
        maic = MAIC()

        result_large = maic.fit(
            trial_data=large_trial,
            target_data=target_data,
            covariates=['age', 'sex'],
            outcome='outcome',
            treatment='treatment',
            outcome_type='binary'
        )

        result_small = maic.fit(
            trial_data=small_trial,
            target_data=target_data,
            covariates=['age', 'sex'],
            outcome='outcome',
            treatment='treatment',
            outcome_type='binary'
        )

        ci_width_large = result_large.ci_upper - result_large.ci_lower
        ci_width_small = result_small.ci_upper - result_small.ci_lower

        assert ci_width_small > ci_width_large, \
            "Smaller sample should have wider CI"

    def test_weights_sum_to_target_size(self):
        """MAIC weights should sum to target population size."""
        np.random.seed(999)

        trial_data = pd.DataFrame({
            'age': np.random.normal(60, 10, 300),
            'sex': np.random.binomial(1, 0.5, 300),
            'treatment': np.random.binomial(1, 0.5, 300),
            'outcome': np.random.binomial(1, 0.4, 300)
        })

        target_data = pd.DataFrame({
            'age': np.random.normal(65, 10, 500),
            'sex': np.random.binomial(1, 0.5, 500)
        })

        maic = MAIC()
        result = maic.fit(
            trial_data=trial_data,
            target_data=target_data,
            covariates=['age', 'sex'],
            outcome='outcome',
            treatment='treatment',
            outcome_type='binary'
        )

        # Check weights sum correctly (within numerical tolerance)
        weight_sum = np.sum(result.weights)
        expected_sum = len(target_data)

        assert abs(weight_sum - expected_sum) < 1.0, \
            f"Weights sum to {weight_sum:.2f}, expected {expected_sum}"

    def test_all_weights_positive(self):
        """All weights should be positive."""
        np.random.seed(1010)

        trial_data = pd.DataFrame({
            'age': np.random.normal(60, 10, 200),
            'sex': np.random.binomial(1, 0.5, 200),
            'treatment': np.random.binomial(1, 0.5, 200),
            'outcome': np.random.binomial(1, 0.4, 200)
        })

        target_data = pd.DataFrame({
            'age': np.random.normal(65, 10, 300),
            'sex': np.random.binomial(1, 0.5, 300)
        })

        maic = MAIC()
        result = maic.fit(
            trial_data=trial_data,
            target_data=target_data,
            covariates=['age', 'sex'],
            outcome='outcome',
            treatment='treatment',
            outcome_type='binary'
        )

        assert np.all(result.weights > 0), "Some weights are non-positive"
        assert np.all(np.isfinite(result.weights)), "Some weights are infinite or NaN"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
