"""
Tests for MAIC (Matching-Adjusted Indirect Comparison)
"""

import pytest
import numpy as np
import pandas as pd
from population_adjustment.core.maic import MAIC


@pytest.fixture
def simple_data():
    """Create simple test data."""
    np.random.seed(42)

    # Trial IPD
    ipd = pd.DataFrame({
        'age': np.random.normal(60, 10, 100),
        'sex': np.random.binomial(1, 0.5, 100),
        'treatment': np.random.binomial(1, 0.5, 100),
        'response': np.random.binomial(1, 0.4, 100)
    })

    # Target aggregate stats
    aggregate = pd.DataFrame({
        'age': [65],
        'sex': [0.6]
    })

    return ipd, aggregate


def test_maic_initialization():
    """Test MAIC initialization."""
    maic = MAIC(method='frequentist', weighting_method='standard')
    assert maic.method == 'frequentist'
    assert maic.weighting_method == 'standard'

    # Test invalid method
    with pytest.raises(ValueError):
        MAIC(method='invalid')

    # Test invalid weighting
    with pytest.raises(ValueError):
        MAIC(weighting_method='invalid')


def test_maic_fit(simple_data):
    """Test MAIC fitting."""
    ipd, aggregate = simple_data

    maic = MAIC(method='frequentist', random_state=42)
    result = maic.fit(
        ipd_data=ipd,
        aggregate_data=aggregate,
        covariates=['age', 'sex'],
        outcome='response',
        treatment='treatment',
        outcome_type='binary'
    )

    # Check result attributes
    assert hasattr(result, 'effect_estimate')
    assert hasattr(result, 'standard_error')
    assert hasattr(result, 'ci_lower')
    assert hasattr(result, 'ci_upper')
    assert result.method_name.startswith('MAIC')

    # Check weights
    assert result.weights is not None
    assert len(result.weights) == len(ipd)
    assert np.all(result.weights > 0)

    # Check diagnostics
    assert 'effective_sample_size' in result.diagnostics
    assert 'balance_metrics' in result.diagnostics
    assert result.effective_sample_size > 0
    assert result.effective_sample_size <= len(ipd)


def test_maic_weights_sum(simple_data):
    """Test that weights are properly normalized."""
    ipd, aggregate = simple_data

    maic = MAIC(method='frequentist', random_state=42)
    maic.fit(
        ipd_data=ipd,
        aggregate_data=aggregate,
        covariates=['age', 'sex'],
        outcome='response',
        treatment='treatment'
    )

    # Weights should sum to n
    n = len(ipd)
    assert np.abs(np.sum(maic.weights_) - n) < 1e-6


def test_maic_balance(simple_data):
    """Test that MAIC achieves balance."""
    ipd, aggregate = simple_data

    maic = MAIC(method='frequentist', random_state=42)
    result = maic.fit(
        ipd_data=ipd,
        aggregate_data=aggregate,
        covariates=['age', 'sex'],
        outcome='response',
        treatment='treatment'
    )

    # Check balance metrics
    balance = result.diagnostics['balance_metrics']

    # Weighted means should match target (within tolerance)
    w_norm = result.weights / np.sum(result.weights)
    weighted_age = np.sum(ipd['age'].values * w_norm)
    weighted_sex = np.sum(ipd['sex'].values * w_norm)

    assert np.abs(weighted_age - aggregate['age'].iloc[0]) < 0.1
    assert np.abs(weighted_sex - aggregate['sex'].iloc[0]) < 0.01


def test_maic_missing_data():
    """Test error handling for missing data."""
    ipd = pd.DataFrame({
        'age': [60, 65, np.nan, 70],
        'sex': [0, 1, 0, 1],
        'treatment': [0, 1, 0, 1],
        'response': [0, 1, 1, 0]
    })

    aggregate = pd.DataFrame({'age': [65], 'sex': [0.5]})

    maic = MAIC()

    with pytest.raises(ValueError, match="missing values"):
        maic.fit(
            ipd_data=ipd,
            aggregate_data=aggregate,
            covariates=['age', 'sex'],
            outcome='response',
            treatment='treatment'
        )


def test_maic_invalid_treatment():
    """Test error handling for invalid treatment coding."""
    ipd = pd.DataFrame({
        'age': [60, 65, 70, 75],
        'sex': [0, 1, 0, 1],
        'treatment': [0, 1, 2, 1],  # Invalid: should be 0/1
        'response': [0, 1, 1, 0]
    })

    aggregate = pd.DataFrame({'age': [65], 'sex': [0.5]})

    maic = MAIC()

    with pytest.raises(ValueError, match="binary"):
        maic.fit(
            ipd_data=ipd,
            aggregate_data=aggregate,
            covariates=['age', 'sex'],
            outcome='response',
            treatment='treatment'
        )


def test_maic_weighting_methods(simple_data):
    """Test different weighting methods."""
    ipd, aggregate = simple_data

    methods = ['standard', 'entropy_balancing', 'calibration']

    for method in methods:
        maic = MAIC(weighting_method=method, random_state=42)
        result = maic.fit(
            ipd_data=ipd,
            aggregate_data=aggregate,
            covariates=['age', 'sex'],
            outcome='response',
            treatment='treatment'
        )

        assert result.weights is not None
        assert len(result.weights) == len(ipd)


def test_maic_continuous_outcome(simple_data):
    """Test MAIC with continuous outcome."""
    ipd, aggregate = simple_data

    # Add continuous outcome
    ipd['continuous_outcome'] = np.random.normal(10, 2, len(ipd))

    maic = MAIC(random_state=42)
    result = maic.fit(
        ipd_data=ipd,
        aggregate_data=aggregate,
        covariates=['age', 'sex'],
        outcome='continuous_outcome',
        treatment='treatment',
        outcome_type='continuous'
    )

    assert hasattr(result, 'effect_estimate')
    assert result.standard_error > 0


def test_maic_result_methods(simple_data):
    """Test result object methods."""
    ipd, aggregate = simple_data

    maic = MAIC(random_state=42)
    result = maic.fit(
        ipd_data=ipd,
        aggregate_data=aggregate,
        covariates=['age', 'sex'],
        outcome='response',
        treatment='treatment'
    )

    # Test summary
    summary = result.summary()
    assert isinstance(summary, str)
    assert 'MAIC' in summary

    # Test confidence interval property
    ci = result.confidence_interval
    assert len(ci) == 2
    assert ci[0] == result.ci_lower
    assert ci[1] == result.ci_upper

    # Test plots (just check they run without error)
    fig_weights = result.plot_weights()
    assert fig_weights is not None

    fig_balance = result.plot_balance(covariates=['age', 'sex'])
    assert fig_balance is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
