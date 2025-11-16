"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)


@pytest.fixture
def sample_ipd():
    """Create sample individual patient data."""
    np.random.seed(42)

    n = 200
    data = pd.DataFrame({
        'age': np.random.normal(60, 10, n),
        'sex': np.random.binomial(1, 0.5, n),
        'baseline_severity': np.random.normal(50, 15, n),
        'treatment': np.random.binomial(1, 0.5, n),
    })

    # Generate outcome
    logit_p = (
        -2 +
        0.02 * data['age'] +
        0.3 * data['sex'] +
        0.01 * data['baseline_severity'] +
        0.6 * data['treatment']
    )
    p = 1 / (1 + np.exp(-logit_p))
    data['response'] = np.random.binomial(1, p)

    return data


@pytest.fixture
def sample_target():
    """Create sample target population data."""
    np.random.seed(43)

    n = 300
    data = pd.DataFrame({
        'age': np.random.normal(65, 10, n),
        'sex': np.random.binomial(1, 0.6, n),
        'baseline_severity': np.random.normal(55, 15, n),
    })

    return data


@pytest.fixture
def sample_aggregate(sample_target):
    """Create sample aggregate statistics."""
    return pd.DataFrame({
        'age': [sample_target['age'].mean()],
        'sex': [sample_target['sex'].mean()],
        'baseline_severity': [sample_target['baseline_severity'].mean()]
    })
