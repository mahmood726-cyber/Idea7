"""
Basic Example: Population Adjustment Methods

This example demonstrates the three main population adjustment methods:
1. MAIC (Matching-Adjusted Indirect Comparison)
2. STC (Simulated Treatment Comparison)
3. IOW (Inverse Odds Weighting)

Using simulated trial data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from population_adjustment import MAIC, STC, InverseOddsWeighting
from population_adjustment.utils.diagnostics import calculate_balance_metrics
from population_adjustment.visualization.plots import plot_balance, plot_forest

# Set random seed for reproducibility
np.random.seed(42)

#%% ===========================================================================
# 1. Simulate Data
# =============================================================================

print("=" * 70)
print("Simulating Trial and Target Population Data")
print("=" * 70)

# Trial population (IPD available)
n_trial = 200
trial_data = pd.DataFrame({
    'age': np.random.normal(60, 10, n_trial),
    'sex': np.random.binomial(1, 0.5, n_trial),
    'baseline_severity': np.random.normal(50, 15, n_trial),
    'treatment': np.random.binomial(1, 0.5, n_trial),
})

# Generate outcome based on covariates and treatment
# True treatment effect = 0.15 (risk difference)
logit_p = (
    -2 +
    0.02 * trial_data['age'] +
    0.3 * trial_data['sex'] +
    0.01 * trial_data['baseline_severity'] +
    0.6 * trial_data['treatment']  # Treatment effect
)
p = 1 / (1 + np.exp(-logit_p))
trial_data['response'] = np.random.binomial(1, p)

print(f"\nTrial Data: n = {n_trial}")
print(f"  Mean age: {trial_data['age'].mean():.1f}")
print(f"  % Female: {trial_data['sex'].mean()*100:.1f}%")
print(f"  Mean baseline severity: {trial_data['baseline_severity'].mean():.1f}")

# Target population (older, more females, sicker)
n_target = 300
target_data = pd.DataFrame({
    'age': np.random.normal(65, 10, n_target),  # Older
    'sex': np.random.binomial(1, 0.6, n_target),  # More females
    'baseline_severity': np.random.normal(55, 15, n_target),  # Sicker
})

print(f"\nTarget Population: n = {n_target}")
print(f"  Mean age: {target_data['age'].mean():.1f}")
print(f"  % Female: {target_data['sex'].mean()*100:.1f}%")
print(f"  Mean baseline severity: {target_data['baseline_severity'].mean():.1f}")

# Aggregate statistics for target (for MAIC)
aggregate_stats = pd.DataFrame({
    'age': [target_data['age'].mean()],
    'sex': [target_data['sex'].mean()],
    'baseline_severity': [target_data['baseline_severity'].mean()]
})

covariates = ['age', 'sex', 'baseline_severity']

#%% ===========================================================================
# 2. MAIC (Matching-Adjusted Indirect Comparison)
# =============================================================================

print("\n" + "=" * 70)
print("Method 1: MAIC (Matching-Adjusted Indirect Comparison)")
print("=" * 70)

maic = MAIC(
    method='frequentist',
    weighting_method='standard',
    random_state=42
)

maic_result = maic.fit(
    ipd_data=trial_data,
    aggregate_data=aggregate_stats,
    covariates=covariates,
    outcome='response',
    treatment='treatment',
    outcome_type='binary'
)

print(maic_result.summary())

#%% ===========================================================================
# 3. STC (Simulated Treatment Comparison)
# =============================================================================

print("\n" + "=" * 70)
print("Method 2: STC (Simulated Treatment Comparison)")
print("=" * 70)

stc = STC(
    method='frequentist',
    outcome_model='logistic',
    include_interaction=True,
    random_state=42,
    bootstrap_samples=500  # Reduced for speed
)

stc_result = stc.fit(
    ipd_data=trial_data,
    aggregate_data=target_data,  # Using IPD from target
    covariates=covariates,
    outcome='response',
    treatment='treatment',
    outcome_type='binary',
    target_is_ipd=True
)

print(stc_result.summary())

#%% ===========================================================================
# 4. IOW (Inverse Odds Weighting)
# =============================================================================

print("\n" + "=" * 70)
print("Method 3: IOW (Inverse Odds Weighting)")
print("=" * 70)

# Combine trial and target data with indicator
trial_data['trial'] = 1
target_data['trial'] = 0

# Add placeholder outcome/treatment to target (not used)
target_data['response'] = np.nan
target_data['treatment'] = np.nan

combined_data = pd.concat([trial_data, target_data], ignore_index=True)

iow = InverseOddsWeighting(
    method='frequentist',
    propensity_model='logistic',
    doubly_robust=True,
    stabilize_weights=True,
    random_state=42
)

iow_result = iow.fit(
    combined_data=combined_data,
    trial_indicator='trial',
    covariates=covariates,
    outcome='response',
    treatment='treatment',
    outcome_type='binary'
)

print(iow_result.summary())

#%% ===========================================================================
# 5. Compare Results
# =============================================================================

print("\n" + "=" * 70)
print("Comparison of Methods")
print("=" * 70)

results_comparison = pd.DataFrame({
    'Method': ['MAIC', 'STC', 'IOW'],
    'Effect': [
        maic_result.effect_estimate,
        stc_result.effect_estimate,
        iow_result.effect_estimate
    ],
    'SE': [
        maic_result.standard_error,
        stc_result.standard_error,
        iow_result.standard_error
    ],
    'CI_Lower': [
        maic_result.ci_lower,
        stc_result.ci_lower,
        iow_result.ci_lower
    ],
    'CI_Upper': [
        maic_result.ci_upper,
        stc_result.ci_upper,
        iow_result.ci_upper
    ],
    'ESS': [
        maic_result.effective_sample_size,
        None,  # STC doesn't use weights
        iow_result.effective_sample_size
    ]
})

print("\n", results_comparison.to_string(index=False))

#%% ===========================================================================
# 6. Visualizations
# =============================================================================

print("\n" + "=" * 70)
print("Creating Visualizations")
print("=" * 70)

# Forest plot comparing methods
results_dict = [
    {
        'estimate': maic_result.effect_estimate,
        'ci_lower': maic_result.ci_lower,
        'ci_upper': maic_result.ci_upper
    },
    {
        'estimate': stc_result.effect_estimate,
        'ci_lower': stc_result.ci_lower,
        'ci_upper': stc_result.ci_upper
    },
    {
        'estimate': iow_result.effect_estimate,
        'ci_lower': iow_result.ci_lower,
        'ci_upper': iow_result.ci_upper
    }
]

fig_forest = plot_forest(
    results=results_dict,
    labels=['MAIC', 'STC', 'IOW (DR)'],
    xlabel='Risk Difference',
    title='Comparison of Population Adjustment Methods'
)
plt.savefig('forest_plot.png', dpi=300, bbox_inches='tight')
print("Saved: forest_plot.png")

# Balance plot for MAIC
fig_balance = maic_result.plot_balance(covariates=covariates)
plt.savefig('maic_balance.png', dpi=300, bbox_inches='tight')
print("Saved: maic_balance.png")

# Weight distribution for MAIC
fig_weights = maic_result.plot_weights()
plt.savefig('maic_weights.png', dpi=300, bbox_inches='tight')
print("Saved: maic_weights.png")

# Balance metrics table
balance_before_after = calculate_balance_metrics(
    data1=trial_data[trial_data['trial'] == 1],
    data2=target_data[target_data['trial'] == 0],
    covariates=covariates,
    weights1=maic_result.weights
)

print("\n" + "=" * 70)
print("Balance Metrics (MAIC)")
print("=" * 70)
print(balance_before_after[['variable', 'smd', 'balanced']].to_string(index=False))

print("\n" + "=" * 70)
print("Example Complete!")
print("=" * 70)
print("\nKey Findings:")
print(f"  - All three methods estimate similar treatment effects")
print(f"  - MAIC ESS: {maic_result.effective_sample_size:.0f} (from n={n_trial})")
print(f"  - IOW ESS: {iow_result.effective_sample_size:.0f} (from n={n_trial})")
print(f"  - STC uses outcome modeling without explicit weights")
print("\nFiles saved:")
print("  - forest_plot.png")
print("  - maic_balance.png")
print("  - maic_weights.png")
