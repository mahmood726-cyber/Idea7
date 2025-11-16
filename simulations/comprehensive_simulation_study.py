"""
Comprehensive Simulation Study for Population Adjustment Methods

This script runs actual simulations to validate the implementation and compare
method performance across different scenarios.

Scenarios tested:
1. Balanced populations (null scenario)
2. Moderate imbalance (SMD=0.5)
3. Severe imbalance (SMD=1.0)
4. Poor overlap
5. Small sample size
6. Model misspecification

Results saved to: simulations/results/simulation_results.csv
"""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path
import sys
import time
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from population_adjustment import MAIC, STC, InverseOddsWeighting

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


def generate_data(
    n_trial=500,
    n_target=500,
    true_effect=0.15,
    imbalance=0.0,
    overlap='good',
    model_correct=True
):
    """
    Generate synthetic trial and target population data.

    Parameters
    ----------
    n_trial : int
        Trial sample size
    n_target : int
        Target population sample size
    true_effect : float
        True treatment effect (risk difference)
    imbalance : float
        Degree of covariate imbalance (SMD)
    overlap : str
        'good', 'moderate', or 'poor'
    model_correct : bool
        Whether outcome model is correctly specified

    Returns
    -------
    trial_data : pd.DataFrame
        IPD from trial
    target_data : pd.DataFrame
        IPD from target population
    target_stats : pd.DataFrame
        Aggregate statistics from target
    """
    # Generate trial data
    age_trial = np.random.normal(60, 10, n_trial)
    sex_trial = np.random.binomial(1, 0.5, n_trial)
    severity_trial = np.random.normal(50, 15, n_trial)
    treatment = np.random.binomial(1, 0.5, n_trial)

    # Generate target data with imbalance
    if overlap == 'poor':
        age_target = np.random.normal(60 + imbalance * 15, 8, n_target)
        sex_target = np.random.binomial(1, np.clip(0.5 + imbalance * 0.4, 0.1, 0.9), n_target)
        severity_target = np.random.normal(50 + imbalance * 20, 12, n_target)
    elif overlap == 'moderate':
        age_target = np.random.normal(60 + imbalance * 10, 10, n_target)
        sex_target = np.random.binomial(1, np.clip(0.5 + imbalance * 0.3, 0.1, 0.9), n_target)
        severity_target = np.random.normal(50 + imbalance * 15, 15, n_target)
    else:  # good overlap
        age_target = np.random.normal(60 + imbalance * 10, 10, n_target)
        sex_target = np.random.binomial(1, np.clip(0.5 + imbalance * 0.2, 0.1, 0.9), n_target)
        severity_target = np.random.normal(50 + imbalance * 10, 15, n_target)

    # Generate outcome with true effect
    if model_correct:
        # Logistic model (correctly specified for STC)
        logit_p = (
            -2.5 +
            0.02 * age_trial +
            0.3 * sex_trial +
            0.01 * severity_trial +
            0.6 * treatment  # True effect on log-odds scale
        )
    else:
        # Add non-linearity (model misspecification)
        logit_p = (
            -2.5 +
            0.02 * age_trial +
            0.005 * (age_trial - 60)**2 +  # Non-linear age effect
            0.3 * sex_trial +
            0.01 * severity_trial +
            0.6 * treatment
        )

    p = 1 / (1 + np.exp(-logit_p))
    response = np.random.binomial(1, p)

    # Create DataFrames
    trial_data = pd.DataFrame({
        'age': age_trial,
        'sex': sex_trial,
        'severity': severity_trial,
        'treatment': treatment,
        'response': response
    })

    target_data = pd.DataFrame({
        'age': age_target,
        'sex': sex_target,
        'severity': severity_target
    })

    target_stats = pd.DataFrame({
        'age': [target_data['age'].mean()],
        'sex': [target_data['sex'].mean()],
        'severity': [target_data['severity'].mean()]
    })

    return trial_data, target_data, target_stats


def calculate_naive_estimate(trial_data):
    """Calculate naive (unadjusted) treatment effect."""
    y1 = trial_data[trial_data['treatment'] == 1]['response'].mean()
    y0 = trial_data[trial_data['treatment'] == 0]['response'].mean()
    return y1 - y0


def run_single_simulation(scenario_params, covariates):
    """Run a single simulation iteration."""
    # Generate data
    trial_data, target_data, target_stats = generate_data(**scenario_params)

    # Calculate naive estimate
    naive_est = calculate_naive_estimate(trial_data)

    results = {'naive': naive_est}

    # MAIC
    try:
        maic = MAIC(method='frequentist', random_state=42)
        maic_result = maic.fit(
            ipd_data=trial_data,
            aggregate_data=target_stats,
            covariates=covariates,
            outcome='response',
            treatment='treatment',
            outcome_type='binary'
        )
        results['maic_est'] = maic_result.effect_estimate
        results['maic_se'] = maic_result.standard_error
        results['maic_ci_lower'] = maic_result.ci_lower
        results['maic_ci_upper'] = maic_result.ci_upper
        results['maic_ess'] = maic_result.effective_sample_size
    except:
        results.update({
            'maic_est': np.nan, 'maic_se': np.nan,
            'maic_ci_lower': np.nan, 'maic_ci_upper': np.nan, 'maic_ess': np.nan
        })

    # STC
    try:
        stc = STC(
            method='frequentist',
            outcome_model='logistic',
            include_interaction=True,
            random_state=42,
            bootstrap_samples=200  # Reduced for speed
        )
        stc_result = stc.fit(
            ipd_data=trial_data,
            aggregate_data=target_data,
            covariates=covariates,
            outcome='response',
            treatment='treatment',
            outcome_type='binary',
            target_is_ipd=True
        )
        results['stc_est'] = stc_result.effect_estimate
        results['stc_se'] = stc_result.standard_error
        results['stc_ci_lower'] = stc_result.ci_lower
        results['stc_ci_upper'] = stc_result.ci_upper
    except:
        results.update({
            'stc_est': np.nan, 'stc_se': np.nan,
            'stc_ci_lower': np.nan, 'stc_ci_upper': np.nan
        })

    # IOW-DR
    try:
        # Combine data
        trial_data_iow = trial_data.copy()
        trial_data_iow['trial'] = 1
        target_data_iow = target_data.copy()
        target_data_iow['trial'] = 0
        target_data_iow['response'] = np.nan
        target_data_iow['treatment'] = np.nan

        combined = pd.concat([trial_data_iow, target_data_iow], ignore_index=True)

        iow = InverseOddsWeighting(
            method='frequentist',
            propensity_model='logistic',
            doubly_robust=True,
            stabilize_weights=True,
            random_state=42
        )
        iow_result = iow.fit(
            combined_data=combined,
            trial_indicator='trial',
            covariates=covariates,
            outcome='response',
            treatment='treatment',
            outcome_type='binary'
        )
        results['iow_est'] = iow_result.effect_estimate
        results['iow_se'] = iow_result.standard_error
        results['iow_ci_lower'] = iow_result.ci_lower
        results['iow_ci_upper'] = iow_result.ci_upper
        results['iow_ess'] = iow_result.effective_sample_size
    except:
        results.update({
            'iow_est': np.nan, 'iow_se': np.nan,
            'iow_ci_lower': np.nan, 'iow_ci_upper': np.nan, 'iow_ess': np.nan
        })

    return results


def run_scenario(scenario_name, scenario_params, n_reps=100, covariates=['age', 'sex', 'severity']):
    """Run full simulation for one scenario."""
    print(f"\nRunning scenario: {scenario_name}")
    print(f"Parameters: {scenario_params}")
    print(f"Replications: {n_reps}")

    all_results = []

    for i in tqdm(range(n_reps), desc=scenario_name):
        # Set different seed for each replication
        np.random.seed(42 + i)

        results = run_single_simulation(scenario_params, covariates)
        results['scenario'] = scenario_name
        results['rep'] = i
        results['true_effect'] = scenario_params['true_effect']

        all_results.append(results)

    return pd.DataFrame(all_results)


def summarize_results(results_df):
    """Calculate summary statistics for each method."""
    summary = []

    for scenario in results_df['scenario'].unique():
        scenario_data = results_df[results_df['scenario'] == scenario]
        true_effect = scenario_data['true_effect'].iloc[0]

        for method in ['naive', 'maic', 'stc', 'iow']:
            est_col = f'{method}_est' if method != 'naive' else 'naive'

            if est_col in scenario_data.columns:
                estimates = scenario_data[est_col].dropna()

                if len(estimates) > 0:
                    bias = estimates.mean() - true_effect
                    rmse = np.sqrt(((estimates - true_effect) ** 2).mean())

                    # Coverage (for methods with CI)
                    if method != 'naive':
                        ci_lower_col = f'{method}_ci_lower'
                        ci_upper_col = f'{method}_ci_upper'

                        if ci_lower_col in scenario_data.columns:
                            coverage = (
                                (scenario_data[ci_lower_col] <= true_effect) &
                                (scenario_data[ci_upper_col] >= true_effect)
                            ).mean()
                        else:
                            coverage = np.nan
                    else:
                        coverage = np.nan

                    # ESS (for weighting methods)
                    if method in ['maic', 'iow']:
                        ess_col = f'{method}_ess'
                        if ess_col in scenario_data.columns:
                            ess_mean = scenario_data[ess_col].mean()
                        else:
                            ess_mean = np.nan
                    else:
                        ess_mean = np.nan

                    summary.append({
                        'scenario': scenario,
                        'method': method.upper(),
                        'mean_estimate': estimates.mean(),
                        'bias': bias,
                        'rmse': rmse,
                        'coverage': coverage,
                        'ess_mean': ess_mean,
                        'n_success': len(estimates),
                        'n_total': len(scenario_data)
                    })

    return pd.DataFrame(summary)


def main():
    """Run all simulation scenarios."""
    print("="*70)
    print("COMPREHENSIVE SIMULATION STUDY")
    print("Population Adjustment Methods Validation")
    print("="*70)

    # Define scenarios
    scenarios = {
        'Scenario 1: Balanced': {
            'n_trial': 500,
            'n_target': 500,
            'true_effect': 0.15,
            'imbalance': 0.0,
            'overlap': 'good',
            'model_correct': True
        },
        'Scenario 2: Moderate Imbalance': {
            'n_trial': 500,
            'n_target': 500,
            'true_effect': 0.15,
            'imbalance': 0.5,
            'overlap': 'good',
            'model_correct': True
        },
        'Scenario 3: Severe Imbalance': {
            'n_trial': 500,
            'n_target': 500,
            'true_effect': 0.15,
            'imbalance': 1.0,
            'overlap': 'moderate',
            'model_correct': True
        },
        'Scenario 4: Poor Overlap': {
            'n_trial': 500,
            'n_target': 500,
            'true_effect': 0.15,
            'imbalance': 0.8,
            'overlap': 'poor',
            'model_correct': True
        },
        'Scenario 5: Small Sample': {
            'n_trial': 100,
            'n_target': 100,
            'true_effect': 0.15,
            'imbalance': 0.5,
            'overlap': 'good',
            'model_correct': True
        },
        'Scenario 6: Model Misspecification': {
            'n_trial': 500,
            'n_target': 500,
            'true_effect': 0.15,
            'imbalance': 0.5,
            'overlap': 'good',
            'model_correct': False
        }
    }

    # Run simulations
    n_reps = 100  # Reduced from 1000 for reasonable runtime
    all_results = []

    start_time = time.time()

    for scenario_name, params in scenarios.items():
        scenario_results = run_scenario(scenario_name, params, n_reps=n_reps)
        all_results.append(scenario_results)

    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)

    # Save detailed results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    combined_results.to_csv(output_dir / 'simulation_results_detailed.csv', index=False)
    print(f"\nDetailed results saved to: {output_dir / 'simulation_results_detailed.csv'}")

    # Calculate and save summary
    summary = summarize_results(combined_results)
    summary.to_csv(output_dir / 'simulation_results_summary.csv', index=False)
    print(f"Summary results saved to: {output_dir / 'simulation_results_summary.csv'}")

    # Print summary table
    print("\n" + "="*70)
    print("SIMULATION RESULTS SUMMARY")
    print("="*70)
    print(summary.to_string(index=False))

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    print(f"Replications per scenario: {n_reps}")
    print(f"Total simulation runs: {len(scenarios) * n_reps}")

    print("\n" + "="*70)
    print("SIMULATION COMPLETE!")
    print("="*70)

    return combined_results, summary


if __name__ == '__main__':
    results, summary = main()
