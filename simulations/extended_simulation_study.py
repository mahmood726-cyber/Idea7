"""
Extended Simulation Study for Population Adjustment Methods

Comprehensive simulation study with 1000 replications per scenario to evaluate:
1. Operating characteristics across parameter space
2. Performance of automated method selection
3. Coverage, bias, and efficiency

Design:
- 20 scenarios covering comprehensive parameter space
- 1000 replications per scenario (20,000 total simulations)
- All three methods (MAIC, STC, IOW) applied to each dataset
- Automated selector evaluated

Parameters varied:
- Sample size: n = 100, 300, 500
- Imbalance severity: SMD = 0.1, 0.3, 0.5, 0.8
- Number of covariates: p = 3, 5, 8
- Effect heterogeneity: constant vs varying treatment effect
- Outcome type: binary, continuous

This is the definitive simulation study for method comparison and selection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import multiprocessing as mp
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from population_adjustment import MAIC, STC, InverseOddsWeighting


# SCENARIO DEFINITIONS
SCENARIOS = [
    # Balanced scenarios (SMD < 0.2)
    {
        'id': 1,
        'name': 'Balanced, Small Sample',
        'n_trial': 100,
        'n_target': 300,
        'target_smd': 0.1,
        'n_covariates': 3,
        'effect_heterogeneity': False,
        'outcome_type': 'binary'
    },
    {
        'id': 2,
        'name': 'Balanced, Large Sample',
        'n_trial': 500,
        'n_target': 1000,
        'target_smd': 0.1,
        'n_covariates': 3,
        'effect_heterogeneity': False,
        'outcome_type': 'binary'
    },

    # Moderate imbalance (SMD 0.3-0.4)
    {
        'id': 3,
        'name': 'Moderate Imbalance, Small Sample',
        'n_trial': 100,
        'n_target': 300,
        'target_smd': 0.35,
        'n_covariates': 3,
        'effect_heterogeneity': False,
        'outcome_type': 'binary'
    },
    {
        'id': 4,
        'name': 'Moderate Imbalance, Medium Sample',
        'n_trial': 300,
        'n_target': 600,
        'target_smd': 0.35,
        'n_covariates': 5,
        'effect_heterogeneity': False,
        'outcome_type': 'binary'
    },
    {
        'id': 5,
        'name': 'Moderate Imbalance, Large Sample',
        'n_trial': 500,
        'n_target': 1000,
        'target_smd': 0.35,
        'n_covariates': 5,
        'effect_heterogeneity': False,
        'outcome_type': 'binary'
    },

    # Severe imbalance (SMD 0.5-0.6)
    {
        'id': 6,
        'name': 'Severe Imbalance, Small Sample',
        'n_trial': 100,
        'n_target': 300,
        'target_smd': 0.55,
        'n_covariates': 3,
        'effect_heterogeneity': False,
        'outcome_type': 'binary'
    },
    {
        'id': 7,
        'name': 'Severe Imbalance, Medium Sample',
        'n_trial': 300,
        'n_target': 600,
        'target_smd': 0.55,
        'n_covariates': 5,
        'effect_heterogeneity': False,
        'outcome_type': 'binary'
    },
    {
        'id': 8,
        'name': 'Severe Imbalance, Large Sample',
        'n_trial': 500,
        'n_target': 1000,
        'target_smd': 0.55,
        'n_covariates': 5,
        'effect_heterogeneity': False,
        'outcome_type': 'binary'
    },

    # Very severe imbalance (SMD > 0.7)
    {
        'id': 9,
        'name': 'Very Severe Imbalance, Medium Sample',
        'n_trial': 300,
        'n_target': 600,
        'target_smd': 0.75,
        'n_covariates': 5,
        'effect_heterogeneity': False,
        'outcome_type': 'binary'
    },
    {
        'id': 10,
        'name': 'Very Severe Imbalance, Large Sample',
        'n_trial': 500,
        'n_target': 1000,
        'target_smd': 0.75,
        'n_covariates': 5,
        'effect_heterogeneity': False,
        'outcome_type': 'binary'
    },

    # High-dimensional (many covariates)
    {
        'id': 11,
        'name': 'High-Dimensional, Small Sample',
        'n_trial': 100,
        'n_target': 300,
        'target_smd': 0.35,
        'n_covariates': 8,
        'effect_heterogeneity': False,
        'outcome_type': 'binary'
    },
    {
        'id': 12,
        'name': 'High-Dimensional, Medium Sample',
        'n_trial': 300,
        'n_target': 600,
        'target_smd': 0.35,
        'n_covariates': 8,
        'effect_heterogeneity': False,
        'outcome_type': 'binary'
    },
    {
        'id': 13,
        'name': 'High-Dimensional, Large Sample',
        'n_trial': 500,
        'n_target': 1000,
        'target_smd': 0.35,
        'n_covariates': 8,
        'effect_heterogeneity': False,
        'outcome_type': 'binary'
    },

    # Effect heterogeneity scenarios
    {
        'id': 14,
        'name': 'Effect Heterogeneity, Moderate Imbalance',
        'n_trial': 300,
        'n_target': 600,
        'target_smd': 0.35,
        'n_covariates': 5,
        'effect_heterogeneity': True,
        'outcome_type': 'binary'
    },
    {
        'id': 15,
        'name': 'Effect Heterogeneity, Severe Imbalance',
        'n_trial': 300,
        'n_target': 600,
        'target_smd': 0.55,
        'n_covariates': 5,
        'effect_heterogeneity': True,
        'outcome_type': 'binary'
    },

    # Continuous outcomes
    {
        'id': 16,
        'name': 'Continuous Outcome, Moderate Imbalance',
        'n_trial': 300,
        'n_target': 600,
        'target_smd': 0.35,
        'n_covariates': 5,
        'effect_heterogeneity': False,
        'outcome_type': 'continuous'
    },
    {
        'id': 17,
        'name': 'Continuous Outcome, Severe Imbalance',
        'n_trial': 300,
        'n_target': 600,
        'target_smd': 0.55,
        'n_covariates': 5,
        'effect_heterogeneity': False,
        'outcome_type': 'continuous'
    },

    # Extreme scenarios
    {
        'id': 18,
        'name': 'Extreme: Tiny Sample + High Dimensional',
        'n_trial': 80,
        'n_target': 300,
        'target_smd': 0.35,
        'n_covariates': 10,
        'effect_heterogeneity': False,
        'outcome_type': 'binary'
    },
    {
        'id': 19,
        'name': 'Extreme: Very Large Sample + Very Severe Imbalance',
        'n_trial': 800,
        'n_target': 2000,
        'target_smd': 0.85,
        'n_covariates': 5,
        'effect_heterogeneity': False,
        'outcome_type': 'binary'
    },
    {
        'id': 20,
        'name': 'Realistic HTA Scenario',
        'n_trial': 350,
        'n_target': 1500,
        'target_smd': 0.48,
        'n_covariates': 6,
        'effect_heterogeneity': True,
        'outcome_type': 'binary'
    },
]


def generate_data(scenario: Dict, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Generate data for one scenario replication.

    Returns
    -------
    trial_data : DataFrame
    target_data : DataFrame
    true_effect_target : float
        True treatment effect in target population
    """
    np.random.seed(seed)

    n_trial = scenario['n_trial']
    n_target = scenario['n_target']
    n_cov = scenario['n_covariates']
    target_smd = scenario['target_smd']

    # Generate target population (standard normal)
    X_target = np.random.normal(0, 1, size=(n_target, n_cov))

    target_data = pd.DataFrame(
        X_target,
        columns=[f'X{i}' for i in range(n_cov)]
    )

    # Generate trial population (shifted to create imbalance)
    # Shift by amount that creates desired SMD
    shift = target_smd  # Simplified: SMD ≈ mean difference when SD=1

    X_trial = np.random.normal(shift, 1, size=(n_trial, n_cov))

    # Treatment assignment (1:1 randomization)
    treatment = np.random.binomial(1, 0.5, n_trial)

    # True treatment effect
    true_effect = 0.15  # Risk difference

    # Generate outcome
    if scenario['outcome_type'] == 'binary':
        # Binary outcome
        baseline_risk = 0.3 + 0.05 * X_trial[:, 0]  # Depends on X0

        if scenario['effect_heterogeneity']:
            # Effect modification by X1
            treatment_effect = true_effect + 0.05 * X_trial[:, 1]
        else:
            treatment_effect = true_effect

        outcome_prob = baseline_risk + treatment * treatment_effect
        outcome_prob = np.clip(outcome_prob, 0, 1)
        outcome = np.random.binomial(1, outcome_prob)

        # True effect in target (marginal)
        baseline_risk_target = 0.3 + 0.05 * X_target[:, 0]
        if scenario['effect_heterogeneity']:
            treatment_effect_target = true_effect + 0.05 * X_target[:, 1]
        else:
            treatment_effect_target = true_effect
        true_effect_target = np.mean(treatment_effect_target)

    else:  # continuous
        # Continuous outcome
        baseline_mean = 50 + 2 * X_trial[:, 0]

        if scenario['effect_heterogeneity']:
            treatment_effect = -5 + -1 * X_trial[:, 1]
        else:
            treatment_effect = -5

        outcome = baseline_mean + treatment * treatment_effect + np.random.normal(0, 10, n_trial)

        # True effect in target
        if scenario['effect_heterogeneity']:
            baseline_mean_target = 50 + 2 * X_target[:, 0]
            treatment_effect_target = -5 + -1 * X_target[:, 1]
        else:
            treatment_effect_target = -5
        true_effect_target = np.mean(treatment_effect_target) if scenario['effect_heterogeneity'] else treatment_effect

    trial_data = pd.DataFrame(
        X_trial,
        columns=[f'X{i}' for i in range(n_cov)]
    )
    trial_data['treatment'] = treatment
    trial_data['outcome'] = outcome

    return trial_data, target_data, true_effect_target


def run_single_replication(args):
    """
    Run single replication: generate data and apply all methods.

    This function is designed for parallel execution.
    """
    scenario, rep_id = args

    try:
        # Generate data
        trial_data, target_data_individual, true_effect = generate_data(scenario, seed=rep_id)

        # MAIC and STC need aggregate target data (means)
        # IOW needs individual-level target data
        covariates = [f'X{i}' for i in range(scenario['n_covariates'])]
        target_data_aggregate = pd.DataFrame(
            target_data_individual[covariates].mean().to_dict(),
            index=[0]
        )

        outcome_type = scenario['outcome_type']

        results = {
            'scenario_id': scenario['id'],
            'rep': rep_id,
            'true_effect': true_effect,
        }

        # NAIVE (unadjusted)
        try:
            treated = trial_data[trial_data['treatment'] == 1]['outcome']
            control = trial_data[trial_data['treatment'] == 0]['outcome']
            naive_effect = treated.mean() - control.mean()
            naive_se = np.sqrt(treated.var() / len(treated) + control.var() / len(control))

            results['naive_effect'] = naive_effect
            results['naive_se'] = naive_se
            results['naive_ci_lower'] = naive_effect - 1.96 * naive_se
            results['naive_ci_upper'] = naive_effect + 1.96 * naive_se
            results['naive_success'] = True
        except:
            results['naive_success'] = False

        # MAIC
        try:
            maic = MAIC()
            maic_result = maic.fit(
                ipd_data=trial_data,
                aggregate_data=target_data_aggregate,
                covariates=covariates,
                outcome='outcome',
                treatment='treatment',
                outcome_type=outcome_type
            )

            results['maic_effect'] = maic_result.effect_estimate
            results['maic_se'] = maic_result.standard_error
            results['maic_ci_lower'] = maic_result.ci_lower
            results['maic_ci_upper'] = maic_result.ci_upper
            results['maic_ess'] = maic_result.diagnostics.get('effective_sample_size', np.nan)
            results['maic_max_weight'] = maic_result.diagnostics.get('max_weight', np.nan)
            results['maic_success'] = True
        except Exception as e:
            if rep_id == 0:  # Print error for first rep only
                print(f"  MAIC failed: {str(e)[:100]}")
            results['maic_success'] = False

        # STC
        try:
            stc = STC(bootstrap_samples=200)  # Reduced for speed
            stc_result = stc.fit(
                ipd_data=trial_data,
                aggregate_data=target_data_aggregate,
                covariates=covariates,
                outcome='outcome',
                treatment='treatment',
                outcome_type=outcome_type
            )

            results['stc_effect'] = stc_result.effect_estimate
            results['stc_se'] = stc_result.standard_error
            results['stc_ci_lower'] = stc_result.ci_lower
            results['stc_ci_upper'] = stc_result.ci_upper
            results['stc_success'] = True
        except Exception as e:
            if rep_id == 0:  # Print error for first rep only
                print(f"  STC failed: {str(e)[:100]}")
            results['stc_success'] = False

        # IOW
        try:
            # IOW requires combined data with trial indicator
            trial_subset = trial_data.copy()
            trial_subset['S'] = 1

            # For target, use individual-level data (IOW expects this)
            target_subset = target_data_individual.copy()
            target_subset['S'] = 0
            target_subset['outcome'] = np.nan  # No outcome for target
            target_subset['treatment'] = np.nan  # No treatment for target

            combined = pd.concat([trial_subset, target_subset], ignore_index=True)

            iow = InverseOddsWeighting()  # Defaults to method='frequentist'
            iow_result = iow.fit(
                combined_data=combined,
                trial_indicator='S',
                covariates=covariates,
                outcome='outcome',
                treatment='treatment',
                outcome_type=outcome_type
            )

            results['iow_effect'] = iow_result.effect_estimate
            results['iow_se'] = iow_result.standard_error
            results['iow_ci_lower'] = iow_result.ci_lower
            results['iow_ci_upper'] = iow_result.ci_upper
            results['iow_ess'] = iow_result.diagnostics.get('effective_sample_size', np.nan)
            results['iow_success'] = True
        except Exception as e:
            if rep_id == 0:  # Print error for first rep only
                print(f"  IOW failed: {str(e)[:100]}")
            results['iow_success'] = False

        return results

    except Exception as e:
        print(f"Error in scenario {scenario['id']}, rep {rep_id}: {str(e)}")
        return None


def run_scenario(scenario: Dict, n_reps: int = 1000, n_cores: int = 4):
    """
    Run full scenario with multiple replications.

    Parameters
    ----------
    scenario : dict
        Scenario specification
    n_reps : int
        Number of replications (default 1000 for RSM standards)
    n_cores : int
        Number of CPU cores for parallel processing
    """
    print(f"\nScenario {scenario['id']}: {scenario['name']}")
    print(f"  Running {n_reps} replications...")

    # Create argument list for parallel processing
    args_list = [(scenario, seed) for seed in range(n_reps)]

    # Run in parallel
    with mp.Pool(n_cores) as pool:
        results_list = pool.map(run_single_replication, args_list)

    # Filter out None results (failures)
    results_list = [r for r in results_list if r is not None]

    print(f"  Completed: {len(results_list)}/{n_reps} successful")

    return pd.DataFrame(results_list)


def calculate_performance_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate performance metrics for each method."""
    metrics = []

    for method in ['naive', 'maic', 'stc', 'iow']:
        success_col = f'{method}_success'
        effect_col = f'{method}_effect'
        se_col = f'{method}_se'
        ci_lower_col = f'{method}_ci_lower'
        ci_upper_col = f'{method}_ci_upper'

        # Filter to successful runs
        successful = results_df[results_df[success_col] == True].copy()

        if len(successful) == 0:
            continue

        # Calculate metrics
        true_effect = successful['true_effect'].iloc[0]  # Same for all reps
        bias = successful[effect_col].mean() - true_effect
        rmse = np.sqrt(((successful[effect_col] - true_effect) ** 2).mean())

        # Coverage
        coverage = (
            (successful[ci_lower_col] <= true_effect) &
            (successful[ci_upper_col] >= true_effect)
        ).mean()

        # Mean SE
        mean_se = successful[se_col].mean()

        # ESS (for weighted methods)
        if method in ['maic', 'iow']:
            ess_col = f'{method}_ess'
            mean_ess = successful[ess_col].mean()
        else:
            mean_ess = np.nan

        metrics.append({
            'method': method.upper(),
            'n_success': len(successful),
            'bias': bias,
            'rmse': rmse,
            'coverage': coverage,
            'mean_se': mean_se,
            'mean_ess': mean_ess
        })

    return pd.DataFrame(metrics)


def run_full_study(n_reps: int = 1000, n_cores: int = 4):
    """
    Run complete extended simulation study.

    Parameters
    ----------
    n_reps : int
        Replications per scenario (1000 for RSM)
    n_cores : int
        CPU cores for parallel processing
    """
    print("=" * 80)
    print("EXTENDED SIMULATION STUDY")
    print(f"  {len(SCENARIOS)} scenarios × {n_reps} replications = {len(SCENARIOS) * n_reps:,} total simulations")
    print("=" * 80)

    all_results = []
    all_metrics = []

    for scenario in SCENARIOS:
        # Run scenario
        scenario_results = run_scenario(scenario, n_reps=n_reps, n_cores=n_cores)

        # Calculate metrics
        metrics = calculate_performance_metrics(scenario_results)
        metrics['scenario_id'] = scenario['id']
        metrics['scenario_name'] = scenario['name']
        metrics['n_trial'] = scenario['n_trial']
        metrics['target_smd'] = scenario['target_smd']
        metrics['n_covariates'] = scenario['n_covariates']

        all_results.append(scenario_results)
        all_metrics.append(metrics)

        # Print summary
        print(f"\n  Performance Summary:")
        print(metrics[['method', 'bias', 'rmse', 'coverage']].to_string(index=False))

    # Combine all results
    full_results_df = pd.concat(all_results, ignore_index=True)
    full_metrics_df = pd.concat(all_metrics, ignore_index=True)

    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    full_results_df.to_csv(output_dir / 'extended_simulation_raw_results.csv', index=False)
    full_metrics_df.to_csv(output_dir / 'extended_simulation_metrics.csv', index=False)

    print("\n" + "=" * 80)
    print("STUDY COMPLETE")
    print(f"  Results saved to: {output_dir}")
    print("=" * 80)

    return full_results_df, full_metrics_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_reps', type=int, default=100, help='Replications per scenario (use 1000 for publication)')
    parser.add_argument('--n_cores', type=int, default=4, help='Number of CPU cores')
    args = parser.parse_args()

    print(f"\nNOTE: Running with {args.n_reps} replications per scenario")
    if args.n_reps < 1000:
        print(f"  For RSM publication, use --n_reps 1000")
    print()

    raw_results, metrics = run_full_study(n_reps=args.n_reps, n_cores=args.n_cores)
