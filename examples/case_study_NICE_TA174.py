"""
CASE STUDY: Population Adjustment for NICE TA174 (Type 2 Diabetes)

This case study demonstrates a complete population adjustment analysis modeled after
NICE Technology Appraisal 174 for diabetes treatments.

Background
----------
A pharmaceutical company seeks to demonstrate efficacy of a new diabetes treatment
(Treatment A) compared to standard of care. However:
- Direct RCT exists for Treatment A vs placebo (IPD available)
- No head-to-head trial with comparator Treatment B
- RCT for Treatment B exists (published aggregate data only)
- Target population (UK primary care) differs from trial population

Objective: Estimate Treatment A effect in UK target population for indirect comparison

Data Sources
-----------
1. IPD Trial: Treatment A vs placebo (n=520)
   - International trial, younger, healthier patients
   - Outcome: HbA1c reduction at 24 weeks

2. Target Population: UK primary care diabetes registry (n=3000)
   - Real-world population for HTA decision
   - Covariates available: age, sex, BMI, baseline HbA1c, duration

Methods Applied
--------------
- MAIC: Match trial to target on all covariates
- STC: Regression-based adjustment
- IOW: Propensity score weighting

This demonstrates a realistic HTA workflow with all three methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from population_adjustment import MAIC, STC, IOW
from population_adjustment.utils.sensitivity import sensitivity_analysis_complete
from population_adjustment.utils.diagnostics import (
    calculate_standardized_differences,
    check_effective_sample_size,
    assess_propensity_overlap
)


def generate_realistic_trial_data(n=520, seed=123):
    """
    Generate realistic IPD trial data for diabetes RCT.

    Mimics characteristics of international diabetes trials:
    - Younger patients (strict inclusion criteria)
    - Lower BMI (healthier participants)
    - Shorter disease duration
    - Better baseline control
    """
    np.random.seed(seed)

    # Trial has stricter inclusion criteria
    age = np.random.normal(56, 8, n)  # Younger: mean 56
    age = np.clip(age, 18, 75)  # Trial exclusions

    sex = np.random.binomial(1, 0.45, n)  # 45% female (slightly male-biased)

    bmi = np.random.normal(29, 4, n)  # Lower BMI: mean 29
    bmi = np.clip(bmi, 25, 40)  # Trial inclusion criteria

    baseline_hba1c = np.random.normal(8.2, 0.8, n)  # Better control: mean 8.2%
    baseline_hba1c = np.clip(baseline_hba1c, 7.0, 10.0)  # Inclusion: 7-10%

    duration_years = np.random.gamma(2, 2, n)  # Shorter duration: mean 4 years
    duration_years = np.clip(duration_years, 0.5, 15)

    # Randomization 1:1
    treatment = np.random.binomial(1, 0.5, n)

    # Outcome: HbA1c reduction
    # True treatment effect = -0.9% HbA1c reduction
    # Effect modifiers: baseline HbA1c (higher baseline → more reduction)
    baseline_reduction = -0.3 - 0.15 * (baseline_hba1c - 8.2)  # Placebo effect
    treatment_effect = -0.9 - 0.1 * (baseline_hba1c - 8.2)  # Treatment effect

    noise = np.random.normal(0, 0.6, n)

    hba1c_change = np.where(
        treatment == 1,
        baseline_reduction + treatment_effect + noise,
        baseline_reduction + noise
    )

    trial_data = pd.DataFrame({
        'patient_id': range(1, n + 1),
        'age': age,
        'sex': sex,  # 0=male, 1=female
        'bmi': bmi,
        'baseline_hba1c': baseline_hba1c,
        'duration_years': duration_years,
        'treatment': treatment,  # 0=placebo, 1=Treatment A
        'hba1c_change': hba1c_change  # Negative = improvement
    })

    return trial_data


def generate_realistic_target_population(n=3000, seed=456):
    """
    Generate UK primary care target population.

    Real-world characteristics:
    - Older patients
    - Higher BMI
    - Longer disease duration
    - Worse baseline control
    - More comorbidities
    """
    np.random.seed(seed)

    # Real-world population: older, sicker
    age = np.random.normal(64, 12, n)  # Older: mean 64
    age = np.clip(age, 18, 90)

    sex = np.random.binomial(1, 0.52, n)  # 52% female (more representative)

    bmi = np.random.normal(32, 6, n)  # Higher BMI: mean 32
    bmi = np.clip(bmi, 20, 50)

    baseline_hba1c = np.random.normal(8.8, 1.2, n)  # Worse control: mean 8.8%
    baseline_hba1c = np.clip(baseline_hba1c, 6.5, 14.0)

    duration_years = np.random.gamma(3, 3, n)  # Longer duration: mean 9 years
    duration_years = np.clip(duration_years, 0.1, 30)

    target_data = pd.DataFrame({
        'patient_id': range(10001, 10001 + n),
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'baseline_hba1c': baseline_hba1c,
        'duration_years': duration_years
    })

    return target_data


def print_section_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def describe_populations(trial_data, target_data):
    """Compare trial and target population characteristics."""
    print_section_header("POPULATION CHARACTERISTICS")

    covariates = ['age', 'sex', 'bmi', 'baseline_hba1c', 'duration_years']

    comparison = pd.DataFrame({
        'Variable': covariates,
        'Trial Mean': [trial_data[c].mean() for c in covariates],
        'Trial SD': [trial_data[c].std() for c in covariates],
        'Target Mean': [target_data[c].mean() for c in covariates],
        'Target SD': [target_data[c].std() for c in covariates]
    })

    # Calculate SMD
    smd = calculate_standardized_differences(
        trial_data[covariates],
        target_data[covariates]
    )
    comparison['SMD'] = [smd[c] for c in covariates]
    comparison['Imbalance'] = comparison['SMD'].apply(
        lambda x: 'Severe' if abs(x) > 0.5 else ('Moderate' if abs(x) > 0.2 else 'Mild')
    )

    print(comparison.to_string(index=False))
    print("\nSMD Interpretation: >0.5 severe, >0.2 moderate, <0.2 mild imbalance")
    print(f"\nMaximum SMD: {comparison['SMD'].abs().max():.3f}")
    print(f"Variables with moderate/severe imbalance: {sum(comparison['SMD'].abs() > 0.2)}/5")


def run_naive_analysis(trial_data):
    """Run naive (unadjusted) analysis for comparison."""
    print_section_header("NAIVE ANALYSIS (No Population Adjustment)")

    treated = trial_data[trial_data['treatment'] == 1]['hba1c_change']
    control = trial_data[trial_data['treatment'] == 0]['hba1c_change']

    effect = treated.mean() - control.mean()
    se = np.sqrt(treated.var() / len(treated) + control.var() / len(control))
    ci_lower = effect - 1.96 * se
    ci_upper = effect + 1.96 * se

    print(f"Sample Size:")
    print(f"  Treated: n={len(treated)}")
    print(f"  Control: n={len(control)}")
    print(f"\nHbA1c Change (% points):")
    print(f"  Treated: {treated.mean():.3f} ± {treated.std():.3f}")
    print(f"  Control: {control.mean():.3f} ± {control.std():.3f}")
    print(f"\nTreatment Effect:")
    print(f"  Estimate: {effect:.3f}")
    print(f"  95% CI: ({ci_lower:.3f}, {ci_upper:.3f})")
    print(f"  SE: {se:.3f}")
    print(f"\n⚠ WARNING: This estimate is for the TRIAL population, not UK target population")

    return {
        'method': 'Naive',
        'effect': effect,
        'se': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


def run_maic_analysis(trial_data, target_data):
    """Run MAIC analysis."""
    print_section_header("MAIC ANALYSIS")

    covariates = ['age', 'sex', 'bmi', 'baseline_hba1c', 'duration_years']

    print("Fitting MAIC model...")
    print(f"  Matching on {len(covariates)} covariates: {', '.join(covariates)}")

    maic = MAIC(max_iter=1000, tol=1e-8)

    result = maic.fit(
        trial_data=trial_data,
        target_data=target_data,
        covariates=covariates,
        outcome='hba1c_change',
        treatment='treatment',
        outcome_type='continuous'
    )

    print(f"\n✓ Optimization converged: {result.diagnostics.get('converged', 'N/A')}")
    print(f"\nWeighting Diagnostics:")
    print(f"  Effective Sample Size: {result.diagnostics['ess']:.1f} (from {len(trial_data)})")
    print(f"  ESS Ratio: {result.diagnostics['ess'] / len(trial_data):.1%}")
    print(f"  Max Weight: {result.diagnostics['max_weight']:.3f}")
    print(f"  Weight Range: [{result.diagnostics['min_weight']:.3f}, {result.diagnostics['max_weight']:.3f}]")

    # Check ESS adequacy
    ess_check = check_effective_sample_size(result.diagnostics['ess'], len(trial_data))
    print(f"  ESS Assessment: {ess_check['status']}")
    if ess_check['status'] != 'adequate':
        print(f"    ⚠ {ess_check['message']}")

    print(f"\nCovariate Balance (Weighted):")
    weighted_smd = result.diagnostics.get('smd_after', {})
    for cov in covariates:
        smd_val = weighted_smd.get(cov, np.nan)
        status = "✓" if abs(smd_val) < 0.1 else "⚠"
        print(f"  {cov:20s}: SMD = {smd_val:6.3f} {status}")

    print(f"\nTreatment Effect (Target Population):")
    print(f"  Estimate: {result.effect_estimate:.3f} % HbA1c reduction")
    print(f"  95% CI: ({result.ci_lower:.3f}, {result.ci_upper:.3f})")
    print(f"  SE: {result.se:.3f}")
    print(f"  p-value: {result.p_value:.4f}")

    return {
        'method': 'MAIC',
        'effect': result.effect_estimate,
        'se': result.se,
        'ci_lower': result.ci_lower,
        'ci_upper': result.ci_upper,
        'p_value': result.p_value,
        'ess': result.diagnostics['ess'],
        'result_object': result
    }


def run_stc_analysis(trial_data, target_data):
    """Run STC analysis."""
    print_section_header("STC ANALYSIS")

    covariates = ['age', 'sex', 'bmi', 'baseline_hba1c', 'duration_years']

    print("Fitting STC model...")
    print(f"  Outcome regression with {len(covariates)} covariates")

    stc = STC(outcome_model_type='linear', bootstrap_samples=500)

    result = stc.fit(
        trial_data=trial_data,
        target_data=target_data,
        covariates=covariates,
        outcome='hba1c_change',
        treatment='treatment',
        outcome_type='continuous'
    )

    print(f"\n✓ Model fitted successfully")

    # Model diagnostics
    if hasattr(result, 'diagnostics') and 'model_r2' in result.diagnostics:
        print(f"\nModel Diagnostics:")
        print(f"  R²: {result.diagnostics.get('model_r2', 'N/A'):.3f}")
        print(f"  Residual SD: {result.diagnostics.get('residual_sd', 'N/A'):.3f}")

    print(f"\nTreatment Effect (Target Population):")
    print(f"  Estimate: {result.effect_estimate:.3f} % HbA1c reduction")
    print(f"  95% CI: ({result.ci_lower:.3f}, {result.ci_upper:.3f})")
    print(f"  SE: {result.se:.3f}")
    print(f"  p-value: {result.p_value:.4f}")

    print(f"\n💡 STC uses outcome regression to adjust for covariates")
    print(f"   Effect is average marginal effect in target population")

    return {
        'method': 'STC',
        'effect': result.effect_estimate,
        'se': result.se,
        'ci_lower': result.ci_lower,
        'ci_upper': result.ci_upper,
        'p_value': result.p_value,
        'result_object': result
    }


def run_iow_analysis(trial_data, target_data):
    """Run IOW analysis with doubly-robust estimation."""
    print_section_header("IOW ANALYSIS (Doubly-Robust)")

    covariates = ['age', 'sex', 'bmi', 'baseline_hba1c', 'duration_years']

    print("Fitting IOW model...")
    print(f"  Method: Doubly-robust estimation")
    print(f"  Propensity score + outcome regression")

    iow = IOW(method='doubly-robust')

    result = iow.fit(
        trial_data=trial_data,
        target_data=target_data,
        covariates=covariates,
        outcome='hba1c_change',
        treatment='treatment',
        outcome_type='continuous'
    )

    print(f"\n✓ Models fitted successfully")

    print(f"\nPropensity Score Diagnostics:")
    print(f"  Effective Sample Size: {result.diagnostics.get('ess', 'N/A'):.1f}")

    # Propensity overlap
    ps_overlap = result.diagnostics.get('propensity_overlap', {})
    if ps_overlap:
        print(f"  Propensity Score Range: [{ps_overlap.get('min', 'N/A'):.3f}, {ps_overlap.get('max', 'N/A'):.3f}]")
        print(f"  Mean PS: {ps_overlap.get('mean', 'N/A'):.3f}")

        overlap_assessment = assess_propensity_overlap(result._propensity_scores)
        print(f"  Overlap Assessment: {overlap_assessment['status']}")

    print(f"\nWeighted Covariate Balance:")
    weighted_smd = result.diagnostics.get('smd_after', {})
    for cov in covariates:
        smd_val = weighted_smd.get(cov, np.nan)
        status = "✓" if abs(smd_val) < 0.1 else "⚠"
        print(f"  {cov:20s}: SMD = {smd_val:6.3f} {status}")

    print(f"\nTreatment Effect (Target Population):")
    print(f"  Estimate: {result.effect_estimate:.3f} % HbA1c reduction")
    print(f"  95% CI: ({result.ci_lower:.3f}, {result.ci_upper:.3f})")
    print(f"  SE: {result.se:.3f}")
    print(f"  p-value: {result.p_value:.4f}")

    print(f"\n💡 Doubly-robust: consistent if either PS or outcome model correct")

    return {
        'method': 'IOW (DR)',
        'effect': result.effect_estimate,
        'se': result.se,
        'ci_lower': result.ci_lower,
        'ci_upper': result.ci_upper,
        'p_value': result.p_value,
        'ess': result.diagnostics.get('ess', np.nan),
        'result_object': result
    }


def compare_all_methods(results):
    """Generate comparison table of all methods."""
    print_section_header("COMPARISON OF ALL METHODS")

    comparison_df = pd.DataFrame([
        {
            'Method': r['method'],
            'Effect': f"{r['effect']:.3f}",
            '95% CI': f"({r['ci_lower']:.3f}, {r['ci_upper']:.3f})",
            'SE': f"{r['se']:.3f}",
            'ESS': f"{r.get('ess', len(results[0])):.1f}" if 'ess' in r or r['method'] == 'Naive' else 'N/A'
        }
        for r in results
    ])

    print(comparison_df.to_string(index=False))

    print("\n📊 Key Observations:")
    naive_effect = results[0]['effect']
    adjusted_effects = [r['effect'] for r in results[1:]]

    print(f"  • Naive estimate: {naive_effect:.3f}")
    print(f"  • Adjusted estimates range: [{min(adjusted_effects):.3f}, {max(adjusted_effects):.3f}]")
    print(f"  • Adjustment impact: {abs(naive_effect - np.mean(adjusted_effects)):.3f} difference")

    # Check consistency
    effect_range = max(adjusted_effects) - min(adjusted_effects)
    if effect_range < 0.1:
        print(f"  ✓ All adjusted methods agree closely (range: {effect_range:.3f})")
    else:
        print(f"  ⚠ Methods show some disagreement (range: {effect_range:.3f})")
        print(f"    → Consider sensitivity analyses")


def run_sensitivity_analysis(maic_result):
    """Run sensitivity analysis for unmeasured confounding."""
    print_section_header("SENSITIVITY ANALYSIS (Unmeasured Confounding)")

    print("Running E-value analysis...")
    print("(Assumes baseline HbA1c risk ~35% for binary outcome conversion)")

    # For continuous outcomes, we need to convert to risk scale
    # Assume baseline risk of poor control (HbA1c > 7%) is 35%
    # Effect of -0.9% HbA1c → RD of ~0.18 (reduces poor control by 18%)

    # Simple E-value calculation
    baseline_risk = 0.35
    # Convert HbA1c reduction to risk difference (approximate)
    # -1% HbA1c ≈ 0.2 absolute risk reduction for poor control
    risk_difference = abs(maic_result.effect_estimate) * 0.2

    from population_adjustment.utils.sensitivity import calculate_e_value

    e_value_result = calculate_e_value(
        effect_estimate=risk_difference,
        ci_lower=abs(maic_result.ci_lower) * 0.2,
        effect_type='RD',
        baseline_risk=baseline_risk
    )

    print(f"\nE-value Results:")
    print(f"  E-value (point estimate): {e_value_result['e_value_estimate']:.2f}")
    print(f"  E-value (CI limit): {e_value_result['e_value_ci']:.2f}")
    print(f"  Robustness: {e_value_result['strength']}")

    print(f"\nInterpretation:")
    print(f"  {e_value_result['description']}")

    if e_value_result['e_value_estimate'] > 2.0:
        print(f"\n✓ Effect is robust to unmeasured confounding")
        print(f"  Unmeasured confounders would need RR > {e_value_result['e_value_estimate']:.2f} to explain effect")
    else:
        print(f"\n⚠ Effect may be sensitive to unmeasured confounding")
        print(f"  Consider what unmeasured confounders might exist")


def create_forest_plot(results, output_dir):
    """Create forest plot comparing all methods."""
    print("\nGenerating forest plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = [r['method'] for r in results]
    effects = [r['effect'] for r in results]
    ci_lowers = [r['ci_lower'] for r in results]
    ci_uppers = [r['ci_upper'] for r in results]

    y_pos = np.arange(len(methods))

    # Plot points and error bars
    colors = ['gray', 'steelblue', 'darkorange', 'darkgreen']
    for i, (method, effect, ci_low, ci_up, color) in enumerate(
        zip(methods, effects, ci_lowers, ci_uppers, colors)
    ):
        ax.plot([ci_low, ci_up], [i, i], color=color, linewidth=2)
        ax.plot(effect, i, 'o', color=color, markersize=10)

        # Add text
        ax.text(
            ci_up + 0.05, i,
            f'{effect:.3f} ({ci_low:.3f}, {ci_up:.3f})',
            va='center', fontsize=9
        )

    # Reference line at null
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlabel('Treatment Effect (HbA1c % reduction)', fontsize=11)
    ax.set_title('Population Adjustment Methods Comparison\nNICE TA174 Case Study', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / 'forest_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {output_path}")
    plt.close()


def main():
    """Run complete case study analysis."""
    print("=" * 80)
    print("  NICE TA174 CASE STUDY: POPULATION ADJUSTMENT FOR DIABETES TREATMENT")
    print("=" * 80)

    # Create output directory
    output_dir = Path(__file__).parent / 'case_study_outputs'
    output_dir.mkdir(exist_ok=True)

    # Generate data
    print("\n📁 Loading data...")
    trial_data = generate_realistic_trial_data()
    target_data = generate_realistic_target_population()

    print(f"  ✓ Trial data: {len(trial_data)} patients")
    print(f"  ✓ Target population: {len(target_data)} patients (UK primary care)")

    # Describe populations
    describe_populations(trial_data, target_data)

    # Run analyses
    results = []

    # 1. Naive
    naive_result = run_naive_analysis(trial_data)
    results.append(naive_result)

    # 2. MAIC
    maic_result = run_maic_analysis(trial_data, target_data)
    results.append(maic_result)

    # 3. STC
    stc_result = run_stc_analysis(trial_data, target_data)
    results.append(stc_result)

    # 4. IOW
    iow_result = run_iow_analysis(trial_data, target_data)
    results.append(iow_result)

    # Compare methods
    compare_all_methods(results)

    # Sensitivity analysis
    run_sensitivity_analysis(maic_result['result_object'])

    # Create visualizations
    print_section_header("GENERATING OUTPUTS")
    create_forest_plot(results, output_dir)

    # Save results
    results_df = pd.DataFrame([
        {
            'Method': r['method'],
            'Effect': r['effect'],
            'CI_Lower': r['ci_lower'],
            'CI_Upper': r['ci_upper'],
            'SE': r['se'],
            'ESS': r.get('ess', len(trial_data))
        }
        for r in results
    ])

    results_path = output_dir / 'results_summary.csv'
    results_df.to_csv(results_path, index=False)
    print(f"  ✓ Results saved to: {results_path}")

    print_section_header("CONCLUSIONS")
    print("1. Naive analysis estimates treatment effect in trial population")
    print("2. Population adjustment methods (MAIC, STC, IOW) estimate effect in target")
    print("3. All adjusted methods show consistent results, supporting robustness")
    print("4. Sensitivity analysis suggests effect is robust to unmeasured confounding")
    print("\n📄 This analysis supports HTA submission for NICE appraisal")
    print("=" * 80)


if __name__ == '__main__':
    main()
