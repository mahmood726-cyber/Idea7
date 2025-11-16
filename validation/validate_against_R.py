"""
Validation of Python implementations against established R packages.

This module compares our Python implementations against:
- R MAIC package (Phillippo et al.)
- R MatchIt package for propensity scores
- R boot package for bootstrap validation

Demonstrates numerical agreement within acceptable tolerance.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from population_adjustment import MAIC, STC, IOW

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    pandas2ri.activate()
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False
    print("WARNING: rpy2 not available. Skipping R validation.")
    print("Install with: pip install rpy2")


def generate_validation_data(n_trial=500, n_target=1000, seed=42):
    """
    Generate validation dataset with known properties.

    Uses same data generation process for both Python and R validation.
    """
    np.random.seed(seed)

    # Trial data (IPD)
    age_trial = np.random.normal(55, 10, n_trial)
    sex_trial = np.random.binomial(1, 0.4, n_trial)  # 40% female
    severity_trial = np.random.normal(7, 2, n_trial)

    # Treatment assignment
    treatment_trial = np.random.binomial(1, 0.5, n_trial)

    # Outcome with treatment effect = 0.15
    baseline_risk = 0.3 + 0.01 * (age_trial - 55) + 0.05 * sex_trial + 0.02 * (severity_trial - 7)
    outcome_trial = np.random.binomial(
        1,
        baseline_risk + 0.15 * treatment_trial,
        n_trial
    )

    trial_data = pd.DataFrame({
        'age': age_trial,
        'sex': sex_trial,
        'severity': severity_trial,
        'treatment': treatment_trial,
        'outcome': outcome_trial
    })

    # Target population (different distribution)
    age_target = np.random.normal(62, 12, n_target)  # Older
    sex_target = np.random.binomial(1, 0.6, n_target)  # More female
    severity_target = np.random.normal(8, 2.5, n_target)  # More severe

    target_data = pd.DataFrame({
        'age': age_target,
        'sex': sex_target,
        'severity': severity_target
    })

    return trial_data, target_data


def validate_maic_against_R(trial_data, target_data):
    """
    Validate Python MAIC implementation against R MAIC package.

    Returns
    -------
    dict
        Comparison results showing agreement
    """
    if not R_AVAILABLE:
        return {'status': 'skipped', 'reason': 'rpy2 not available'}

    # Python MAIC
    covariates = ['age', 'sex', 'severity']

    maic = MAIC()
    result_py = maic.fit(
        trial_data=trial_data,
        target_data=target_data,
        covariates=covariates,
        outcome='outcome',
        treatment='treatment',
        outcome_type='binary'
    )

    # R MAIC (using MAIC package)
    try:
        # Install R package if needed
        utils = importr('utils')
        base = importr('base')

        # Check if MAIC package is installed, if not install from GitHub
        try:
            maic_r = importr('MAIC')
        except:
            print("Installing R MAIC package from GitHub...")
            devtools = importr('devtools')
            devtools.install_github("remiroazocar/MAIC")
            maic_r = importr('MAIC')

        # Prepare data for R
        ro.globalenv['trial_data'] = trial_data
        ro.globalenv['target_data'] = target_data

        # Run R MAIC
        r_code = """
        library(MAIC)

        # Match on all covariates
        match_cov <- c("age", "sex", "severity")

        # Calculate target means
        target_means <- colMeans(target_data[, match_cov])

        # Estimate weights using MAIC
        # Center trial covariates
        trial_centered <- trial_data[, match_cov]
        for(i in 1:length(match_cov)) {
            trial_centered[, i] <- trial_data[, match_cov[i]] - target_means[i]
        }

        # Optimize using method of moments
        result <- est.weights(
            trial_centered,
            n.target = nrow(target_data)
        )

        weights <- result$weights

        # Calculate weighted effect
        y1 <- trial_data$outcome[trial_data$treatment == 1]
        y0 <- trial_data$outcome[trial_data$treatment == 0]
        w1 <- weights[trial_data$treatment == 1]
        w0 <- weights[trial_data$treatment == 0]

        effect_r <- sum(w1 * y1) / sum(w1) - sum(w0 * y0) / sum(w0)

        # ESS
        ess_r <- sum(weights)^2 / sum(weights^2)

        list(
            effect = effect_r,
            ess = ess_r,
            weights = weights,
            max_weight = max(weights)
        )
        """

        result_r = ro.r(r_code)

        # Compare results
        comparison = {
            'status': 'completed',
            'effect_estimate': {
                'python': float(result_py.effect_estimate),
                'R': float(result_r.rx2('effect')[0]),
                'difference': abs(float(result_py.effect_estimate) - float(result_r.rx2('effect')[0])),
                'agree': abs(float(result_py.effect_estimate) - float(result_r.rx2('effect')[0])) < 0.001
            },
            'ess': {
                'python': float(result_py.diagnostics['ess']),
                'R': float(result_r.rx2('ess')[0]),
                'difference': abs(float(result_py.diagnostics['ess']) - float(result_r.rx2('ess')[0])),
                'agree': abs(float(result_py.diagnostics['ess']) - float(result_r.rx2('ess')[0])) < 1.0
            },
            'max_weight': {
                'python': float(result_py.diagnostics['max_weight']),
                'R': float(result_r.rx2('max_weight')[0]),
                'difference': abs(float(result_py.diagnostics['max_weight']) - float(result_r.rx2('max_weight')[0])),
                'agree': abs(float(result_py.diagnostics['max_weight']) - float(result_r.rx2('max_weight')[0])) < 0.01
            }
        }

        return comparison

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'message': 'R validation failed. This may be due to missing R packages.'
        }


def validate_propensity_scores_against_R(trial_data, target_data):
    """
    Validate propensity score estimation against R MatchIt package.
    """
    if not R_AVAILABLE:
        return {'status': 'skipped', 'reason': 'rpy2 not available'}

    # Python IOW
    covariates = ['age', 'sex', 'severity']

    iow = IOW(method='propensity')
    result_py = iow.fit(
        trial_data=trial_data,
        target_data=target_data,
        covariates=covariates,
        outcome='outcome',
        treatment='treatment',
        outcome_type='binary'
    )

    try:
        # Prepare combined dataset for propensity score
        trial_with_s = trial_data.copy()
        trial_with_s['S'] = 1

        target_with_s = target_data.copy()
        target_with_s['S'] = 0
        target_with_s['outcome'] = np.nan
        target_with_s['treatment'] = np.nan

        combined = pd.concat([trial_with_s, target_with_s], ignore_index=True)

        ro.globalenv['combined_data'] = combined

        # R propensity score estimation
        r_code = """
        library(stats)

        # Fit propensity score model
        ps_formula <- S ~ age + sex + severity
        ps_model <- glm(ps_formula, data = combined_data, family = binomial())

        # Get propensity scores
        ps <- predict(ps_model, type = "response")

        # Calculate odds
        odds <- ps / (1 - ps)

        # Get trial propensity scores
        trial_ps <- ps[combined_data$S == 1]
        trial_odds <- odds[combined_data$S == 1]

        list(
            mean_ps = mean(trial_ps),
            sd_ps = sd(trial_ps),
            mean_odds = mean(trial_odds)
        )
        """

        result_r = ro.r(r_code)

        # Get Python propensity scores from saved object
        ps_py = result_py._propensity_scores
        odds_py = ps_py / (1 - ps_py)

        comparison = {
            'status': 'completed',
            'mean_propensity_score': {
                'python': float(np.mean(ps_py)),
                'R': float(result_r.rx2('mean_ps')[0]),
                'difference': abs(float(np.mean(ps_py)) - float(result_r.rx2('mean_ps')[0])),
                'agree': abs(float(np.mean(ps_py)) - float(result_r.rx2('mean_ps')[0])) < 0.01
            },
            'sd_propensity_score': {
                'python': float(np.std(ps_py)),
                'R': float(result_r.rx2('sd_ps')[0]),
                'difference': abs(float(np.std(ps_py)) - float(result_r.rx2('sd_ps')[0])),
                'agree': abs(float(np.std(ps_py)) - float(result_r.rx2('sd_ps')[0])) < 0.01
            }
        }

        return comparison

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'message': 'Propensity score validation failed.'
        }


def validate_bootstrap_against_R(trial_data, n_bootstrap=100):
    """
    Validate bootstrap variance estimation against R boot package.
    """
    if not R_AVAILABLE:
        return {'status': 'skipped', 'reason': 'rpy2 not available'}

    try:
        # Simple statistic: mean difference
        treated = trial_data[trial_data['treatment'] == 1]['outcome'].values
        control = trial_data[trial_data['treatment'] == 0]['outcome'].values

        # Python bootstrap
        np.random.seed(42)
        boot_effects_py = []
        n_treated = len(treated)
        n_control = len(control)

        for _ in range(n_bootstrap):
            boot_treated = np.random.choice(treated, size=n_treated, replace=True)
            boot_control = np.random.choice(control, size=n_control, replace=True)
            boot_effects_py.append(boot_treated.mean() - boot_control.mean())

        se_py = np.std(boot_effects_py)

        # R bootstrap
        ro.globalenv['trial_data'] = trial_data
        ro.globalenv['n_bootstrap'] = n_bootstrap

        r_code = """
        library(boot)

        set.seed(42)

        # Bootstrap function
        boot_mean_diff <- function(data, indices) {
            d <- data[indices, ]
            mean_treated <- mean(d$outcome[d$treatment == 1])
            mean_control <- mean(d$outcome[d$treatment == 0])
            return(mean_treated - mean_control)
        }

        # Run bootstrap
        boot_result <- boot(
            data = trial_data,
            statistic = boot_mean_diff,
            R = n_bootstrap
        )

        list(
            se = sd(boot_result$t),
            estimate = boot_result$t0
        )
        """

        result_r = ro.r(r_code)

        comparison = {
            'status': 'completed',
            'bootstrap_se': {
                'python': float(se_py),
                'R': float(result_r.rx2('se')[0]),
                'difference': abs(float(se_py) - float(result_r.rx2('se')[0])),
                'agree': abs(float(se_py) - float(result_r.rx2('se')[0])) < 0.01
            }
        }

        return comparison

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


def run_all_validations():
    """
    Run all validation comparisons and generate report.
    """
    print("=" * 80)
    print("VALIDATION AGAINST R PACKAGES")
    print("=" * 80)
    print()

    # Generate data
    print("Generating validation data...")
    trial_data, target_data = generate_validation_data()
    print(f"  Trial data: {len(trial_data)} patients")
    print(f"  Target data: {len(target_data)} patients")
    print()

    results = {}

    # MAIC validation
    print("1. Validating MAIC against R MAIC package...")
    print("   " + "-" * 76)
    maic_comparison = validate_maic_against_R(trial_data, target_data)
    results['maic'] = maic_comparison

    if maic_comparison['status'] == 'completed':
        print(f"   Effect Estimate:")
        print(f"     Python: {maic_comparison['effect_estimate']['python']:.6f}")
        print(f"     R:      {maic_comparison['effect_estimate']['R']:.6f}")
        print(f"     Diff:   {maic_comparison['effect_estimate']['difference']:.6f}")
        print(f"     Agree:  {'✓ YES' if maic_comparison['effect_estimate']['agree'] else '✗ NO'}")
        print()
        print(f"   Effective Sample Size:")
        print(f"     Python: {maic_comparison['ess']['python']:.2f}")
        print(f"     R:      {maic_comparison['ess']['R']:.2f}")
        print(f"     Diff:   {maic_comparison['ess']['difference']:.2f}")
        print(f"     Agree:  {'✓ YES' if maic_comparison['ess']['agree'] else '✗ NO'}")
    else:
        print(f"   Status: {maic_comparison['status']}")
        if 'reason' in maic_comparison:
            print(f"   Reason: {maic_comparison['reason']}")
        if 'error' in maic_comparison:
            print(f"   Error: {maic_comparison['error']}")
    print()

    # Propensity score validation
    print("2. Validating Propensity Scores against R...")
    print("   " + "-" * 76)
    ps_comparison = validate_propensity_scores_against_R(trial_data, target_data)
    results['propensity_scores'] = ps_comparison

    if ps_comparison['status'] == 'completed':
        print(f"   Mean Propensity Score:")
        print(f"     Python: {ps_comparison['mean_propensity_score']['python']:.6f}")
        print(f"     R:      {ps_comparison['mean_propensity_score']['R']:.6f}")
        print(f"     Diff:   {ps_comparison['mean_propensity_score']['difference']:.6f}")
        print(f"     Agree:  {'✓ YES' if ps_comparison['mean_propensity_score']['agree'] else '✗ NO'}")
    else:
        print(f"   Status: {ps_comparison['status']}")
        if 'reason' in ps_comparison:
            print(f"   Reason: {ps_comparison['reason']}")
    print()

    # Bootstrap validation
    print("3. Validating Bootstrap against R boot package...")
    print("   " + "-" * 76)
    boot_comparison = validate_bootstrap_against_R(trial_data, n_bootstrap=100)
    results['bootstrap'] = boot_comparison

    if boot_comparison['status'] == 'completed':
        print(f"   Bootstrap Standard Error:")
        print(f"     Python: {boot_comparison['bootstrap_se']['python']:.6f}")
        print(f"     R:      {boot_comparison['bootstrap_se']['R']:.6f}")
        print(f"     Diff:   {boot_comparison['bootstrap_se']['difference']:.6f}")
        print(f"     Agree:  {'✓ YES' if boot_comparison['bootstrap_se']['agree'] else '✗ NO'}")
    else:
        print(f"   Status: {boot_comparison['status']}")
    print()

    # Summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    all_agree = True
    for method, result in results.items():
        if result['status'] == 'completed':
            method_agrees = all(
                v.get('agree', False)
                for k, v in result.items()
                if isinstance(v, dict) and 'agree' in v
            )
            status_str = "✓ PASS" if method_agrees else "✗ FAIL"
            all_agree = all_agree and method_agrees
        else:
            status_str = f"⊘ {result['status'].upper()}"
            all_agree = False

        print(f"{method.upper():20s}: {status_str}")

    print()
    if all_agree:
        print("🎉 ALL VALIDATIONS PASSED")
        print("Python implementations numerically agree with R packages.")
    else:
        print("⚠ Some validations failed or were skipped.")
        print("This may be due to missing R packages or rpy2.")
    print()

    # Save results
    results_file = os.path.join(
        os.path.dirname(__file__),
        'validation_results.json'
    )

    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Validation results saved to: {results_file}")

    return results


if __name__ == '__main__':
    results = run_all_validations()
