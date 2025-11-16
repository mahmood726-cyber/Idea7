"""
Sensitivity analysis tools for unmeasured confounding.

CORRECTED Implementation using E-values (VanderWeele & Ding 2017).
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd


def calculate_e_value(
    effect_estimate: float,
    ci_lower: Optional[float] = None,
    effect_type: str = 'RD',
    baseline_risk: float = None
) -> Dict:
    """
    Calculate E-value for unmeasured confounding sensitivity.

    The E-value represents the minimum strength of association (on the risk ratio scale)
    that an unmeasured confounder would need to have with both the treatment and outcome
    to fully explain away the observed effect, conditional on measured confounders.

    Based on VanderWeele & Ding (2017) "Sensitivity Analysis in Observational Research:
    Introducing the E-Value", Annals of Internal Medicine.

    Parameters
    ----------
    effect_estimate : float
        Observed effect estimate
    ci_lower : float, optional
        Lower confidence limit
    effect_type : str
        Type of effect measure: 'RR' (risk ratio), 'OR' (odds ratio),
        'HR' (hazard ratio), 'RD' (risk difference)
    baseline_risk : float, optional
        Baseline risk (required for RD to convert to RR)

    Returns
    -------
    Dict
        Dictionary containing E-value for estimate and CI, with interpretation

    Examples
    --------
    >>> # Risk difference of 0.15 with baseline risk of 0.30
    >>> result = calculate_e_value(0.15, 0.05, effect_type='RD', baseline_risk=0.30)
    >>> print(f"E-value: {result['e_value_estimate']:.2f}")

    References
    ----------
    VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational
    research: introducing the E-value. Annals of Internal Medicine, 167(4), 268-274.
    """
    # Convert effect to risk ratio scale
    if effect_type == 'RD':
        if baseline_risk is None:
            raise ValueError("baseline_risk required for risk difference")
        # Convert RD to RR: RR = (p1 / p0) = (p0 + RD) / p0
        RR = (baseline_risk + effect_estimate) / baseline_risk
        if ci_lower is not None:
            RR_CI = (baseline_risk + ci_lower) / baseline_risk
        else:
            RR_CI = None
    elif effect_type == 'OR':
        # For rare outcomes OR ≈ RR
        RR = effect_estimate
        RR_CI = ci_lower if ci_lower is not None else None
    elif effect_type in ['RR', 'HR']:
        RR = effect_estimate
        RR_CI = ci_lower if ci_lower is not None else None
    else:
        raise ValueError(f"Unknown effect_type: {effect_type}")

    # E-value formula: RR + sqrt(RR * (RR - 1))
    # For protective effects (RR < 1), use 1/RR
    if RR < 1:
        RR_for_evalue = 1 / RR
    else:
        RR_for_evalue = RR

    if RR_for_evalue <= 1:
        e_value_est = 1.0
    else:
        e_value_est = RR_for_evalue + np.sqrt(RR_for_evalue * (RR_for_evalue - 1))

    # E-value for confidence interval
    if RR_CI is not None:
        # For CI, use the limit closer to null
        if RR_CI < 1:
            RR_CI_for_evalue = 1 / RR_CI
        else:
            RR_CI_for_evalue = RR_CI

        if RR_CI_for_evalue <= 1:
            e_value_ci = 1.0
        else:
            e_value_ci = RR_CI_for_evalue + np.sqrt(RR_CI_for_evalue * (RR_CI_for_evalue - 1))
    else:
        e_value_ci = None

    # Interpretation
    if e_value_est < 1.5:
        strength = "Weak"
        interpretation = "Small unmeasured confounding could explain away the effect"
    elif e_value_est < 2.0:
        strength = "Moderate"
        interpretation = "Moderate unmeasured confounding needed to explain away effect"
    elif e_value_est < 3.0:
        strength = "Strong"
        interpretation = "Substantial unmeasured confounding needed to explain away effect"
    else:
        strength = "Very Strong"
        interpretation = "Very strong unmeasured confounding needed to explain away effect"

    return {
        'e_value_estimate': float(e_value_est),
        'e_value_ci': float(e_value_ci) if e_value_ci is not None else None,
        'strength': strength,
        'interpretation': interpretation,
        'effect_estimate': float(effect_estimate),
        'effect_type': effect_type,
        'rr_converted': float(RR),
        'description': (
            f"An unmeasured confounder associated with both treatment and outcome "
            f"by a risk ratio of {e_value_est:.2f}-fold each, above and beyond the "
            f"measured confounders, could explain away the observed effect estimate of "
            f"{effect_estimate:.3f}, but weaker confounding could not."
        )
    }


def tipping_point_analysis(
    effect_estimate: float,
    rr_range: np.ndarray = None,
    baseline_risk: float = 0.3,
    prevalence_u: float = 0.2
) -> pd.DataFrame:
    """
    Perform tipping point analysis for unmeasured confounding.

    Explores combinations of confounder-treatment and confounder-outcome associations
    that would nullify the observed effect.

    Parameters
    ----------
    effect_estimate : float
        Observed effect estimate (risk difference)
    rr_range : np.ndarray, optional
        Range of RR values to explore
    baseline_risk : float
        Baseline risk in control group
    prevalence_u : float
        Prevalence of unmeasured confounder

    Returns
    -------
    pd.DataFrame
        Tipping point analysis results
    """
    if rr_range is None:
        rr_range = np.linspace(1.0, 4.0, 30)

    results = []

    for rr_uy in rr_range:  # Confounder-outcome association
        for rr_ux in rr_range:  # Confounder-treatment association
            # Bias formula for risk difference scale
            # Bias ≈ (RR_UY - 1) * (RR_UX - 1) * P(U) * baseline_risk
            bias = (rr_uy - 1) * (rr_ux - 1) * prevalence_u * baseline_risk

            # Adjusted effect
            adjusted_effect = effect_estimate - bias

            # Check if effect is nullified (crosses zero)
            nullified = (effect_estimate > 0 and adjusted_effect <= 0) or \
                       (effect_estimate < 0 and adjusted_effect >= 0)

            # Check if CI would include null
            ci_includes_null = abs(adjusted_effect) < abs(effect_estimate) * 0.3  # Rough approximation

            results.append({
                'RR_confound_outcome': float(rr_uy),
                'RR_confound_treatment': float(rr_ux),
                'bias': float(bias),
                'adjusted_effect': float(adjusted_effect),
                'effect_nullified': nullified,
                'ci_likely_includes_null': ci_includes_null
            })

    return pd.DataFrame(results)


def sensitivity_analysis_complete(
    result_object,
    baseline_risk: float = 0.3,
    prevalence_u: float = 0.2
) -> Dict:
    """
    Comprehensive sensitivity analysis using E-values and tipping point analysis.

    Parameters
    ----------
    result_object : AdjustmentResult
        Fitted adjustment result object
    baseline_risk : float
        Baseline risk for converting RD to RR
    prevalence_u : float
        Assumed prevalence of unmeasured confounder

    Returns
    -------
    Dict
        Complete sensitivity analysis results

    Examples
    --------
    >>> from population_adjustment import MAIC
    >>> maic = MAIC()
    >>> result = maic.fit(...)
    >>> sensitivity = sensitivity_analysis_complete(result, baseline_risk=0.35)
    >>> print(f"E-value: {sensitivity['e_value']['e_value_estimate']:.2f}")
    """
    # Calculate E-value
    e_value = calculate_e_value(
        effect_estimate=result_object.effect_estimate,
        ci_lower=result_object.ci_lower,
        effect_type='RD',
        baseline_risk=baseline_risk
    )

    # Tipping point analysis
    tipping = tipping_point_analysis(
        effect_estimate=result_object.effect_estimate,
        baseline_risk=baseline_risk,
        prevalence_u=prevalence_u
    )

    # Find minimum confounding strength to nullify
    nullified = tipping[tipping['effect_nullified']]

    if len(nullified) > 0:
        min_to_null = nullified[['RR_confound_outcome', 'RR_confound_treatment']].min()
        tipping_point_found = True
    else:
        min_to_null = None
        tipping_point_found = False

    # Summary
    summary = {
        'original_effect': float(result_object.effect_estimate),
        'original_ci': (float(result_object.ci_lower), float(result_object.ci_upper)),
        'e_value': e_value,
        'tipping_point': {
            'found': tipping_point_found,
            'min_RR_outcome': float(min_to_null['RR_confound_outcome']) if tipping_point_found else None,
            'min_RR_treatment': float(min_to_null['RR_confound_treatment']) if tipping_point_found else None
        },
        'tipping_point_data': tipping,
        'interpretation': _generate_interpretation(e_value, tipping_point_found, min_to_null)
    }

    return summary


def _generate_interpretation(e_value: Dict, tipping_found: bool, min_to_null) -> str:
    """Generate human-readable interpretation of sensitivity analysis."""

    interp = []

    interp.append(f"**E-value Analysis:**")
    interp.append(f"- E-value for point estimate: {e_value['e_value_estimate']:.2f} ({e_value['strength']})")

    if e_value['e_value_ci'] is not None:
        interp.append(f"- E-value for CI limit: {e_value['e_value_ci']:.2f}")

    interp.append(f"\n{e_value['description']}")

    if tipping_found:
        interp.append(f"\n**Tipping Point Analysis:**")
        interp.append(
            f"- Minimum confounding strength to nullify effect:\n"
            f"  - Confounder-Outcome RR: {min_to_null['RR_confound_outcome']:.2f}\n"
            f"  - Confounder-Treatment RR: {min_to_null['RR_confound_treatment']:.2f}"
        )
    else:
        interp.append(
            f"\n**Tipping Point:** Effect robust to confounding explored in tipping point range."
        )

    # Contextualization
    if e_value['e_value_estimate'] < 1.5:
        interp.append(
            f"\n**Interpretation:** The effect estimate is sensitive to unmeasured confounding. "
            f"Even modest confounders could explain the observed association."
        )
    elif e_value['e_value_estimate'] < 2.5:
        interp.append(
            f"\n**Interpretation:** Moderate unmeasured confounding would be needed to explain "
            f"away the effect. Consider whether plausible confounders of this strength exist."
        )
    else:
        interp.append(
            f"\n**Interpretation:** The effect estimate is robust to unmeasured confounding. "
            f"Very strong confounders would be needed to explain away the observed association."
        )

    return "\n".join(interp)
