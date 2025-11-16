# Method Selection Guide
## When to Use MAIC, STC, or IOW?

This guide helps you choose the appropriate population adjustment method based on your data, research question, and assumptions.

---

## Quick Decision Tree

```
START: Do you have IPD from target population?
│
├─ NO (only aggregate statistics available)
│   └─ Use MAIC
│       ├─ Check overlap diagnostics
│       ├─ If ESS < 50: High risk, interpret cautiously
│       └─ If max_weight > 20: Extreme extrapolation, consider STC if possible
│
└─ YES (have target population IPD)
    │
    ├─ Is effect modification suspected?
    │   ├─ YES → Use STC with treatment×covariate interactions
    │   └─ NO → Continue to next question
    │
    ├─ Is overlap poor (SMD > 0.8)?
    │   ├─ YES → Prefer STC (more robust to poor overlap)
    │   └─ NO → Continue to next question
    │
    ├─ Want doubly-robust protection?
    │   ├─ YES → Use IOW with doubly_robust=True
    │   └─ NO → Any method appropriate
    │
    └─ Sample size small (n < 100)?
        ├─ YES → Prefer STC or IOW (MAIC unstable)
        └─ NO → Choose based on preferences

FINAL CHECK: Run diagnostics on chosen method!
```

---

## Detailed Method Comparison

### MAIC (Matching-Adjusted Indirect Comparison)

**When to use:**
- Target population data only available as aggregate statistics
- Want to balance covariates to match target moments
- Have sufficient sample size (n > 100) and good overlap

**Strengths:**
✅ Works with aggregate data only
✅ Directly balances covariate distributions
✅ Well-established in HTA (NICE accepts)
✅ Straightforward interpretation

**Limitations:**
❌ Requires good covariate overlap
❌ Can produce unstable weights (ESS << n)
❌ Doesn't handle effect modification explicitly
❌ Sensitive to extreme weights

**Key Diagnostics:**
- **ESS (Effective Sample Size):** Should be > 50
- **Max weight:** Should be < 20
- **Weight CV:** Should be < 2.0
- **SMD after adjustment:** Should be < 0.1

**When MAIC fails:**
- ESS < 30: Unstable, inference unreliable
- Max weight > 20: Extreme extrapolation
- Poor convergence: Optimization issues
- No overlap: Weights explode

---

### STC (Simulated Treatment Comparison)

**When to use:**
- Have IPD from target population
- Suspect treatment effect modification
- Poor covariate overlap between populations
- Want to model outcome process explicitly

**Strengths:**
✅ Handles effect modification naturally
✅ Robust to poor overlap
✅ Uses full target population (no ESS loss)
✅ Can model complex outcome relationships

**Limitations:**
❌ Requires target population IPD
❌ Sensitive to outcome model misspecification
❌ More complex to explain to stakeholders
❌ Bootstrap for variance is slow

**Key Diagnostics:**
- **Model fit:** Check residuals, calibration
- **Interaction significance:** Test treatment×covariate interactions
- **Prediction accuracy:** Validate on held-out data

**When STC fails:**
- Outcome model badly misspecified
- Sparse data in treatment arms
- Insufficient overlap in any covariate space
- Model won't converge

---

### IOW (Inverse Odds Weighting)

**When to use:**
- Have IPD from both trial and target
- Want doubly-robust protection
- Overlap is moderate to good
- Want to combine PS and outcome modeling

**Strengths:**
✅ Doubly-robust (consistent if either model correct)
✅ More efficient than pure IPW
✅ Combines weighting and outcome modeling
✅ Standard causal inference tool

**Limitations:**
❌ Requires target population IPD
❌ Positivity assumption (PS not near 0 or 1)
❌ Both models can still be wrong
❌ More complex than MAIC

**Key Diagnostics:**
- **PS overlap:** Check histograms, should overlap
- **ESS:** Should be reasonable (> 50)
- **Balance after weighting:** SMD < 0.1
- **Positivity:** No PS near 0 or 1

**When IOW fails:**
- No positivity (PS → 0 or 1)
- Both PS and outcome models wrong (for DR)
- Extreme weights
- Poor overlap

---

## Comparison Table

| Criterion | MAIC | STC | IOW |
|-----------|------|-----|-----|
| **Data Requirements** |
| Needs target IPD? | No | Yes | Yes |
| Needs trial IPD? | Yes | Yes | Yes |
| Works with aggregate only? | Yes | No | No |
| **Robustness** |
| Poor overlap | ⚠️ Fails | ✅ Robust | ⚠️ Moderate |
| Effect modification | ❌ No | ✅ Yes | ⚠️ Moderate |
| Model misspecification | N/A | ❌ Sensitive | ⚠️ DR protects |
| **Computational** |
| Speed | Fast | Moderate | Fast |
| Bootstrap time | ~1 min | ~5-10 min | ~2-3 min |
| Scales to large n? | Yes | Yes | Yes |
| **Interpretation** |
| Easy to explain? | ✅ Simple | ⚠️ Complex | ⚠️ Moderate |
| HTA acceptance | ✅ High | ⚠️ Growing | ⚠️ Moderate |
| **Diagnostics** |
| ESS reported? | Yes | No | Yes |
| Balance metrics? | Yes | Yes | Yes |
| Model diagnostics? | No | Yes | Yes |

**Legend:** ✅ Strong, ⚠️ Moderate, ❌ Weak/No

---

## Scenario-Based Recommendations

### Scenario 1: Standard HTA Submission

**Setup:**
- IPD from trial A (n=250)
- Only aggregate data from trial B
- Moderate population differences (SMD ≈ 0.5)

**Recommendation: MAIC**

**Rationale:**
- This is the classic MAIC use case
- NICE expects MAIC in this scenario
- Aggregate data precludes STC/IOW

**Implementation:**
```python
from population_adjustment import MAIC

maic = MAIC(method='frequentist', weighting_method='standard')
result = maic.fit(
    ipd_data=trial_a,
    aggregate_data=trial_b_stats,
    covariates=['age', 'sex', 'baseline_severity'],
    outcome='response',
    treatment='treatment'
)

# Check diagnostics
print(f"ESS: {result.effective_sample_size:.0f}")
print(f"Max weight: {result.diagnostics['max_weight']:.2f}")

if result.effective_sample_size < 50:
    print("WARNING: Low ESS, results may be unstable")
```

---

### Scenario 2: Effect Modification Suspected

**Setup:**
- IPD from trial (n=400)
- IPD from target registry (n=2000)
- Evidence that treatment effect varies by age

**Recommendation: STC with interactions**

**Rationale:**
- Need to model age×treatment interaction
- STC naturally handles this
- Large target sample provides power

**Implementation:**
```python
from population_adjustment import STC

stc = STC(
    method='frequentist',
    outcome_model='logistic',
    include_interaction=True,  # KEY: Models treatment×covariate
    bootstrap_samples=500
)

result = stc.fit(
    ipd_data=trial,
    aggregate_data=target_registry,
    covariates=['age', 'sex', 'severity'],
    outcome='response',
    treatment='treatment',
    target_is_ipd=True
)

# Check if interactions significant
# (would require access to model coefficients)
```

---

### Scenario 3: Poor Overlap

**Setup:**
- Trial enrolls young, healthy patients (mean age=55)
- Target is elderly, sicker (mean age=72)
- Large covariate differences (SMD > 1.0)

**Recommendation: STC (if have target IPD) or EXTREME CAUTION**

**Rationale:**
- MAIC will produce very low ESS (possibly < 20)
- Extreme weights = extreme extrapolation
- STC more robust but still risky
- Consider: Can we get more representative trial?

**Implementation:**
```python
# Try MAIC first, check ESS
maic_result = MAIC().fit(...)

if maic_result.effective_sample_size < 50:
    print("CAUTION: ESS too low for reliable inference")
    print(f"ESS = {maic_result.effective_sample_size:.0f}")
    print(f"Max weight = {maic_result.diagnostics['max_weight']:.1f}")
    print("\nRecommendation: Obtain target IPD and use STC")

    # If have target IPD
    stc_result = STC(include_interaction=True).fit(...)

    # Still check SMD
    # Large SMD means substantial extrapolation regardless of method
```

---

### Scenario 4: Want Maximum Robustness

**Setup:**
- IPD from trial and target
- Unsure about modeling assumptions
- Want protection against misspecification

**Recommendation: IOW with doubly_robust=True**

**Rationale:**
- DR gives protection if one model correct
- Most robust option when both data sources available

**Implementation:**
```python
from population_adjustment import InverseOddsWeighting

# Combine data
trial_data['trial'] = 1
target_data['trial'] = 0
combined = pd.concat([trial_data, target_data])

iow = InverseOddsWeighting(
    method='frequentist',
    propensity_model='logistic',
    doubly_robust=True,  # KEY: DR protection
    stabilize_weights=True
)

result = iow.fit(
    combined_data=combined,
    trial_indicator='trial',
    covariates=['age', 'sex', 'severity'],
    outcome='response',
    treatment='treatment'
)

# Check both PS overlap AND balance
print(f"PS overlap: {result.diagnostics['propensity_score_overlap']}")
print(f"ESS: {result.effective_sample_size:.0f}")
```

---

## Diagnostic Thresholds

### Critical Thresholds (Analysis Unreliable)

| Diagnostic | Threshold | Interpretation |
|------------|-----------|----------------|
| ESS | < 30 | STOP: Too unstable |
| Max weight | > 50 | STOP: Extreme extrapolation |
| Weight CV | > 5 | STOP: Highly variable |
| SMD after adjustment | > 0.5 | STOP: Poor balance |
| PS min/max | < 0.01 or > 0.99 | STOP: Positivity violated |

### Warning Thresholds (Interpret Cautiously)

| Diagnostic | Threshold | Interpretation |
|------------|-----------|----------------|
| ESS | 30-50 | WARNING: Low power |
| Max weight | 20-50 | WARNING: Large extrapolation |
| Weight CV | 2-5 | WARNING: High variability |
| SMD after adjustment | 0.1-0.5 | WARNING: Residual imbalance |
| PS min/max | 0.01-0.05 or 0.95-0.99 | WARNING: Limited overlap |

### Acceptable Thresholds

| Diagnostic | Threshold | Interpretation |
|------------|-----------|----------------|
| ESS | > 50 | OK |
| Max weight | < 20 | OK |
| Weight CV | < 2 | OK |
| SMD after adjustment | < 0.1 | OK: Good balance |
| PS | 0.05-0.95 | OK: Good overlap |

---

## Common Pitfalls

### Pitfall 1: Ignoring Diagnostics

**Problem:** Running method and reporting result without checking diagnostics.

**Solution:**
```python
result = method.fit(...)

# ALWAYS check diagnostics
print(result.diagnostics)

if 'effective_sample_size' in result.diagnostics:
    if result.diagnostics['effective_sample_size'] < 50:
        raise ValueError("ESS too low - results unreliable")

# Plot balance
result.plot_balance()
plt.savefig('balance_check.png')
```

### Pitfall 2: Using MAIC with Poor Overlap

**Problem:** Applying MAIC when populations very different → ESS = 15.

**Solution:**
- Check SMD before running MAIC
- If SMD > 0.8, expect problems
- Consider STC if target IPD available
- Or acknowledge limitations honestly

### Pitfall 3: Ignoring Effect Modification

**Problem:** Using MAIC/IOW when treatment effect varies by age, gender, etc.

**Solution:**
- Test for interactions in trial data
- If significant, use STC with `include_interaction=True`
- Report subgroup results

### Pitfall 4: Over-Interpreting Point Estimates

**Problem:** Focusing only on effect estimate, ignoring uncertainty.

**Solution:**
```python
result = method.fit(...)

print(f"Effect: {result.effect_estimate:.3f}")
print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")

# Check if CI includes null
if result.ci_lower < 0 < result.ci_upper:
    print("CI includes null - effect not significant")

# Run sensitivity analysis
from population_adjustment.utils.sensitivity import sensitivity_analysis_complete
sens = sensitivity_analysis_complete(result, baseline_risk=0.3)
print(f"E-value: {sens['e_value']['e_value_estimate']:.2f}")
```

### Pitfall 5: Not Checking Model Fit (STC)

**Problem:** Using STC without validating outcome model.

**Solution:**
```python
stc = STC().fit(...)

# Check model (requires access to fitted model)
# - Residual plots
# - Calibration
# - Cross-validation

# At minimum: test interactions
# If interactions significant, MUST include them
```

---

## Reporting Checklist

When reporting population adjustment results, include:

**Methods Section:**
- [ ] Which method used (MAIC/STC/IOW) and why
- [ ] Software and version
- [ ] Covariates adjusted for (and justification)
- [ ] Weighting/modeling approach details

**Results Section:**
- [ ] Sample sizes (trial and target)
- [ ] Baseline covariate distributions
- [ ] SMD before and after adjustment
- [ ] For MAIC/IOW: ESS, max weight, weight distribution
- [ ] Adjusted effect estimate with 95% CI
- [ ] Comparison to naive (unadjusted) estimate

**Diagnostics (in supplement):**
- [ ] Balance metrics table
- [ ] Love plot
- [ ] Weight distribution plot (for MAIC/IOW)
- [ ] PS overlap plot (for IOW)
- [ ] Sensitivity analysis (E-values or tipping point)

**Interpretation:**
- [ ] Acknowledge assumptions
- [ ] Discuss limitations (overlap, model specification)
- [ ] Compare to unadjusted analysis
- [ ] Sensitivity to unmeasured confounding

---

## Example Analysis Workflow

### Complete Workflow with Diagnostics

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from population_adjustment import MAIC
from population_adjustment.utils.sensitivity import sensitivity_analysis_complete

# 1. Load and examine data
trial = pd.read_csv('trial_ipd.csv')
target_stats = pd.read_csv('target_aggregate.csv')

print(f"Trial n = {len(trial)}")
print(f"\nTrial demographics:")
print(trial[['age', 'sex', 'severity']].describe())

# 2. Check baseline imbalance
from population_adjustment.utils.diagnostics import calculate_balance_metrics

# Need to simulate target for comparison
# (or compute SMD from aggregate stats)
target_sim = pd.DataFrame({
    'age': np.random.normal(target_stats['age'][0], 10, 1000),
    'sex': np.random.binomial(1, target_stats['sex'][0], 1000),
    'severity': np.random.normal(target_stats['severity'][0], 15, 1000)
})

balance_before = calculate_balance_metrics(
    data1=trial,
    data2=target_sim,
    covariates=['age', 'sex', 'severity']
)

print("\nBalance before adjustment:")
print(balance_before[['variable', 'smd']])

if balance_before['smd'].abs().max() > 0.8:
    print("\nWARNING: Large imbalance detected!")

# 3. Fit MAIC
maic = MAIC(method='frequentist', random_state=42)
result = maic.fit(
    ipd_data=trial,
    aggregate_data=target_stats,
    covariates=['age', 'sex', 'severity'],
    outcome='response',
    treatment='treatment',
    outcome_type='binary'
)

# 4. Check diagnostics
print(f"\n{'='*50}")
print("DIAGNOSTICS")
print(f"{'='*50}")
print(f"ESS: {result.effective_sample_size:.0f} (from n={len(trial)})")
print(f"Max weight: {result.diagnostics['max_weight']:.2f}")
print(f"Weight CV: {result.diagnostics['weight_cv']:.2f}")

if result.effective_sample_size < 50:
    print("\n❌ WARNING: ESS < 50 - Results may be unstable!")
elif result.effective_sample_size < 100:
    print("\n⚠️  CAUTION: ESS < 100 - Interpret carefully")
else:
    print("\n✅ ESS acceptable")

# 5. Plot balance
fig = result.plot_balance(covariates=['age', 'sex', 'severity'])
plt.savefig('balance_plot.png', dpi=300, bbox_inches='tight')
print("Balance plot saved: balance_plot.png")

# 6. Plot weights
fig = result.plot_weights()
plt.savefig('weight_distribution.png', dpi=300, bbox_inches='tight')
print("Weight plot saved: weight_distribution.png")

# 7. Results
print(f"\n{'='*50}")
print("RESULTS")
print(f"{'='*50}")
print(result.summary())

# 8. Sensitivity analysis
sensitivity = sensitivity_analysis_complete(result, baseline_risk=0.3)
print(f"\nE-value: {sensitivity['e_value']['e_value_estimate']:.2f}")
print(f"Interpretation: {sensitivity['e_value']['interpretation']}")

# 9. Compare to naive
naive_effect = (trial[trial['treatment']==1]['response'].mean() -
                trial[trial['treatment']==0]['response'].mean())
print(f"\nNaive estimate: {naive_effect:.3f}")
print(f"MAIC estimate: {result.effect_estimate:.3f}")
print(f"Difference: {abs(naive_effect - result.effect_estimate):.3f}")

print("\n✅ Analysis complete!")
```

---

## Decision Support Tool

### Interactive Decision Helper

Answer these questions:

1. **Do you have IPD from target population?**
   - YES → Go to Q2
   - NO → **Use MAIC**

2. **Is sample size small (n < 100)?**
   - YES → Avoid MAIC → **Use STC or IOW**
   - NO → Go to Q3

3. **Is there evidence of effect modification?**
   - YES → **Use STC with interactions**
   - NO → Go to Q4

4. **Is overlap poor (SMD > 0.8)?**
   - YES → **Use STC** (most robust)
   - NO → Go to Q5

5. **Want doubly-robust protection?**
   - YES → **Use IOW-DR**
   - NO → **Any method OK** (choose by preference)

---

## Summary Recommendations

**Default choice (aggregate data only):** MAIC

**Have target IPD + Effect modification:** STC

**Have target IPD + Want robustness:** IOW-DR

**Poor overlap:** STC (if target IPD) or extreme caution

**Small sample:** STC or IOW (not MAIC)

**Always:** Check diagnostics before interpreting results!

---

## Further Reading

- NICE DSU TSD 18 (Phillippo et al. 2020) - Comprehensive guidance
- Remiro-Azócar et al. (2022) - Method comparison study
- Signorovitch et al. (2012) - Original MAIC paper
- Bang & Robins (2005) - Doubly-robust methods

---

**For questions or to report issues with this guide:**
GitHub: https://github.com/mahmood726-cyber/Idea7/issues
