# Population Adjustment Methods for Health Technology Assessment

A Python implementation of established population adjustment methods for target population meta-analysis, following NICE guidelines and published methodologies.

## Overview

This package implements three validated population adjustment methods for health technology assessment (HTA) and indirect treatment comparisons:

1. **Matching-Adjusted Indirect Comparison (MAIC)** - Entropy minimization with moment matching
2. **Simulated Treatment Comparison (STC)** - Outcome regression with G-computation
3. **Inverse Odds Weighting (IOW)** - Propensity score weighting with doubly-robust estimation

## Scope and Contributions

This implementation provides:

- **Correct Implementations**: Methods following published algorithms (Phillippo 2020, Signorovitch 2012, Remiro-Azócar 2020)
- **Proper Variance Estimation**: Bootstrap procedures accounting for two-stage estimation uncertainty
- **Doubly-Robust Estimators**: IOW with outcome regression augmentation (Bang & Robins 2005)
- **Comprehensive Diagnostics**: ESS, SMD, propensity overlap, weight distributions
- **Sensitivity Analysis**: E-values and tipping point analysis (VanderWeele & Ding 2017)
- **Validation**: Simulation studies, R package comparisons, correctness tests

## Installation

```bash
pip install -e .
```

Or install with development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from population_adjustment import MAIC, STC, IOW
import pandas as pd

# Load your data
# trial_data: IPD from your trial (with outcome and treatment)
# target_data: Target population characteristics (covariates only)
trial_data = pd.read_csv('trial_ipd.csv')
target_data = pd.read_csv('target_population.csv')

# Matching-Adjusted Indirect Comparison
maic = MAIC()
result = maic.fit(
    trial_data=trial_data,
    target_data=target_data,
    covariates=['age', 'sex', 'baseline_severity'],
    outcome='outcome',
    treatment='treatment',
    outcome_type='binary'
)

print(f"Treatment Effect (Target Population): {result.effect_estimate:.3f}")
print(f"95% CI: ({result.ci_lower:.3f}, {result.ci_upper:.3f})")
print(f"Effective Sample Size: {result.diagnostics['ess']:.1f}")

# Access diagnostics
print(f"SMD after adjustment: {result.diagnostics['smd_after']}")
```

## Methods

### 1. Matching-Adjusted Indirect Comparison (MAIC)

MAIC reweights IPD to match target population covariate distributions using entropy minimization.

**Implementation:**
- Moment matching via constrained optimization (Signorovitch 2012)
- Bootstrap variance estimation accounting for weight uncertainty
- Effective sample size and balance diagnostics
- Supports binary and continuous outcomes

### 2. Simulated Treatment Comparison (STC)

STC predicts counterfactual outcomes in the target population using outcome regression.

**Implementation:**
- Outcome regression with treatment-covariate interactions (Phillippo 2020)
- G-computation for marginal treatment effects
- Bootstrap resampling of IPD trial data with model refitting
- Supports binary and continuous outcomes

### 3. Inverse Odds Weighting (IOW)

IOW uses propensity scores for trial membership to weight IPD toward target population.

**Implementation:**
- Propensity score estimation via logistic regression
- Inverse odds weighting with stabilization
- Doubly-robust estimation (Bang & Robins 2005)
- Propensity overlap diagnostics

## Documentation

See the [docs/](docs/) directory for:
- [REFERENCES.md](docs/REFERENCES.md) - Comprehensive bibliography (67 references)
- [METHOD_SELECTION_GUIDE.md](docs/METHOD_SELECTION_GUIDE.md) - Decision trees and practical guidance
- API documentation in docstrings

## Validation and Examples

This package includes:

- **Simulation Studies** ([simulations/](simulations/)) - 6 scenarios × 100 replications showing bias, RMSE, coverage
- **R Validation** ([validation/](validation/)) - Comparison against R MAIC package
- **Case Study** ([examples/](examples/)) - NICE TA174 diabetes example with full workflow
- **Correctness Tests** ([tests/](tests/)) - Tests verifying methods recover known ground truth

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.21
- SciPy ≥ 1.7
- Pandas ≥ 1.3
- Statsmodels ≥ 0.13
- Scikit-learn ≥ 1.0
- Matplotlib ≥ 3.5 (for visualizations)
- Seaborn ≥ 0.11 (for visualizations)

## Current Status and Limitations

**Status:** Research implementation (Version 0.1.0)

This package provides correct implementations of established methods with proper validation. Current limitations:

- **Frequentist only**: Bayesian implementations are planned but not yet available
- **Limited outcome types**: Binary and continuous outcomes supported; time-to-event planned
- **Single trial**: Currently handles one IPD trial; multi-trial extensions planned
- **Validation scope**: R validation framework created but requires rpy2 setup

**Use Cases:**
- ✅ HTA submissions requiring population adjustment (NICE, EUnetHTA)
- ✅ Research comparing population adjustment methods
- ✅ Educational/teaching purposes
- ⚠ Production use: Recommend cross-validating against R packages

## Citation

If you use this package in your research, please cite:

```bibtex
@software{population_adjustment2025,
  title={Population Adjustment Methods for Health Technology Assessment},
  author={[Your Name]},
  year={2025},
  version={0.1.0},
  url={https://github.com/mahmood726-cyber/Idea7}
}
```

And cite the original methodological papers (see [REFERENCES.md](docs/REFERENCES.md)).

## References

1. Phillippo, D. M., Ades, A. E., Dias, S., Palmer, S., Abrams, K. R., & Welton, N. J. (2020). NICE DSU Technical Support Document 18: Methods for population-adjusted indirect comparisons in submission to NICE. National Institute for Health and Care Excellence.

2. Remiro-Azócar, A., Heath, A., & Baio, G. (2020). Methods for population adjustment with limited access to individual patient data: A review and simulation study. Research Synthesis Methods, 12(6), 750-775.

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].
