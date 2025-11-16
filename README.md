# Target Population Meta-Analysis: Novel Methods for Population Adjustment

A comprehensive Python package implementing state-of-the-art population adjustment methods for target population meta-analysis, with novel Bayesian extensions.

## Overview

This package implements three major population adjustment methods for health technology assessment (HTA) and meta-analysis:

1. **Inverse Odds Weighting (IOW)** - Population adjustment using propensity score weighting
2. **Matching-Adjusted Indirect Comparison (MAIC)** - Weighting individual patient data to match aggregate data
3. **Simulated Treatment Comparison (STC)** - Outcome regression-based population adjustment

## Novel Contributions

This implementation extends existing methods (Phillippo 2020, Remiro-Azócar 2020) with:

- **Unified Bayesian Framework**: Full Bayesian implementations of all three methods with uncertainty propagation
- **Doubly-Robust Estimators**: Combined propensity score and outcome regression approaches
- **Adaptive Weighting**: Novel variance-minimizing weight calibration
- **Comprehensive Diagnostics**: Balance metrics, effective sample size, sensitivity analyses
- **Bootstrap and Bayesian Uncertainty**: Multiple approaches to confidence/credible intervals

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
from population_adjustment import MAIC, STC, InverseOddsWeighting
import pandas as pd

# Load your IPD (Individual Patient Data) and aggregate data
ipd = pd.read_csv('trial_ipd.csv')
aggregate_data = pd.read_csv('target_population.csv')

# Matching-Adjusted Indirect Comparison
maic = MAIC(method='bayesian')
result = maic.fit(
    ipd_data=ipd,
    aggregate_data=aggregate_data,
    covariates=['age', 'sex', 'baseline_severity'],
    outcome='response',
    treatment='treatment_arm'
)

print(f"Adjusted Treatment Effect: {result.effect_estimate:.3f}")
print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")

# Visualize balance
result.plot_balance()
```

## Methods

### 1. Matching-Adjusted Indirect Comparison (MAIC)

MAIC adjusts IPD from one trial to match the population characteristics of another trial or target population using propensity score weighting.

**Features:**
- Standard MAIC (moment matching)
- Bayesian MAIC with prior regularization
- Entropy balancing for stable weights
- Effective sample size diagnostics

### 2. Simulated Treatment Comparison (STC)

STC uses outcome regression to predict counterfactual outcomes in the target population.

**Features:**
- Frequentist regression-based STC
- Bayesian hierarchical outcome models
- Multiple outcome types (binary, continuous, time-to-event)
- G-computation for marginal effects

### 3. Inverse Odds Weighting (IOW)

IOW combines propensity scores with inverse odds of trial participation for population adjustment.

**Features:**
- Propensity score estimation (logistic, boosting, random forest)
- Stabilized weights for variance reduction
- Doubly-robust estimation
- Cross-validation for model selection

## Documentation

See the [docs/](docs/) directory for:
- [Methods Overview](docs/methods.md) - Detailed mathematical descriptions
- [API Reference](docs/api.md) - Complete API documentation
- [Examples](docs/examples.md) - Usage examples and case studies
- [User Guide](docs/guide.md) - Step-by-step tutorials

## Examples

See the [examples/](examples/) and [notebooks/](notebooks/) directories for comprehensive examples:

- Basic population adjustment workflows
- Indirect treatment comparisons
- Sensitivity analyses
- Simulation studies replicating published results

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.21
- SciPy ≥ 1.7
- Pandas ≥ 1.3
- Statsmodels ≥ 0.13
- Scikit-learn ≥ 1.0
- PyMC ≥ 5.0 (for Bayesian methods)

## Citation

If you use this package in your research, please cite:

```bibtex
@software{population_adjustment2025,
  title={Target Population Meta-Analysis: Novel Methods for Population Adjustment},
  author={[Your Name]},
  year={2025},
  url={https://github.com/mahmood726-cyber/Idea7}
}
```

## References

1. Phillippo, D. M., Ades, A. E., Dias, S., Palmer, S., Abrams, K. R., & Welton, N. J. (2020). NICE DSU Technical Support Document 18: Methods for population-adjusted indirect comparisons in submission to NICE. National Institute for Health and Care Excellence.

2. Remiro-Azócar, A., Heath, A., & Baio, G. (2020). Methods for population adjustment with limited access to individual patient data: A review and simulation study. Research Synthesis Methods, 12(6), 750-775.

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].
