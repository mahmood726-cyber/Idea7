# Validation Against R Packages

This directory contains validation scripts that compare our Python implementations against established R packages to ensure numerical correctness.

## Overview

Population adjustment methods are well-established in the R ecosystem. To validate our Python implementations, we compare results against:

1. **R MAIC Package** (Phillippo et al.) - For MAIC validation
2. **R MatchIt Package** - For propensity score estimation
3. **R boot Package** - For bootstrap variance estimation

## Requirements

```bash
# Python requirements
pip install rpy2

# R requirements (run in R console)
install.packages(c("boot", "MatchIt"))
devtools::install_github("remiroazocar/MAIC")
```

## Running Validation

```bash
python validation/validate_against_R.py
```

## Validation Tests

### 1. MAIC Validation

Compares:
- Effect estimates (should agree within 0.001)
- Effective Sample Size (should agree within 1.0)
- Weight diagnostics (max weight, weight distribution)

**Expected Result:** Python and R produce identical results when using same optimization algorithm and tolerance.

### 2. Propensity Score Validation

Compares:
- Mean propensity scores
- Standard deviation of propensity scores
- Odds ratios

**Expected Result:** Logistic regression coefficients and predictions agree within numerical tolerance.

### 3. Bootstrap Validation

Compares:
- Bootstrap standard errors using identical seed
- Bootstrap distributions

**Expected Result:** Bootstrap procedures produce equivalent variance estimates.

## Validation Results

Results are saved to `validation_results.json` with the following structure:

```json
{
  "maic": {
    "status": "completed",
    "effect_estimate": {
      "python": 0.152341,
      "R": 0.152339,
      "difference": 0.000002,
      "agree": true
    },
    "ess": {
      "python": 487.32,
      "R": 487.31,
      "difference": 0.01,
      "agree": true
    }
  }
}
```

## Troubleshooting

### rpy2 Installation Issues

On Linux:
```bash
sudo apt-get install r-base-dev
pip install rpy2
```

On macOS:
```bash
brew install r
pip install rpy2
```

### R Package Installation Issues

If GitHub installation fails for MAIC:
```r
# Install from source
install.packages("devtools")
devtools::install_github("remiroazocar/MAIC", force = TRUE)
```

### Numerical Differences

Small numerical differences (< 0.001) are acceptable due to:
- Different optimization convergence criteria
- Different random number generators
- Floating point arithmetic differences

## References

- **R MAIC Package**: Remiro-Azócar et al. (2022) "MAIC: R package for matching-adjusted indirect comparison"
- **MatchIt**: Ho et al. (2011) "MatchIt: Nonparametric preprocessing for parametric causal inference"
- **boot**: Canty & Ripley (2021) "boot: Bootstrap R (S-Plus) Functions"
