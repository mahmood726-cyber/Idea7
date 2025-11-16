# Population Adjustment Methods

## Overview

This document provides detailed mathematical descriptions of the three main population adjustment methods implemented in this package.

## 1. Matching-Adjusted Indirect Comparison (MAIC)

### Method Description

MAIC reweights individual patient data (IPD) from one trial to match the population characteristics of another trial or target population. It uses propensity score weighting based on the method of moments.

### Mathematical Formulation

Let $X_i$ denote the vector of baseline covariates for individual $i$ in the IPD trial, and let $\bar{X}_{target}$ denote the mean covariate values in the target population.

MAIC estimates weights $w_i$ that minimize entropy subject to moment-matching constraints:

$$
\min_{w_i} \sum_{i=1}^{n} w_i \log(w_i)
$$

subject to:

$$
\sum_{i=1}^{n} w_i X_i = n \bar{X}_{target}
$$

$$
\sum_{i=1}^{n} w_i = n
$$

### Implementation

The dual problem is solved using constrained optimization:

$$
w_i = \frac{\exp(-\beta^T X_i)}{\sum_{j=1}^{n} \exp(-\beta^T X_j)}
$$

where $\beta$ is chosen to satisfy the moment constraints.

### Adjusted Treatment Effect

After obtaining weights, the population-adjusted treatment effect is estimated as:

$$
\hat{\Delta} = \frac{\sum_{i:T_i=1} w_i Y_i}{\sum_{i:T_i=1} w_i} - \frac{\sum_{i:T_i=0} w_i Y_i}{\sum_{i:T_i=0} w_i}
$$

where $T_i$ is the treatment indicator and $Y_i$ is the outcome.

### Variance Estimation

The variance of $\hat{\Delta}$ accounts for weighting:

$$
\text{Var}(\hat{\Delta}) = \frac{\text{Var}_1}{ESS_1} + \frac{\text{Var}_0}{ESS_0}
$$

where $ESS_j = \frac{(\sum_{i:T_i=j} w_i)^2}{\sum_{i:T_i=j} w_i^2}$ is the effective sample size.

## 2. Simulated Treatment Comparison (STC)

### Method Description

STC uses outcome regression to model the relationship between covariates, treatment, and outcome in the IPD trial, then predicts counterfactual outcomes in the target population.

### Mathematical Formulation

Step 1: Fit outcome model in IPD:

$$
E[Y|X, T] = g(\alpha_0 + \alpha_X^T X + \alpha_T T + \alpha_{TX}^T (X \times T))
$$

where $g()$ is the link function (identity for continuous outcomes, logit for binary outcomes).

Step 2: Predict in target population using G-computation:

For each individual $j$ in the target population with covariates $X_j$:

$$
\hat{Y}_j(T=1) = g(\hat{\alpha}_0 + \hat{\alpha}_X^T X_j + \hat{\alpha}_T + \hat{\alpha}_{TX}^T X_j)
$$

$$
\hat{Y}_j(T=0) = g(\hat{\alpha}_0 + \hat{\alpha}_X^T X_j)
$$

Step 3: Marginal treatment effect:

$$
\hat{\Delta}_{STC} = \frac{1}{N_{target}} \sum_{j=1}^{N_{target}} [\hat{Y}_j(T=1) - \hat{Y}_j(T=0)]
$$

### Variance Estimation

Bootstrap resampling of the target population is used to estimate variance:

$$
\text{SE}(\hat{\Delta}_{STC}) = \sqrt{\frac{1}{B-1} \sum_{b=1}^{B} (\hat{\Delta}_{STC}^{(b)} - \bar{\Delta}_{STC})^2}
$$

## 3. Inverse Odds Weighting (IOW)

### Method Description

IOW combines propensity scores for trial membership with inverse odds weighting to adjust for population differences. It can be doubly robust when combined with outcome modeling.

### Mathematical Formulation

Step 1: Estimate propensity scores for trial membership:

$$
e(X_i) = P(S_i = 1 | X_i)
$$

where $S_i = 1$ indicates trial membership.

Step 2: Calculate inverse odds weights:

$$
w_i = \frac{1 - e(X_i)}{e(X_i)} = \frac{P(S_i=0|X_i)}{P(S_i=1|X_i)}
$$

These weights represent the odds of being in the target population relative to the trial.

Step 3: Estimate weighted treatment effect:

$$
\hat{\Delta}_{IOW} = \frac{\sum_{i:T_i=1} w_i Y_i}{\sum_{i:T_i=1} w_i} - \frac{\sum_{i:T_i=0} w_i Y_i}{\sum_{i:T_i=0} w_i}
$$

### Doubly-Robust Estimation

For improved robustness, IOW can be combined with outcome modeling:

$$
\hat{\Delta}_{DR} = \frac{1}{n} \sum_{i=1}^{n} \left[\hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) + w_i T_i (Y_i - \hat{\mu}_1(X_i)) - w_i(1-T_i)(Y_i - \hat{\mu}_0(X_i))\right]
$$

where $\hat{\mu}_t(X)$ are fitted outcome models for treatment $t$.

This estimator is consistent if either the propensity score model or the outcome model is correctly specified.

## Novel Bayesian Extensions

### Bayesian MAIC

We extend MAIC with Bayesian inference by placing priors on $\beta$:

$$
\beta \sim N(0, \sigma_\beta^2 I)
$$

and using MCMC to sample from the posterior distribution, propagating uncertainty through to the treatment effect estimate.

### Bayesian STC

The outcome model is specified as a Bayesian hierarchical model:

$$
Y_i | X_i, T_i \sim \text{Bernoulli}(\text{logit}^{-1}(\alpha_0 + \alpha_X^T X_i + \alpha_T T_i))
$$

$$
\alpha \sim N(0, \sigma_\alpha^2)
$$

Posterior predictive distributions provide credible intervals that account for both parameter and predictive uncertainty.

## Diagnostics

### Effective Sample Size

For weighted methods (MAIC, IOW):

$$
ESS = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}
$$

### Standardized Mean Difference

Balance is assessed using SMD:

$$
SMD = \frac{\bar{X}_{trial} - \bar{X}_{target}}{\sqrt{\frac{s_{trial}^2 + s_{target}^2}{2}}}
$$

Balance is considered adequate if $|SMD| < 0.1$.

## References

1. Signorovitch et al. (2012). Comparative effectiveness without head-to-head trials. Pharmacoeconomics, 30(5), 351-363.

2. Phillippo et al. (2020). NICE DSU Technical Support Document 18: Methods for population-adjusted indirect comparisons.

3. Remiro-Azócar et al. (2020). Methods for population adjustment with limited access to individual patient data. Research Synthesis Methods, 12(6), 750-775.

4. Caro & Ishak (2010). No head-to-head trial? Simulate the missing arms. Pharmacoeconomics, 28(10), 957-967.
