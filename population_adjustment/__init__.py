"""
Population Adjustment Methods for Target Population Meta-Analysis

This package implements state-of-the-art methods for adjusting treatment effects
to target populations, including:
- Matching-Adjusted Indirect Comparison (MAIC)
- Simulated Treatment Comparison (STC)
- Inverse Odds Weighting (IOW)

With novel Bayesian extensions and comprehensive diagnostics.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from population_adjustment.core.maic import MAIC
from population_adjustment.core.stc import STC
from population_adjustment.core.iow import InverseOddsWeighting
from population_adjustment.core.base import AdjustmentResult

__all__ = [
    "MAIC",
    "STC",
    "InverseOddsWeighting",
    "AdjustmentResult",
]
