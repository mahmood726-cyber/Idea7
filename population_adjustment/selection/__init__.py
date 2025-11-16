"""
Automated Method Selection Module

This module provides novel algorithms for automatically selecting the most
appropriate population adjustment method based on data characteristics.
"""

from .automated_selection import (
    AutomatedMethodSelector,
    SelectionRecommendation,
    quick_select
)

__all__ = [
    'AutomatedMethodSelector',
    'SelectionRecommendation',
    'quick_select'
]
