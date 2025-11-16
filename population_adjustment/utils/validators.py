"""
Data validation utilities for population adjustment methods.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def validate_data_structure(
    data: pd.DataFrame,
    required_columns: List[str],
    allow_missing: bool = False
) -> None:
    """
    Validate data structure and completeness.

    Parameters
    ----------
    data : pd.DataFrame
        Data to validate
    required_columns : List[str]
        Required column names
    allow_missing : bool
        Whether to allow missing values

    Raises
    ------
    ValueError
        If validation fails
    """
    # Check if DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")

    # Check for required columns
    missing_cols = set(required_columns) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Data missing required columns: {missing_cols}")

    # Check for missing values
    if not allow_missing:
        if data[required_columns].isnull().any().any():
            null_cols = data[required_columns].columns[
                data[required_columns].isnull().any()
            ].tolist()
            raise ValueError(f"Data contains missing values in columns: {null_cols}")

    # Check for empty data
    if len(data) == 0:
        raise ValueError("Data is empty")


def check_binary_variable(
    data: pd.DataFrame,
    var_name: str,
    values: Optional[List] = None
) -> None:
    """
    Check if variable is properly coded as binary.

    Parameters
    ----------
    data : pd.DataFrame
        Data
    var_name : str
        Variable name
    values : Optional[List]
        Expected values (default: [0, 1])

    Raises
    ------
    ValueError
        If not properly coded
    """
    if values is None:
        values = [0, 1]

    unique_vals = data[var_name].dropna().unique()

    if not set(unique_vals).issubset(set(values)):
        raise ValueError(
            f"Variable '{var_name}' must only contain values {values}. "
            f"Found: {unique_vals}"
        )


def check_overlap(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    covariates: List[str],
    tolerance: float = 0.01
) -> None:
    """
    Check if there is sufficient overlap in covariate distributions.

    Parameters
    ----------
    data1 : pd.DataFrame
        First dataset
    data2 : pd.DataFrame
        Second dataset
    covariates : List[str]
        Covariates to check
    tolerance : float
        Tolerance for range overlap

    Raises
    ------
    Warning
        If limited overlap detected
    """
    import warnings

    for cov in covariates:
        x1 = data1[cov].values
        x2 = data2[cov].values

        min1, max1 = np.min(x1), np.max(x1)
        min2, max2 = np.min(x2), np.max(x2)

        # Check for overlap
        overlap_min = max(min1, min2)
        overlap_max = min(max1, max2)

        if overlap_max <= overlap_min:
            warnings.warn(
                f"No overlap in covariate '{cov}' between datasets. "
                f"Data1 range: [{min1:.2f}, {max1:.2f}], "
                f"Data2 range: [{min2:.2f}, {max2:.2f}]"
            )

        # Check proportion in common support
        range1 = max1 - min1
        range2 = max2 - min2

        if range1 > 0:
            overlap_prop1 = (overlap_max - overlap_min) / range1
            if overlap_prop1 < tolerance:
                warnings.warn(
                    f"Limited overlap in '{cov}': "
                    f"{overlap_prop1*100:.1f}% of data1 range overlaps"
                )

        if range2 > 0:
            overlap_prop2 = (overlap_max - overlap_min) / range2
            if overlap_prop2 < tolerance:
                warnings.warn(
                    f"Limited overlap in '{cov}': "
                    f"{overlap_prop2*100:.1f}% of data2 range overlaps"
                )
