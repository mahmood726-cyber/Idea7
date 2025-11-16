"""
Data loading and exporting utilities.
"""

from typing import Dict, List, Optional
import pandas as pd
import json


def load_ipd(
    file_path: str,
    file_type: str = 'csv',
    **kwargs
) -> pd.DataFrame:
    """
    Load individual patient data from file.

    Parameters
    ----------
    file_path : str
        Path to data file
    file_type : str
        Type of file: 'csv', 'excel', 'stata', 'sas'
    **kwargs
        Additional arguments passed to pandas reader

    Returns
    -------
    pd.DataFrame
        Loaded data
    """
    if file_type == 'csv':
        return pd.read_csv(file_path, **kwargs)
    elif file_type == 'excel':
        return pd.read_excel(file_path, **kwargs)
    elif file_type == 'stata':
        return pd.read_stata(file_path, **kwargs)
    elif file_type == 'sas':
        return pd.read_sas(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def load_aggregate_data(
    file_path: str,
    file_type: str = 'csv',
    **kwargs
) -> pd.DataFrame:
    """
    Load aggregate statistics from file.

    Parameters
    ----------
    file_path : str
        Path to data file
    file_type : str
        Type of file: 'csv', 'excel', 'json'
    **kwargs
        Additional arguments

    Returns
    -------
    pd.DataFrame
        Aggregate statistics
    """
    if file_type == 'json':
        with open(file_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
    else:
        return load_ipd(file_path, file_type, **kwargs)


def export_results(
    result,
    output_path: str,
    format: str = 'csv',
    include_diagnostics: bool = True
) -> None:
    """
    Export adjustment results to file.

    Parameters
    ----------
    result : AdjustmentResult
        Result object to export
    output_path : str
        Path for output file
    format : str
        Output format: 'csv', 'json', 'excel'
    include_diagnostics : bool
        Whether to include diagnostic information
    """
    # Create summary dataframe
    summary_data = {
        'method': result.method_name,
        'effect_estimate': result.effect_estimate,
        'standard_error': result.standard_error,
        'ci_lower': result.ci_lower,
        'ci_upper': result.ci_upper,
    }

    if include_diagnostics:
        # Flatten diagnostics
        for key, value in result.diagnostics.items():
            if not isinstance(value, dict):
                summary_data[f'diagnostic_{key}'] = value

    df = pd.DataFrame([summary_data])

    # Export
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'json':
        with open(output_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")

    # Also export weights if available
    if result.weights is not None and result.ipd_data is not None:
        weights_path = output_path.replace(f'.{format}', f'_weights.{format}')
        weights_df = result.ipd_data.copy()
        weights_df['adjustment_weight'] = result.weights

        if format == 'csv':
            weights_df.to_csv(weights_path, index=False)
        elif format == 'excel':
            weights_df.to_excel(weights_path, index=False)
