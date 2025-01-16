from typing import List
import pandas as pd
import numpy as np

def _format_table(df: pd.DataFrame, format: str) -> str:
    if format == "markdown":
        return df.to_markdown(index=False)
    elif format == "html":
        return df.to_html(index=False)
    elif format == "latex":
        return df.to_latex(index=False)
    elif format == "csv":
        return df.to_csv(index=False)
    elif format == "pandas":
        return df.to_string(index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")


def raw_table(df: pd.DataFrame, format: str = "csv", variables: List[str] = None, bins: int = 5, group_by: str = None) -> str:
    """
    Convert a pandas DataFrame to a string representation in a given format.
    """
    df = df.copy()
    if variables is not None:
        df = df[variables]

    if group_by is not None:
        df[group_by] = pd.cut(df[group_by], bins=bins)

    return _format_table(df, format)


def frequency_table(df: pd.DataFrame, format: str = "csv", variables: List[str] = None, bins: int = None, group_by: str = None) -> str:
    """
    Convert a pandas DataFrame to a frequency table.
    """
    df = df.copy()
    if variables is not None:
        if group_by is not None:
            df = df[variables + [group_by]]
            try:
                group_by_values = sorted([t.item() for t in df[group_by].unique()])
            except:
                group_by_values = sorted([t for t in df[group_by].unique()])
        else:
            df = df[variables]

    if bins is not None:
        for col in variables:
            lower, higher = df[col].min(), df[col].max()
            edges = np.linspace(lower, higher, bins+1)
            labels = ['(%.2f, %.2f]'%(edges[i], edges[i+1]) for i in range(len(edges)-1)]
            df[col] = pd.cut(df[col], bins=edges, labels=labels, precision=1)
            df = df[df[col].notna()]
    
    if group_by is not None:
        # Create a cross-tabulation of frequencies
        df = pd.crosstab(df[group_by], [df[col] for col in sorted(df.columns) if col != group_by], dropna=False).T
        df = df.reset_index()
        df.columns = variables + [f"{group_by}={group_by_value}" for group_by_value in group_by_values]
    else:
        # Original behavior
        df = df.value_counts(dropna=False).to_frame().reset_index()
    
    # Sort by values
    df = df.sort_values(by=variables, ascending=True)

    return _format_table(df, format)


def mean_table(df: pd.DataFrame, variables: List[str] = None, values: List[str] = None, columns: List[str] = None, format: str = "csv") -> str:
    """
    Convert a pandas DataFrame to a mean table.
    """
    df = df.copy()

    if columns is None:
        df = df[variables + values]
        df = df.groupby(variables)[values].mean().reset_index()
        df.columns = variables + [f"{v} (mean)" for v in values]
    else:
        df = df[variables + values + columns]
        df = pd.pivot_table(data=df, index=variables, columns=columns, values=values, aggfunc="mean")

    return _format_table(df, format)


def correlation_table(df: pd.DataFrame, format: str = "csv", variables: List[str] = None) -> str:
    """
    Convert a pandas DataFrame to a correlation table.
    """
    if variables is not None:
        df = df[variables]
    df = df.corr()

    return _format_table(df, format)


def confidence_table(df: pd.DataFrame, variables: List[str], values: List[str], group_by: List[str] = None, format: str = "csv", bins: int = 10) -> str:
    """
    Create a table showing means and confidence intervals for values grouped by variables and optionally group_by
    """
    # Create grouping columns
    grouping_cols = []
    for var in variables:
        if df.is_categorical(var):
            grouping_cols.append(var)
        else:
            df[f"{var} (intervals)"] = pd.cut(df[var], bins=bins)
            grouping_cols.append(f"{var} (intervals)")
            
    if group_by is not None:
        grouping_cols.extend(group_by)

    # Calculate statistics
    stats = df.groupby(grouping_cols, observed=True)[values].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count')
    ])

    # Calculate confidence intervals
    stats_dict = {}
    for value in values:
        value_stats = stats[value]
        ci = 1.96 * value_stats['std'] / np.sqrt(value_stats['count'])
        stats_dict[f"{value} (mean)"] = value_stats['mean']
        stats_dict[f"{value} (ci)"] = ci

    result = pd.DataFrame(stats_dict).round(2).reset_index()

    if group_by is not None:
        result = result.sort_values(group_by)
    return _format_table(result, format)
