import warnings
import re

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Set default plot settings
plt.rc("axes", labelweight="bold", labelsize=16)
#color_palette = sns.color_palette("husl", 10)
color_palette = sns.color_palette("Set2", 10)
sns.set_palette(color_palette)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    '''
    if isinstance(text, pd.Series):
        return text.apply(lambda x: [atoi(c) for c in re.split(r'(\d+)', str(x))])
    return [atoi(c) for c in re.split(r'(\d+)', str(text))]


def scatter_plot(data, x, y, hue=None, style=None, legend=True, **kwargs):
    """
    Create a scatter plot. Takes one (continuous) x & one (continuous) y. 
    Optionally takes one (continuous/categorical) hue and/or one (categorical) style.
    
    Parameters:
        data (pd.DataFrame): Input dataframe
        x (str): Column name for x-axis
        y (str): Column name for y-axis
        hue (str, optional): Column name for color mapping
        style (str, optional): Column name for style mapping
        legend (bool, optional): Whether to show legend. Defaults to True.
        **kwargs: Additional keyword arguments to pass to sns.scatterplot
        
    Returns:
        matplotlib.axes.Axes: The plot object
    """
    assert isinstance(y, str), "y must be a single column name"

    if hue is None:
        color = color_palette[np.random.choice(range(len(color_palette)))]
        ax = sns.scatterplot(data=data, x=x, y=y, style=style, color=color, alpha=0.75, **kwargs)
    else:
        # Check if hue is numeric and has many unique values
        is_numeric_hue = pd.api.types.is_numeric_dtype(data[hue])
        n_unique = len(data[hue].unique())
        
        if is_numeric_hue: #and n_unique > 5:
            # Use continuous colormap instead of discrete palette
            scatter = plt.scatter(data[x], data[y], c=data[hue], alpha=0.75, **kwargs)
            ax = plt.gca()
            ax.set_xlabel(x)

            if legend:
                cbar = plt.colorbar(scatter)
                cbar.set_label(hue, rotation=0)
        else:
            ax = sns.scatterplot(data=data, x=x, y=y, hue=hue, style=style, alpha=0.75, **kwargs)
    
    ax.set_ylabel(y, rotation=0)
    
    if not legend:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    if ax.get_legend() is not None:
        # Get the handles and labels
        handles, labels = ax.get_legend_handles_labels()
        
        # Split handles and labels into color and style groups
        if hue is not None and style is not None:
            unique_hue_values = data[hue].unique()
            unique_style_values = data[style].unique()
            
            hue_handles = []
            hue_labels = []
            style_handles = []
            style_labels = []
            
            for h, l in zip(handles, labels):
                if l in unique_hue_values:
                    hue_handles.append(h)
                    hue_labels.append(l)
                elif l in unique_style_values:
                    style_handles.append(h)
                    style_labels.append(l)
            
            # Natural sort function for labels with numbers
            def natural_sort_key(s):
                import re
                return [int(text) if text.isdigit() else text.lower()
                        for text in re.split('([0-9]+)', str(s))]
            
            # Sort each group separately using natural sort
            hue_sorted = sorted(zip(hue_labels, hue_handles), key=lambda x: natural_sort_key(x[0]))
            style_sorted = sorted(zip(style_labels, style_handles), key=lambda x: natural_sort_key(x[0]))

            hue_sorted = [h for h in hue_sorted if h[0] != hue]
            style_sorted = [h for h in style_sorted if h[0] != style]
            
            # Create two legend sections - one for hue and one for style
            hue_labels, hue_handles = zip(*hue_sorted)
            style_labels, style_handles = zip(*style_sorted)
            
            # Create legend with sections and titles
            legend_elements = []
            # Add hue title and elements
            legend_elements.append(plt.Line2D([0], [0], color='none', label=hue))
            legend_elements.extend(hue_handles)
            # Add style title and elements
            legend_elements.append(plt.Line2D([0], [0], color='none', label=style))
            legend_elements.extend(style_handles)
            
            # Create labels list with titles
            legend_labels = [hue] + list(hue_labels) + [style] + list(style_labels)
            
            ax.legend(legend_elements, legend_labels,
                     bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # If only one type, sort normally using natural sort
            def natural_sort_key(s):
                import re
                return [int(text) if text.isdigit() else text.lower()
                        for text in re.split('([0-9]+)', str(s))]
                
            sorted_pairs = sorted(zip(labels, handles), key=lambda x: natural_sort_key(x[0]))
            labels, handles = zip(*sorted_pairs)
            
            # Update legend with title
            title = hue if hue is not None else style
            legend_elements = [plt.Line2D([0], [0], color='none', label=title)] + list(handles)
            legend_labels = [title] + list(labels)
            
            ax.legend(legend_elements, legend_labels,
                     bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.xticks()
    plt.tight_layout()
    return ax

def histogram_plot(data, x, hue=None, legend=True, **kwargs):  # TODO: add logic for when distributions overlap too much
    """
    Create a histogram.
    
    Parameters:
        data (pd.DataFrame): Input dataframe
        x (str): Column name for x-axis
        hue (str, optional): Column name for color mapping
        legend (bool, optional): Whether to show legend. Defaults to True.
        **kwargs: Additional keyword arguments to pass to sns.histplot
        
    Returns:
        matplotlib.axes.Axes: The plot object
    """
    data = data.sort_values(by=x, key=natural_keys)

    if hue is None:
        ax = sns.histplot(data=data, x=x, color=color_palette[np.random.choice(range(len(color_palette)))], **kwargs)
    else:
        rand_colors = np.random.choice(
            range(len(sns.color_palette(palette="Dark2", n_colors=len(data[hue].unique())))), 
            size=len(data[hue].unique()), 
            replace=False,
        )
        palette = [sns.color_palette(palette="Dark2", n_colors=len(data[hue].unique()))[i] for i in rand_colors]
        ax = sns.histplot(data=data, x=x, hue=hue, palette=palette, **kwargs)

    if not legend:
        ax.legend().set_visible(False)
    elif ax.get_legend() is not None:
        ax.get_legend().set_bbox_to_anchor((1.05, 1))
        ax.get_legend().set_loc('upper left')

    if data.is_categorical(x) or isinstance(data[x][0], str):
        ax.set_xticks(range(len(data[x].unique())))
        ax.set_xticklabels(data[x].unique())
    else:  # binned continuous data
        # Get the min and max values of the data
        data_min = data[x].min()
        data_max = data[x].max()
        
        # Get the number of bins from the plot
        num_bins = len(ax.containers[0])
        
        # Calculate bin edges evenly spaced between min and max
        bin_edges = np.linspace(data_min, data_max, num_bins + 1)
        
        # Determine interval based on number of bins
        if num_bins == 15:
            interval = 3
        elif num_bins == 20:
            interval = 4
        else:
            interval = 1
            
        # Select edges at the specified interval
        selected_edges = bin_edges[::interval]
        if bin_edges[-1] not in selected_edges:
            selected_edges = np.append(selected_edges, bin_edges[-1])
            
        # Set ticks at selected bin edges and label with edge values
        ax.set_xticks(selected_edges)
        ax.set_xticklabels([f'{edge:.2f}' for edge in selected_edges], rotation=0, ha='center')
    
    plt.tight_layout()
    return ax

def bar_plot(data, x, y, hue=None, legend=True, **kwargs):
    """
    Create a bar plot. If multiple y values are provided, creates a stacked bar chart.
    
    Parameters:
        data (pd.DataFrame): Input dataframe
        x (str): Column name for x-axis
        y (str or list): Column name(s) for y-axis
        hue (str, optional): Column name for color mapping
        legend (bool, optional): Whether to show legend. Defaults to True.
    
    Returns:
        matplotlib.axes.Axes: The plot object
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        palette = np.random.permutation(color_palette)[:1 if hue is None else len(data[hue].unique()) + 1]
        # Sort data by x values first
        sorted_data = data.sort_values(by=x, key=natural_keys)
        ax = sns.barplot(data=sorted_data, x=x, y=y, hue=hue, palette=list(palette), **kwargs)
    
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")  # Zero line

    ax.set_ylabel(f"{y} (mean)")
    
    if not legend:
        ax.legend().set_visible(False)
    elif ax.get_legend() is not None:
        ax.get_legend().set_bbox_to_anchor((1.05, 1))
        ax.get_legend().set_loc('upper left')
    plt.xticks()
    
    plt.tight_layout()
    return ax

def heatmap_plot(data, x, y, hue=None, legend=True, correlation=False):  # TODO: fix
    """
    Create a heatmap.
    
    Parameters:
        data (pd.DataFrame): Input dataframe
        x (str): Column name for x-axis
        y (str): Column name for y-axis
        hue (str, optional): Column name for values in heatmap
        legend (bool, optional): Whether to show colorbar. Defaults to True.
        correlation (bool, optional): Whether to plot correlation matrix. Defaults to False.
    
    Returns:
        matplotlib.axes.Axes: The plot object
    """
    assert hue is not None or correlation, "hue must be provided if correlation is False"

    if hue is not None:
        pivot_data = data.pivot(index=y, columns=x, values=hue)
        ax = sns.heatmap(pivot_data, cbar=legend)
    else:   
        pivot_data = data[[x, y]].corr()
        ax = sns.heatmap(pivot_data, cbar=legend, vmin=-1, vmax=1, cmap="coolwarm")
    plt.xticks()
    return ax

def line_plot(data, x, y, hue=None, legend=True, **kwargs):
    """
    Create a line plot. If multiple y values are provided, creates multiple lines.
    
    Parameters:
        data (pd.DataFrame): Input dataframe
        x (str): Column name for x-axis
        y (str): Column name(s) for y-axis
        hue (str, optional): Column name for color mapping
        legend (bool, optional): Whether to show legend. Defaults to True.
        
    Returns:
        matplotlib.axes.Axes: The plot object
    """
    if data.is_sequential(x):
        data = data.sort_values(by=x, key=natural_keys)
    ax = sns.lineplot(data=data, x=x, y=y, hue=hue, **kwargs)
    
    ax.set_ylabel(f"{y} (mean)")
    
    if not legend:
        ax.legend().set_visible(False)
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks()
    
    plt.tight_layout()
    return ax

def box_plot(data, x, y, hue=None, legend=True):  # TODO: fix
    """
    Create a box plot.
    
    Parameters:
        data (pd.DataFrame): Input dataframe
        x (str): Column name for x-axis
        y (str or list): Column name(s) for y-axis (only first y used if multiple provided)
        hue (str, optional): Column name for color mapping
        legend (bool, optional): Whether to show legend. Defaults to True.
        
    Returns:
        matplotlib.axes.Axes: The plot object
    """
    y_val = y[0] if isinstance(y, list) else y
    ax = sns.boxplot(data=data, x=x, y=y_val, hue=hue)
    
    if not legend:
        ax.legend().set_visible(False)
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks()
    plt.tight_layout()
    return ax

def violin_plot(data, x, y, hue=None, legend=True):
    """
    Create a violin plot.
    
    Parameters:
        data (pd.DataFrame): Input dataframe
        x (str): Column name for x-axis
        y (str or list): Column name(s) for y-axis (only first y used if multiple provided)
        hue (str, optional): Column name for color mapping
        legend (bool, optional): Whether to show legend. Defaults to True.
        
    Returns:
        matplotlib.axes.Axes: The plot object
    """
    y_val = y[0] if isinstance(y, list) else y
    ax = sns.violinplot(data=data, x=x, y=y_val, hue=hue)
    
    if not legend:
        ax.legend().set_visible(False)
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks()
    plt.tight_layout()
    return ax
