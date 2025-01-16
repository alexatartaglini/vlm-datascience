import itertools
import string
import os
import sys
import yaml

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter

# Get the project root directory (assuming it's one level up from rendering/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from generators import Generator, Data
from .utils import *


def random_string(length=10):
    characters = string.ascii_letters + string.digits  # abcdef...XYZ012345...
    return ''.join(np.random.choice(list(characters), size=length))


def random_render_id(length=10):
    render_id = random_string(length)
    while os.path.exists(os.path.join(PROJECT_ROOT, f"data/metadata/renders/{render_id}.yaml")):
        render_id = random_string(length)
    return render_id


def get_metadata(**kwargs):
    return kwargs


def save_renders(ax, table, render_id):
    plt.savefig(os.path.join(PROJECT_ROOT, f"data/plots/{render_id}.png"))
    plt.tight_layout()
    plt.close()

    with open(os.path.join(PROJECT_ROOT, f"data/tables/{render_id}.txt"), "w") as f:
        f.write(table)


def save_metadata(**kwargs):
    metadata = get_metadata(**kwargs)  # Store metadata for plot & table
    metadata["renders"] = {  # TODO: store more sophisticated metadata for plots & tables, e.g. table format
        "plot": f"data/plots/{metadata['render_id']}.png",
        "table": f"data/tables/{metadata['render_id']}.txt",
    }

    with open(os.path.join(PROJECT_ROOT, f"data/metadata/renders/{metadata['render_id']}.yaml"), "w") as f:
        yaml.dump(metadata, f)


def clear_renders():
    for file in os.listdir(os.path.join(PROJECT_ROOT, "data/plots")):
        os.remove(os.path.join(PROJECT_ROOT, f"data/plots/{file}"))
    for file in os.listdir(os.path.join(PROJECT_ROOT, "data/tables")):
        os.remove(os.path.join(PROJECT_ROOT, f"data/tables/{file}"))
    for file in os.listdir(os.path.join(PROJECT_ROOT, "data/metadata/renders")):
        os.remove(os.path.join(PROJECT_ROOT, f"data/metadata/renders/{file}"))


def autorender_scatter(data):
    """
    Generate all scatter plots and "raw" tables for a given dataset.
    """
    if data.n_numeric < 2:  # Invalid
        return

    renders = {}

    generator_name = data.generator_name
    dataset_name = data.name

    # Two variable combinations
    xy = itertools.combinations(data.numeric_vars, 2)
    for x, y in xy:
        ax = scatter_plot(data, x=x, y=y)
        table = raw_table(data, format="csv", variables=[x, y])
        render_id = random_render_id()
        renders[render_id] = {
            "render_id": render_id,
            "x": x,
            "x_type": "numeric",
            "y": y,
            "y_type": "numeric",
            "num_vars": 2,
            "legend": False,
        }
        save_renders(ax, table, render_id)

    # Three variable combinations
    if data.n_categorical > 0:
        hues = data.categorical_vars
        xy = itertools.permutations(data.numeric_vars, 2)

        for (x, y), hue in itertools.product(xy, hues):
            ax = scatter_plot(data, x=x, y=y, hue=hue)
            table = raw_table(data, format="csv", variables=[x, y, hue])
            render_id = random_render_id()
            try:
                hue_values = [t.item() for t in data[hue].unique()]
            except:
                hue_values = [t for t in data[hue].unique()]
            
            renders[render_id] = {
                "render_id": render_id,
                "x": x,
                "x_type": "numeric",
                "y": y,
                "y_type": "numeric",
                "hue": hue,
                "hue_type": "categorical",
                "hue_values": hue_values,
                "num_vars": 3,
                "legend": True,
            }
            save_renders(ax, table, render_id)

    # TODO: fix colormap
    if data.n_numeric > 2:
        for x, y, hue in itertools.permutations(data.numeric_vars, 3):
            temp_data = data.copy() 
            temp_data[hue] = pd.cut(temp_data[hue], bins=5)
            ax = scatter_plot(temp_data, x=x, y=y, hue=hue)
            table = raw_table(data, format="csv", variables=[x, y, hue], group_by=hue)
            render_id = random_render_id()
            renders[render_id] = {
                "render_id": render_id,
                "x": x,
                "x_type": "numeric",
                "y": y,
                "y_type": "numeric",
                "hue": hue,
                "hue_type": "numeric",
                "num_vars": 3,
                "legend": True,
            }
            save_renders(ax, table, render_id)

    # Four variable combinations
    if data.n_categorical > 1:
        xy = itertools.permutations(data.numeric_vars, 2)
        huestyle = itertools.permutations(data.categorical_vars, 2)

        for (x, y), (hue, style) in itertools.product(xy, huestyle):
            ax = scatter_plot(data, x=x, y=y, hue=hue, style=style)
            table = raw_table(data, format="csv", variables=[x, y, hue, style])
            render_id = random_render_id()
            renders[render_id] = {
                "render_id": render_id,
                "x": x,
                "x_type": "numeric",
                "y": y,
                "y_type": "numeric",
                "hue": hue,
                "hue_type": "categorical",
                "style": style,
                "style_type": "categorical",
                "num_vars": 4,
                "legend": True,
            }
            save_renders(ax, table, render_id)

        if data.n_numeric > 2:
            xyhue = itertools.permutations(data.numeric_vars, 3)
            for (x, y, hue), style in itertools.product(xyhue, data.categorical_vars):

                if len(data[style].unique()) <= 4:
                    ax = scatter_plot(data, x=x, y=y, hue=hue, style=style)
                    table = raw_table(data, format="csv", variables=[x, y, hue, style])
                    render_id = random_render_id()
                    renders[render_id] = {
                        "render_id": render_id,
                        "x": x,
                        "x_type": "numeric",
                        "y": y,
                        "y_type": "numeric",
                        "hue": hue,
                        "hue_type": "numeric",
                        "style": style,
                        "style_type": "categorical",
                        "num_vars": 4,
                        "legend": True,
                    }
                    save_renders(ax, table, render_id)
                else:
                    continue

    for render_id, render in renders.items():
        renders[render_id]["type"] = "scatter"
        renders[render_id]["generator_name"] = generator_name
        renders[render_id]["dataset_name"] = dataset_name

        save_metadata(**render)

def autorender_histogram(data, generator):
    """
    Generate all histogram plots and frequency tables for a given dataset.
    """
    renders = {}

    generator_name = data.generator_name
    dataset_name = data.name

    # Single variable
    for x in data.numeric_vars:
        bins = np.random.choice([10, 15, 20])
        ax = histogram_plot(data, x=x, discrete=False, bins=bins)
        table = frequency_table(data, format="pandas", variables=[x], bins=bins)
        render_id = random_render_id()
        renders[render_id] = {
            "render_id": render_id,
            "x": x,
            "x_type": "numeric",
            "num_vars": 1,
            "legend": False,
            "bins": bins.item(),
            "plot_xticks": [t.item() for t in ax.get_xticks()],
        }
        save_renders(ax, table, render_id)

    for x in data.categorical_vars:
        ax = histogram_plot(data, x=x, discrete=True)
        table = frequency_table(data, format="pandas", variables=[x])
        render_id = random_render_id()
        renders[render_id] = {
            "render_id": render_id,
            "x": x,
            "x_type": "categorical",
            "num_vars": 1,
            "legend": False,
            "bins": len(data[x].unique()),
        }
        save_renders(ax, table, render_id)

    # Two variable combinations
    if data.n_categorical > 0:
        for x, hue in itertools.product(data.numeric_vars, data.categorical_vars):
            if len(data[hue].unique()) <= 4:
                bins = np.random.choice([10, 15, 20])
                ax = histogram_plot(data, x=x, hue=hue, bins=bins)
                table = frequency_table(data, format="pandas", variables=[x], group_by=hue, bins=bins)
                render_id = random_render_id()

                try:
                    hue_values = [t.item() for t in data[hue].unique()]
                except:
                    hue_values = [t for t in data[hue].unique()]

                renders[render_id] = {
                    "render_id": render_id,
                    "x": x,
                    "x_type": "numeric",
                    "hue": hue,
                    "hue_type": "categorical",
                    "hue_values": hue_values,
                    "num_vars": 2,
                    "legend": True,
                    "bins": bins.item(),
                    "plot_xticks": [t.item() for t in ax.get_xticks()],
                }
                save_renders(ax, table, render_id)

    if data.n_categorical > 1:
        for x, hue in itertools.permutations(data.categorical_vars, 2):
            if len(data[hue].unique()) <= 4:
                ax = histogram_plot(data, x=x, hue=hue, discrete=True, multiple="stack")
                table = frequency_table(data, format="pandas", variables=[x], group_by=hue)
                render_id = random_render_id()

                try:
                    hue_values = [t.item() for t in data[hue].unique()]
                except:
                    hue_values = [t for t in data[hue].unique()]

                renders[render_id] = {
                    "render_id": render_id,
                    "x": x,
                    "x_type": "categorical",
                    "hue": hue,
                    "hue_type": "categorical",
                    "hue_values": hue_values,
                    "subtype": "stacked",
                    "num_vars": 2,
                    "legend": True,
                    "bins": len(data[x].unique()),
                }
                save_renders(ax, table, render_id)
    
    # TODO: Find plots with similar structure
    for render_id, render in renders.items():
        renders[render_id]["type"] = "histogram"
        renders[render_id]["generator_name"] = generator_name
        renders[render_id]["dataset_name"] = dataset_name

        save_metadata(**render)


def autorender_bar(data):
    """
    Generate all bar plots and mean tables for a given dataset.
    """
    if data.n_categorical < 1 or data.n_numeric < 1:  # Invalid
        return

    renders = {}
    
    generator_name = data.generator_name
    dataset_name = data.name

    # Two variable combinations
    for x, y in itertools.product(data.categorical_vars, data.numeric_vars):
        ax = bar_plot(data, x=x, y=y, errorbar=None)
        table = mean_table(data, format="pandas", variables=[x], values=[y])
        render_id = random_render_id()
        renders[render_id] = {
            "render_id": render_id,
            "x": x,
            "x_type": "categorical",
            "y": y,
            "y_type": "numeric",
            "num_vars": 2,
            "legend": False,
            "agg_func": "mean",
        }
        save_renders(ax, table, render_id)

    # Three variable combinations
    if data.n_categorical > 1:
        xhue = itertools.permutations(data.categorical_vars, 2)
        for (x, hue), y in itertools.product(xhue, data.numeric_vars):
            ax = bar_plot(data, x=x, y=y, hue=hue, errorbar=None)
            table = mean_table(data, format="pandas", variables=[x, hue], values=[y])
            render_id = random_render_id()
            renders[render_id] = {
                "render_id": render_id,
                "x": x,
                "x_type": "categorical",
                "y": y,
                "y_type": "numeric",
                "hue": hue,
                "hue_type": "categorical",
                "num_vars": 3,
                "legend": True,
                "agg_func": "mean",
            }
            save_renders(ax, table, render_id)

    # TODO: Find plots with similar structure
    for render_id, render in renders.items():
        renders[render_id]["type"] = "bar"
        renders[render_id]["generator_name"] = generator_name
        renders[render_id]["dataset_name"] = dataset_name

        save_metadata(**render)


def autorender_heatmap(data):
    """
    Generate all heatmap plots and correlation tables for a given dataset.
    """
    pass


def autorender_line(data):
    """
    Generate all line plots and confidence tables for a given dataset.
    """
    if data.n_numeric < 1:  # Invalid
        return

    renders = {}
    
    generator_name = data.generator_name
    dataset_name = data.name

    # Two variable combinations
    for x, y in itertools.combinations(data.numeric_vars, 2):  # TODO: add smoothing
        ax = line_plot(data, x=x, y=y)
        table = confidence_table(data, format="pandas", variables=[x], values=[y], bins=10)
        render_id = random_render_id()
        renders[render_id] = {
            "render_id": render_id,
            "x": x,
            "x_type": "numeric",
            "y": y,
            "y_type": "numeric",
            "num_vars": 2,
            "legend": False,
            "bins": 10,
        }
        save_renders(ax, table, render_id)

    if data.n_sequential > 0:
        for x, y in itertools.product(data.sequential_vars, data.numeric_vars):
            ax = line_plot(data, x=x, y=y)
            table = confidence_table(data, format="pandas", variables=[x], values=[y])
            render_id = random_render_id()
            renders[render_id] = {
                "render_id": render_id,
                "x": x,
                "x_type": "categorical",
                "y": y,
                "y_type": "numeric",
                "num_vars": 2,
                "legend": False,
            }
            save_renders(ax, table, render_id)

    # Three variable combinations
    if data.n_categorical > 0:  # TODO: add smoothing
        xy = itertools.combinations(data.numeric_vars, 2)
        for (x, y), hue in itertools.product(xy, data.categorical_vars):
            ax = line_plot(data, x=x, y=y, hue=hue)
            table = confidence_table(data, format="pandas", variables=[x], values=[y], group_by=[hue], bins=10)
            render_id = random_render_id()
            renders[render_id] = {
                "render_id": render_id,
                "x": x,
                "x_type": "numeric",
                "y": y,
                "y_type": "numeric",
                "hue": hue,
                "hue_type": "categorical",
                "hue_values": list(data[hue].unique()),
                "num_vars": 3,
                "legend": True,
                "bins": 10,
            }
            save_renders(ax, table, render_id)

        if data.n_sequential > 0 and data.n_categorical > 1:
            xhue = itertools.permutations(data.categorical_vars, 2)
            xhue = [i for i in xhue if i[0] in data.sequential_vars]
            for (x, hue), y in itertools.product(xhue, data.numeric_vars):
                ax = line_plot(data, x=x, y=y, hue=hue)
                table = confidence_table(data, format="pandas", variables=[x], values=[y], group_by=[hue])
                render_id = random_render_id()
                renders[render_id] = {
                    "render_id": render_id,
                    "x": x,
                    "x_type": "categorical",
                    "y": y,
                    "y_type": "numeric",
                    "hue": hue,
                    "hue_type": "categorical",
                    "hue_values": [t.item() for t in data[hue].unique()],
                    "num_vars": 3,
                    "legend": True,
                }
                save_renders(ax, table, render_id)

    # TODO: 4 variable combinations w/ linestyle
    # TODO: Find plots with similar structure
    for render_id, render in renders.items():
        renders[render_id]["type"] = "line"
        renders[render_id]["generator_name"] = generator_name
        renders[render_id]["dataset_name"] = dataset_name

        save_metadata(**render)


def autorender_data(data, clear_existing=False):
    """
    Generate all plots and tables for a given dataset.

    Args:
        data (Data): The dataset to render.
        clear_existing (bool): Whether to clear existing renders.
    """
    if clear_existing:
        clear_renders()

    data_name = data.name
    generator = Generator(config=f"data/generator_configs/generator_{data_name.split('_')[-1]}.yaml")

    autorender_scatter(data)
    autorender_histogram(data, generator)
    autorender_bar(data)
    #autorender_line(data)  # TODO: fix line plots