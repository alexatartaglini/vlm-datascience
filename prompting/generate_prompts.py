import os
import yaml
from pathlib import Path

import pandas as pd
import numpy as np
from typing import List, Dict, Any

from prompting.utils import to_ordinal
from generators import Generator

PROJECT_ROOT = Path(__file__).parent.parent


def load_dataset_metadata(dataset_name: str) -> Dict[str, Any]:
    """Load metadata for a dataset from its YAML file"""
    metadata_path = os.path.join(PROJECT_ROOT, f"data/metadata/datasets/{dataset_name}.yaml")
    with open(metadata_path) as f:
        return yaml.safe_load(f)


def load_generator_config(generator_name: str) -> Dict[str, Any]:
    """Load configuration for a generator from its YAML file"""
    config_path = os.path.join(PROJECT_ROOT, f"data/generator_configs/{generator_name}.yaml") 
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_render_metadata(dataset_name: str) -> List[Dict[str, Any]]:
    """Load metadata for all renders associated with a dataset"""
    render_dir = os.path.join(PROJECT_ROOT, "data/metadata/renders")
    render_metadata = []
    
    for render_id in os.listdir(render_dir):
        with open(os.path.join(render_dir, render_id)) as f:
            metadata = yaml.safe_load(f)
            if metadata["dataset_name"] == dataset_name:
                render_metadata.append(metadata)
                
    return render_metadata


def generate_retrieval_prompts(render_metadata: Dict[str, Any], table_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate tier 1 'reading the data' prompts - both quantitative and qualitative.

    These prompts are designed to test the model's ability to retrieve specific values from the data.
    """
    prompts = []
    
    if render_metadata["type"] == "scatter":  # TODO: the scatter prompts are unfair for large quantities of data; don't take hue into account
        x, y = render_metadata["x"], render_metadata["y"]
        point = table_data.sample(n=1).iloc[0]
        # Quantitative
        prompts.append({
            "prompt": f"approximately what is the {y} value for {x}={point[x]:.2f}?",
            "answer": point[y].item(),
            "prompt_type": "quantitative",
            "prompt_tier": "retrieval",
            #"operations": ["retrieve*1"],  # TODO: add operations and counts to each prompt
        })
        """  # TODO: Fix this prompt
        # Qualitative
        points = table_data.sample(n=2)
        prompts.append({
            "prompt": f"Which point has a larger {y} value: the point at {x}={points.iloc[0][x]:.2f} or {x}={points.iloc[1][x]:.2f}?",
            "answer": f"{x}={points.iloc[0][x]:.2f}" if points.iloc[0][y] > points.iloc[1][y] else f"{x}={points.iloc[1][x]:.2f}",
            "prompt_type": "qualitative",
            "prompt_tier": "retrieval",
        })
        """

    elif render_metadata["type"] == "histogram":
        x = render_metadata["x"]

        if "hue" not in render_metadata:
            counts = table_data["count"]

            if render_metadata["x_type"] == "categorical":
                category = table_data[x].sample(n=1).iloc[0]
                count = table_data.loc[table_data[x] == category, 'count'].iloc[0]

                # Quantitative
                prompts.append({
                    "prompt": f"approximately how many observations are there for {x}={category}?",
                    "answer": count.item(),
                    "prompt_type": "quantitative",
                    "prompt_tier": "retrieval",
                })
            else:
                num_bins = render_metadata["bins"]
                bin_idx = np.random.choice([0, 1, 2, num_bins - 1]).item()  # np.random.choice(range(num_bins)).item() TODO: maybe fix this prompt to be more flexible
                bin_label = "last" if bin_idx == num_bins - 1 else to_ordinal(bin_idx + 1)

                # Quantitative
                prompts.append({
                    "prompt": f"approximately how many observations are in the {bin_label} bin?",
                    "answer": counts[bin_idx].item(),
                    "prompt_type": "quantitative",
                    "prompt_tier": "retrieval",
                })
        else:
            hue = render_metadata["hue"]
            category2 = np.random.choice([col.split('=')[1] for col in table_data.columns if col.startswith(f"{hue}=")])
            #category2 = table_data[hue].sample(n=1).iloc[0]
            
            if render_metadata["x_type"] == "categorical":
                category = table_data[x].sample(n=1).iloc[0]
                count = table_data.loc[(table_data[x] == category)][f"{hue}={category2}"].iloc[0]

                # Quantitative
                prompts.append({
                    "prompt": f"approximately how many observations are there for {x}={category} and {hue}={category2}?",
                    "answer": count.item(),
                    "prompt_type": "quantitative",
                    "prompt_tier": "retrieval",
                })
            else:
                num_bins = render_metadata["bins"]
                bin_idx = np.random.choice([0, 1, 2, num_bins - 1]).item()
                bin_label = "last" if bin_idx == num_bins - 1 else to_ordinal(bin_idx + 1)
                
                counts = table_data[f"{hue}={category2}"].reset_index(drop=True)

                # Quantitative
                prompts.append({
                    "prompt": f"approximately how many observations are in the {bin_label} bin for {hue}={category2}?",
                    "answer": counts[bin_idx].item(),
                    "prompt_type": "quantitative",
                    "prompt_tier": "retrieval",
                })
            
        """  # TODO: Fix this prompt
        # Qualitative
        prompts.append({
            "prompt": f"Which bin contains the most observations?",
            "answer": f"{max_bin}",
            "prompt_type": "qualitative",
            "prompt_tier": "retrieval",
        })
        """

    elif render_metadata["type"] == "bar":
        x, y = render_metadata["x"], render_metadata["y"]
        category = table_data[x].sample(n=1).iloc[0]
        if "hue" not in render_metadata:
            value = table_data.loc[table_data[x] == category, f"{y} (mean)"].iloc[0]
            # Quantitative
            prompts.append({
                "prompt": f"approximately what is the mean {y} value for {x}={category}?",
                "answer": value.item(),
                "prompt_type": "quantitative",
                "prompt_tier": "retrieval",
            })
        else:
            hue = render_metadata["hue"]
            category2 = table_data[table_data[x] == category][hue].sample(n=1).iloc[0]
            value = table_data.loc[(table_data[x] == category) & (table_data[hue] == category2), f"{y} (mean)"].iloc[0]
            # Quantitative
            prompts.append({
                "prompt": f"approximately what is the mean {y} value for {x}={category} and {hue}={category2}?",
                "answer": value.item(),
                "prompt_type": "quantitative",
                "prompt_tier": "retrieval",
            })
        """  # TODO: Fix this prompt
        # Qualitative
        categories = table_data[x].sample(n=2).tolist()
        values = table_data[table_data[x].isin(categories)][y]
        prompts.append({
            "prompt": f"Which category has a larger {y} value: {categories[0]} or {categories[1]}?",
            "answer": f"{categories[0]}" if values.iloc[0] > values.iloc[1] else f"{categories[1]}",
            "prompt_type": "qualitative",
            "prompt_tier": "retrieval",
        })
        """

    elif render_metadata["type"] == "line":  # TODO: fix line prompts
        x, y = render_metadata["x"], render_metadata["y"]
        point = table_data.sample(n=1).iloc[0]
        # Quantitative
        prompts.append({
            "prompt": f"approximately what is the {y} value at {x}={point[x]}?",
            "answer": point[y].item(),
            "prompt_type": "quantitative",
            "prompt_tier": "retrieval",
        })
        """  # TODO: Fix this prompt
        # Qualitative
        start_y = table_data[y].iloc[0]
        end_y = table_data[y].iloc[-1]
        trend = "increasing" if end_y > start_y else "decreasing"
        prompts.append({
            "prompt": f"Is the trend in {y} increasing or decreasing over {x}?",
            "answer": trend,
            "prompt_type": "qualitative",
            "prompt_tier": "retrieval",
        })
        """
        
    return prompts


def generate_arithmetic_prompts(render_metadata: Dict[str, Any], table_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate tier 2a 'reading between the data' prompts - both quantitative and qualitative.

    These prompts are designed to test the model's ability to perform basic arithmetic on the data.

    """
    prompts = []

    if render_metadata["type"] == "scatter":
        # Compute range of x or y axis
        x, y = render_metadata["x"], render_metadata["y"]
        # Randomly select x or y axis
        selected_axis = np.random.choice([x, y])
        axis_range = table_data[selected_axis].max() - table_data[selected_axis].min()
        prompts.append({
            "prompt": f"approximately what is the range of {selected_axis} values in the data?",
            "answer": float(f"{axis_range:.2f}"),
            "prompt_type": "quantitative", 
            "prompt_tier": "arithmetic"
        })

        # TODO: qualitative prompts

    elif render_metadata["type"] == "histogram":
        x = render_metadata["x"]
        
        if "hue" in render_metadata:
            hue_var = render_metadata["hue"]
            hue_val = np.random.choice([col.split('=')[1] for col in table_data.columns if col.startswith(f"{hue_var}=")])

            if render_metadata["x_type"] == "numeric":
                # Range sum for specific hue
                # Get random consecutive range from xticks
                xticks = render_metadata["plot_xticks"]
                range_start_idx = np.random.randint(len(xticks)-1)
                range_start, range_end = xticks[range_start_idx], xticks[range_start_idx + 1]

                hue_data = table_data[[x, f"{hue_var}={hue_val}"]]
                range_count = 0
                for i in range(len(hue_data)):
                    irange_start = float(hue_data.iloc[i][x].split(',')[0].replace('(', '').strip())
                    irange_end = float(hue_data.iloc[i][x].split(',')[-1].replace(']', '').strip())
                    if float(f"{range_start:.2f}") <= irange_start and irange_end <= float(f"{range_end:.2f}"):
                        range_count += hue_data.iloc[i][f"{hue_var}={hue_val}"]

                prompts.append({
                    "prompt": f"for {hue_var}={hue_val}, approximately how many total observations occur between {x}={range_start:.2f} and {x}={range_end:.2f}?",
                    "answer": range_count.item(),
                    "prompt_type": "quantitative",
                    "prompt_tier": "arithmetic"
                })
            else:
                # Category difference for specific hue
                cats = list(table_data[x].unique())
                categories = np.random.choice(cats, size=2, replace=False)
                hue_data = table_data[f"{hue_var}={hue_val}"]
                count_diff = abs(hue_data.iloc[cats.index(categories[0])] - hue_data.iloc[cats.index(categories[1])])
                prompts.append({
                    "prompt": f"for {hue_var}={hue_val}, approximately what is the difference in number of observations between {categories[0]} and {categories[1]}?",
                    "answer": count_diff,
                    "prompt_type": "quantitative",
                    "prompt_tier": "arithmetic"
                })
        else:
            # Check if x is numeric (binned) or categorical
            if render_metadata["x_type"] == "numeric":
                # Get random consecutive range from xticks
                xticks = render_metadata["plot_xticks"]
                range_start_idx = np.random.randint(len(xticks)-1)
                range_start, range_end = xticks[range_start_idx], xticks[range_start_idx + 1]

                range_count = 0
                for i in range(len(table_data)):
                    irange_start = float(table_data.iloc[i][x].split(',')[0].replace('(', '').strip())
                    irange_end = float(table_data.iloc[i][x].split(',')[-1].replace(']', '').strip())

                    if float(f"{range_start:.2f}") <= irange_start and irange_end <= float(f"{range_end:.2f}"):
                        range_count += table_data.iloc[i]["count"]
                prompts.append({
                    "prompt": f"approximately how many observations occur between {x}={range_start:.2f} and {x}={range_end:.2f}?",
                    "answer": range_count.item(),
                    "prompt_type": "quantitative",
                    "prompt_tier": "arithmetic"
                })
            else:
                # Get difference between two random categories
                categories = np.random.choice(table_data[x].unique(), size=2, replace=False)
                count_diff = abs(table_data.loc[(table_data[x] == categories[0]), "count"].item() - table_data.loc[(table_data[x] == categories[1]), "count"].item())
                prompts.append({
                    "prompt": f"approximately what is the difference in number of observations between {categories[0]} and {categories[1]}?",
                    "answer": count_diff,
                    "prompt_type": "quantitative",
                    "prompt_tier": "arithmetic"
                })

    elif render_metadata["type"] == "bar":
        x = render_metadata["x"]
        y = render_metadata["y"]

        if "hue" in render_metadata:
            # Get difference between two random categories for specific hue
            hue_var = render_metadata["hue"]
            hues = table_data[hue_var].unique()
            hue_val = np.random.choice(hues)
            xs = table_data[table_data[hue_var] == hue_val][x].unique()

            if len(xs) > 1:
                categories = np.random.choice(xs, size=2, replace=False)

                count_diff = abs(table_data.loc[(table_data[x] == categories[0]) & (table_data[hue_var] == hue_val), f"{y} (mean)"].item() - table_data.loc[(table_data[x] == categories[1]) & (table_data[hue_var] == hue_val), f"{y} (mean)"].item())
                prompts.append({
                    "prompt": f"for {hue_var}={hue_val}, approximately what is the difference in mean {y} value between {x}={categories[0]} and {x}={categories[1]}?",
                    "answer": count_diff,
                    "prompt_type": "quantitative",
                    "prompt_tier": "arithmetic"
                })

            # Get difference between two random hues for specific category
            category = np.random.choice(table_data[x].unique())
            hues = table_data[table_data[x] == category][hue_var].unique()

            if len(hues) > 1:
                hues = np.random.choice(hues, size=2, replace=False)

                count_diff = abs(table_data.loc[(table_data[x] == category) & (table_data[hue_var] == hues[0]), f"{y} (mean)"].item() - table_data.loc[(table_data[x] == category) & (table_data[hue_var] == hues[1]), f"{y} (mean)"].item())
                prompts.append({
                    "prompt": f"for {x}={category}, approximately what is the difference in mean {y} value between {hue_var}={hues[0]} and {hue_var}={hues[1]}?",
                    "answer": count_diff,
                    "prompt_type": "quantitative",
                    "prompt_tier": "arithmetic"
                })
        else:
            categories = np.random.choice(table_data[x].unique(), size=2, replace=False)
            count_diff = abs(table_data.loc[(table_data[x] == categories[0]), f"{y} (mean)"].item() - table_data.loc[(table_data[x] == categories[1]), f"{y} (mean)"].item())
            prompts.append({
                "prompt": f"approximately what is the difference in mean {y} value between {x}={categories[0]} and {x}={categories[1]}?",
                "answer": count_diff,
                "prompt_type": "quantitative",
                "prompt_tier": "arithmetic"
            })

    elif render_metadata["type"] == "line":  # TODO: fix line prompts
        pass

    return prompts
    

def generate_boolean_prompts(render_metadata: Dict[str, Any], table_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate tier 2b 'reading between the data' prompts - both quantitative and qualitative.

    These prompts are designed to test the model's ability to make boolean judgements (e.g. greater than, 
    less than) about the data.
    """
    prompts = []
    
    if render_metadata["type"] == "scatter":
        x, y = render_metadata["x"], render_metadata["y"]
        # Randomly choose x or y axis
        chosen_axis = np.random.choice([x, y])
        
        # Find max value along chosen axis
        max_val = table_data[chosen_axis].max()
        min_val = table_data[chosen_axis].min()
        
        # Ask about max value
        prompts.append({
            "prompt": f"approximately what is the maximum {chosen_axis} value?",
            "answer": float(f"{max_val:.2f}"),
            "prompt_type": "quantitative",
            "prompt_tier": "boolean"
        })
        
        # Ask about min value 
        prompts.append({
            "prompt": f"approximately what is the minimum {chosen_axis} value?", 
            "answer": float(f"{min_val:.2f}"),
            "prompt_type": "quantitative",
            "prompt_tier": "boolean"
        })

        # Calculate ranges for both axes
        x_range = table_data[x].max() - table_data[x].min()
        y_range = table_data[y].max() - table_data[y].min()
        
        # Ask which range is larger
        larger_range = x if x_range > y_range else y
        prompts.append({
            "prompt": f"which has a larger range of values, {x} or {y}?",
            "answer": larger_range,
            "prompt_type": "qualitative",
            "prompt_tier": "boolean",
            "options": [x, y],
        })

    elif render_metadata["type"] == "histogram":
        if "hue" in render_metadata:
            # Get two random hue values
            hue = render_metadata["hue"]
            hue_values = render_metadata["hue_values"]
            chosen_hues = np.random.choice(hue_values, size=2, replace=False)
            
            # Sum total observations for each hue value
            counts = [table_data[f"{hue}={h}"].sum() for h in chosen_hues]
            larger_hue = chosen_hues[0] if counts[0] > counts[1] else chosen_hues[1]
            
            prompts.append({
                "prompt": f"which has more total observations, {hue}={chosen_hues[0]} or {hue}={chosen_hues[1]}?",
                "answer": larger_hue.item(),
                "prompt_type": "qualitative", 
                "prompt_tier": "boolean",
                "options": [h.item() for h in chosen_hues],
            })

            # Ask about max/min count for a single hue value
            chosen_hue = np.random.choice(hue_values)
            hue_counts = table_data[f"{hue}={chosen_hue}"]
            max_count = hue_counts.max()
            
            prompts.append({
                "prompt": f"what is the maximum count for {hue}={chosen_hue}?",
                "answer": float(f"{max_count:.0f}"),
                "prompt_type": "quantitative",
                "prompt_tier": "boolean"
            })
        else:
            # Get max and min counts
            x = render_metadata["x"]
            max_count = table_data['count'].max()
            min_count = table_data[table_data['count'] > 0]['count'].min()
            
            # Get x values associated with max/min counts
            max_x = table_data.loc[table_data['count'] == max_count, x].iloc[0]
            min_x = table_data.loc[table_data['count'] == min_count, x].iloc[0]

            if render_metadata["x_type"] == "numeric":
                max_x = ( float(max_x.split(',')[0].replace('(', '').strip()) + float(max_x.split(',')[-1].replace(']', '').strip()) ) / 2
                min_x = ( float(min_x.split(',')[0].replace('(', '').strip()) + float(min_x.split(',')[-1].replace(']', '').strip()) ) / 2
            
            prompts.append({
                "prompt": f"what is the maximum count for {x}?",  # TODO: fix answer formats. should be number or string when appropriate
                "answer": float(f"{max_count:.0f}"),
                "prompt_type": "quantitative",
                "prompt_tier": "boolean"
            })

            prompts.append({
                "prompt": f"which value of {x} appears most frequently?",
                "answer": float(f"{max_x:.2f}") if isinstance(max_x, (float, int)) else str(max_x),
                "prompt_type": "quantitative" if isinstance(max_x, (float, int)) else "qualitative",
                "prompt_tier": "boolean"
            })

            prompts.append({
                "prompt": f"which nonzero value of {x} appears least frequently?",
                "answer": float(f"{min_x:.2f}") if isinstance(min_x, (float, int)) else str(min_x),
                "prompt_type": "quantitative" if isinstance(min_x, (float, int)) else "qualitative",
                "prompt_tier": "boolean"
            })
            
            # Compare counts between two random bins
            if render_metadata["x_type"] == "numeric":
                # Choose two random x_ticks
                xticks = render_metadata["plot_xticks"]
                chosen_tick_indices = np.random.choice(len(xticks)-1, size=2, replace=False)
                chosen_ticks = [xticks[i] for i in chosen_tick_indices]
                
                # Find the corresponding bins in table_data
                chosen_bins = pd.DataFrame()
                for tick in chosen_ticks:
                    # Find the bin that starts with this tick
                    matching_bin = table_data[table_data[x].str.startswith(f"({tick:.2f}")].iloc[0]
                    chosen_bins = pd.concat([chosen_bins, pd.DataFrame([matching_bin])], ignore_index=True)
                
                bin1, bin2 = chosen_bins.iloc[0], chosen_bins.iloc[1]
            else:
                chosen_bins = table_data.sample(n=2)
                bin1, bin2 = chosen_bins.iloc[0], chosen_bins.iloc[1]
            more_frequent = bin1 if bin1['count'] > bin2['count'] else bin2
            
            if render_metadata["x_type"] == "numeric":
                avg1 = ( float(bin1[x].split(',')[0].replace('(', '').strip()) + float(bin1[x].split(',')[-1].replace(']', '').strip()) ) / 2
                avg2 = ( float(bin2[x].split(',')[0].replace('(', '').strip()) + float(bin2[x].split(',')[-1].replace(']', '').strip()) ) / 2
                prompt_text = f"which {x} value is more frequent, {x}={avg1:.2f} or {x}={avg2:.2f}?"
                answer = ( float(more_frequent[x].split(',')[0].replace('(', '').strip()) + float(more_frequent[x].split(',')[-1].replace(']', '').strip()) ) / 2
                answer = float(f"{answer:.2f}")
                options = [float(f"{avg1:.2f}"), float(f"{avg2:.2f}")]
            else:
                prompt_text = f"which {x} value is more frequent, {bin1[x]} or {bin2[x]}?"
                answer = str(more_frequent[x])
                options = [str(bin1[x]), str(bin2[x])]
            prompts.append({
                "prompt": prompt_text,
                "answer": answer,
                "prompt_type": "qualitative",
                "prompt_tier": "boolean",
                "options": options,
            })
        
    elif render_metadata["type"] == "bar":
        x, y = render_metadata["x"], render_metadata["y"]

        if "hue" in render_metadata:
            hue = render_metadata["hue"]
            hue_values = table_data[hue].unique()
            
            # Choose two random hue values
            chosen_hues = np.random.choice(hue_values, size=2, replace=False)
            
            # Choose one random x value
            chosen_x = np.random.choice(table_data[x].unique())
            
            # Get mean y values for the chosen x and hues
            try:
                y_val1 = table_data.loc[(table_data[x] == chosen_x) & (table_data[hue] == chosen_hues[0]), f"{y} (mean)"].iloc[0]
                y_val2 = table_data.loc[(table_data[x] == chosen_x) & (table_data[hue] == chosen_hues[1]), f"{y} (mean)"].iloc[0]

                higher_hue = chosen_hues[0] if y_val1 > y_val2 else chosen_hues[1]
            
                prompts.append({
                    "prompt": f"For {x}={chosen_x}, which has a higher mean {y} value: {hue}={chosen_hues[0]} or {hue}={chosen_hues[1]}?",
                    "answer": f"{hue}={higher_hue}",
                    "prompt_type": "qualitative",
                    "prompt_tier": "boolean",
                    "options": chosen_hues,
                })
            except:
                pass
        else:
            # Qualitative
            categories = table_data[x].tolist()
            values = table_data[f"{y} (mean)"]
            max_cat = table_data.loc[values.idxmax(), x]
            min_cat = table_data.loc[values.idxmin(), x]

            prompts.append({
                "prompt": f"which category has the highest mean {y} value?",
                "answer": max_cat,
                "prompt_type": "qualitative",
                "prompt_tier": "boolean",
                "options": categories,
            })

            prompts.append({
                "prompt": f"which category has the lowest mean {y} value?",
                "answer": min_cat,
                "prompt_type": "qualitative",
                "prompt_tier": "boolean",
                "options": categories,
            })

    elif render_metadata["type"] == "line":  # TODO: fix line prompts
        x, y = render_metadata["x"], render_metadata["y"]
        points = table_data.sample(n=2)
        # Quantitative
        diff = abs(points.iloc[0][y] - points.iloc[1][y])
        prompts.append({
            "prompt": f"What is the change in {y} between {x}={points.iloc[0][x]} and {x}={points.iloc[1][x]}?",
            "answer": f"{diff:.2f}"
        })
        # Qualitative
        mid_point = len(table_data) // 2
        first_half = table_data.iloc[:mid_point][y].mean()
        second_half = table_data.iloc[mid_point:][y].mean()
        prompts.append({
            "prompt": f"Is the average {y} value higher in the first half or second half of the {x} range?",
            "answer": "first half" if first_half > second_half else "second half"
        })
        
    return prompts


def generate_inference_prompts(render_metadata: Dict[str, Any], table_data: pd.DataFrame, generator_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate tier 3 'reading beyond the data' prompts - both quantitative and qualitative.

    These prompts are designed to test the model's ability to make inferences about the data based on its 
    relationship to the underlying generator.
    """
    prompts = []
    generator = Generator(config=generator_config)
    dependency_matrix = generator.dependency_matrix
    
    if render_metadata["type"] == "scatter":
        pass
        x, y = render_metadata["x"], render_metadata["y"]
        rel = dependency_matrix.get_relationship(x, y)
        if rel is not None and rel.type == "polynomial":
            degree = generator_config.get("polynomial_degree", 1)
            prompts.append({
                "prompt": f"what is the degree of the polynomial that best fits the relationship between {x} and {y}?",
                "answer": degree,
                "prompt_type": "quantitative",
                "prompt_tier": "inference"
            })
            prompts.append({
                "prompt": f"is the relationship between {x} and {y} linear or nonlinear?",
                "answer": "linear" if degree == 1 else "nonlinear",
                "prompt_type": "qualitative",
                "prompt_tier": "inference",
                "options": ["linear", "nonlinear"],
            })

        if "hue" in render_metadata:
            hue = render_metadata["hue"]
            rel = dependency_matrix.get_relationship(hue, y)
            if rel is None or rel.type != "booleanfunction" or rel.output_type == "std":
                rel = dependency_matrix.get_relationship(hue, x)

            if rel is None or rel.type != "booleanfunction" or rel.output_type == "std":
                rel = dependency_matrix.get_relationship(y, hue)

            if rel is None or rel.type != "booleanfunction" or rel.output_type == "std":
                rel = dependency_matrix.get_relationship(x, hue)
            
            if rel is not None and rel.type == "booleanfunction" and rel.output_type != "std":
                answer = "yes"
            else:
                answer = "no"
            prompts.append({
                "prompt": f"can the data be meaningfully clustered by {hue}?",
                "answer": answer,
                "prompt_type": "qualitative",
                "prompt_tier": "inference",
                "options": ["yes", "no"],
            })

    elif render_metadata["type"] == "histogram":
        if "hue" not in render_metadata:
            x = render_metadata["x"]
            # Get frequencies from histogram data
            total = table_data["count"].sum()
            probabilities = table_data["count"] / total

            if render_metadata["x_type"] == "numeric":
                # For numeric x, ask about probability above a threshold
                # Get xticks from render metadata
                xticks = render_metadata["plot_xticks"]
                # Choose random threshold from xticks, excluding the last one
                threshold = float(f"{np.random.choice(xticks[:-1]):.2f}")
                
                # Sum probabilities for all bins above threshold
                prob = float(table_data[table_data[x].apply(lambda x: float(x.split(',')[0].strip('('))).astype(float) >= threshold]["count"].sum() / total)
                
                prompts.append({
                    "prompt": f"what is the probability that a new observation will have {x} greater than {threshold:.2f}?",
                    "answer": prob,
                    "prompt_type": "quantitative",
                    "prompt_tier": "inference"
                })
            else:
                # For categorical x, keep original question
                random_x = np.random.choice(table_data[x])
                prob = float(probabilities[table_data[x] == random_x].iloc[0])
                
                prompts.append({
                    "prompt": f"what is the probability that a new observation will have {x}={random_x}?",
                    "answer": prob,
                    "prompt_type": "quantitative",
                    "prompt_tier": "inference"
                })
        else:
            x = render_metadata["x"]
            hue = render_metadata["hue"]
            
            # Get x value to ask about
            if render_metadata["x_type"] == "numeric":
                xticks = render_metadata["plot_xticks"]
                x_value = float(f"{np.random.choice(xticks):.2f}")
                # Find the bin containing this x value
                x_bin = table_data[x].iloc[np.argmin(np.abs(table_data[x].apply(lambda x: float(x.split(',')[0].strip('('))).astype(float) - x_value))]
                row = table_data[table_data[x] == x_bin]
            else:
                x_value = np.random.choice(table_data[x].unique())
                row = table_data[table_data[x] == x_value]
            
            # Get counts from hue columns
            hue_cols = [col for col in row.columns if col.startswith(f"{hue}=")]
            counts = {col.split('=')[-1]: row[col].iloc[0] for col in hue_cols}
            most_likely_hue = max(counts.items(), key=lambda x: x[1])[0]
            
            prompts.append({
                "prompt": f"for observations with {x}={x_value}, which category of {hue} are they most likely to belong to?",
                "answer": most_likely_hue,
                "prompt_type": "qualitative", 
                "prompt_tier": "inference",
                "options": list(counts.keys()),
            })
            
    elif render_metadata["type"] == "bar":
        pass

    elif render_metadata["type"] == "line":
        pass
            
    return prompts


def generate_prompts(dataset_name: str) -> List[Dict[str, Any]]:
    """Generate prompts for a dataset based on its renders and metadata"""
    # Load metadata and configurations
    dataset_metadata = load_dataset_metadata(dataset_name)
    generator_config = load_generator_config(dataset_metadata["generator_name"])
    render_metadata_list = load_render_metadata(dataset_name)
    
    all_prompts = []
    
    # Generate prompts for each render
    for render_metadata in render_metadata_list:
        # Load the table data
        plot_path = os.path.join(PROJECT_ROOT, f"data/plots/{render_metadata['render_id']}.png")
        table_path = os.path.join(PROJECT_ROOT, f"data/tables/{render_metadata['render_id']}.txt")
        if render_metadata["type"] == "scatter":
            table_data = pd.read_csv(table_path)
        else:
            table_data = pd.read_fwf(table_path)
            # Merge Unnamed:0 with hue column if it exists
            if "Unnamed: 0" in table_data.columns:
                x_col = render_metadata["x"]
                table_data[x_col] = table_data["Unnamed: 0"].astype(str) + table_data[x_col].astype(str)
                table_data = table_data.drop("Unnamed: 0", axis=1)

        # Generate prompts for each tier
        retrieval_prompts = generate_retrieval_prompts(render_metadata, table_data)
        arithmetic_prompts = generate_arithmetic_prompts(render_metadata, table_data)
        boolean_prompts = generate_boolean_prompts(render_metadata, table_data)
        inference_prompts = generate_inference_prompts(render_metadata, table_data, generator_config)
        
        # Combine prompts with metadata
        for prompt in retrieval_prompts + arithmetic_prompts + boolean_prompts + inference_prompts:
            # Convert NumPy types to native Python types
            if isinstance(prompt["answer"], (np.int64, np.float64)):
                prompt["answer"] = prompt["answer"].item()
            elif isinstance(prompt["answer"], np.ndarray):
                prompt["answer"] = prompt["answer"].tolist()
            
            if "options" in prompt:
                if isinstance(prompt["options"], (np.int64, np.float64)):
                    prompt["options"] = prompt["options"].item()
                elif isinstance(prompt["options"], np.ndarray):
                    prompt["options"] = prompt["options"].tolist()
            
            prompt.update({
                "dataset": dataset_name,
                "generator": dataset_metadata["generator_name"],
                "render_id": render_metadata["render_id"],
                "render_type": render_metadata["type"],
                "answer_type": str(type(prompt["answer"])),
                "renders": {"plot": plot_path, "table": table_path}
            })
            all_prompts.append(prompt)
            
    return all_prompts
