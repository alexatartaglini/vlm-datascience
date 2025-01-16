import os
import yaml
from pathlib import Path

from PIL import Image
import pandas as pd
from typing import List, Union

from .generate_prompts import generate_prompts


PROJECT_ROOT = Path(__file__).parent.parent


def _load_prompts(dataset_name, clear_existing=False):
    """
    Load prompts for a dataset from yaml file, generating them if they don't exist.
    
    Args:
        dataset_name (str): Name of the dataset to load/generate prompts for
        clear_existing (bool): Whether to regenerate prompts even if they exist
        
    Returns:
        list: List of prompt dictionaries
    """
    
    prompt_path = os.path.join(PROJECT_ROOT, "data/prompts", f"{dataset_name}_prompts.yaml")
    
    if os.path.exists(prompt_path) and not clear_existing:
        with open(prompt_path, 'r') as f:
            prompts = yaml.safe_load(f)
    else:
        prompts = generate_prompts(dataset_name)
        os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
        with open(prompt_path, 'w') as f:
            yaml.dump(prompts, f)
            
    return prompts


def load_prompts(dataset_names: List[str]=None, clear_existing: bool=False):
    """Load all prompts for all datasets"""
    if dataset_names is None:
        dataset_names = [f.stem for f in Path(PROJECT_ROOT / "data/prompts").glob("*.yaml")]
    return {dataset_name: _load_prompts(dataset_name, clear_existing) for dataset_name in dataset_names}


def _load_renders(prompts, load_as_paths: bool=False, all=False):
    """Load renders for a dataset from yaml file"""
    for dataset_name, dataset_prompts in prompts.items():
        for prompt in dataset_prompts:
            prompt["renders"]["plot"] = Image.open(prompt["renders"]["plot"]).convert("RGB") if not load_as_paths else prompt["renders"]["plot"]
            prompt["renders"]["table"] = pd.read_csv(prompt["renders"]["table"]) if prompt["render_type"] == "scatter" else pd.read_fwf(prompt["renders"]["table"])


def load_prompts_and_renders(dataset_names: Union[List[str], str]=None, clear_existing: bool=False, load_renders_as_paths: bool=False):
    """Load all prompts and renders for all datasets"""
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    prompts = load_prompts(dataset_names, clear_existing)
    _load_renders(prompts, load_as_paths=load_renders_as_paths, all="all" in dataset_names)
    return prompts