import yaml
import pandas as pd
import os
from math import sin, cos, tan, pi

from .utils import (
    DependencyMatrix,
    UniformDistribution,
    NormalDistribution,
    MultivariateNormalDistribution,
    ExponentialDistribution,
    PoissonDistribution,
    CategoricalDistribution,
    BetaDistribution,
    BooleanFunction,
)


class Data(pd.DataFrame):
    """
    A class for holding generated data.

    Inherits from pandas.DataFrame and adds additional methods for rendering.
    """
    @property
    def _constructor(self):
        return Data

    def __init__(
        self, 
        data=None, 
        index=None, 
        columns=None, 
        dtype=None, 
        copy=None, 
        numeric_vars=None, 
        categorical_vars=None, 
        sequential_vars=None,
        generator_name=None,
        name=None,
        **kwargs,
    ):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)

        # Store metadata
        object.__setattr__(self, '_props', {
            'generator_name': generator_name,
            'numeric_vars': numeric_vars or [],
            'categorical_vars': categorical_vars or [],
            'sequential_vars': sequential_vars or [],
            'name': name,
            'variables': list(self.columns),
            'n_points': self.shape[0],
            'n_variables': self.shape[1],
        })
        
        # Set computed properties
        self._props['n_numeric'] = len(self._props['numeric_vars']) if self._props['numeric_vars'] else 0
        self._props['n_categorical'] = len(self._props['categorical_vars']) if self._props['categorical_vars'] else 0
        self._props['n_sequential'] = len(self._props['sequential_vars']) if self._props['sequential_vars'] else 0

    @classmethod
    def from_csv(cls, filepath, **kwargs):
        return cls(
            data=pd.read_csv(filepath, **kwargs),
        )

    @classmethod
    def from_yaml(cls, filepath, **kwargs):
        with open(filepath, 'r') as f:
            metadata = yaml.safe_load(f)
        return cls(
            data=pd.read_csv(f"data/spreadsheets/{metadata['name']}.csv", **kwargs),
            **metadata,
        )

    @classmethod
    def load(cls, filepath):
        return cls.from_yaml(f"data/metadata/datasets/{filepath}.yaml")

    # Properties to access the metadata
    @property
    def generator_name(self):
        return self._props['generator_name']

    @property
    def variables(self):
        return self._props['variables']

    @property
    def n_variables(self):
        return self._props['n_variables']
    
    @property
    def numeric_vars(self):
        return self._props['numeric_vars']
    
    @property
    def categorical_vars(self):
        return self._props['categorical_vars']

    @property
    def sequential_vars(self):
        return self._props['sequential_vars']
    
    @property
    def name(self):
        return self._props['name']
    
    @property
    def n_numeric(self):
        return self._props['n_numeric']
    
    @property
    def n_categorical(self):
        return self._props['n_categorical']

    @property
    def n_sequential(self):
        return self._props['n_sequential']

    def set_numeric_vars(self, numeric_vars):
        self._props['numeric_vars'] = numeric_vars
        self._props['n_numeric'] = len(numeric_vars)

    def set_categorical_vars(self, categorical_vars):
        self._props['categorical_vars'] = categorical_vars
        self._props['n_categorical'] = len(categorical_vars)

    def set_sequential_vars(self, sequential_vars):
        self._props['sequential_vars'] = sequential_vars
        self._props['n_sequential'] = len(sequential_vars)

    def is_numeric(self, var):
        return var in self.numeric_vars

    def is_categorical(self, var):
        return var in self.categorical_vars

    def is_sequential(self, var):
        return var in self.sequential_vars

    def to_yaml(self, filepath):
        metadata = {
            'generator_name': self.generator_name,
            'name': filepath.split('/')[-1].split('.')[0] if self.name is None else self.name,
            'num_points': self.shape[0],
            'variables': self.variables,
            'numeric_vars': self.numeric_vars,
            'categorical_vars': self.categorical_vars,
            'sequential_vars': self.sequential_vars,
            'n_numeric': self.n_numeric,
            'n_categorical': self.n_categorical,
            'n_sequential': self.n_sequential,
        }
        with open(filepath, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)

    def save(self, filepath):
        assert not os.path.exists(f"data/spreadsheets/{filepath}.csv"), \
            f"File already exists: data/spreadsheets/{filepath}.csv"
        assert not os.path.exists(f"data/metadata/datasets/{filepath}.yaml"), \
            f"File already exists: data/metadata/datasets/{filepath}.yaml"
        
        for var in self.categorical_vars:
            self[var] = self[var].str.replace("'", "")

        self.to_csv(f"data/spreadsheets/{filepath}.csv", index=False)
        self.to_yaml(f"data/metadata/datasets/{filepath}.yaml")


class Generator:
    """
    A class for generating random data points according to specified distributions and dependencies.
    """
    def __init__(self, config, name=None):
        """
        Initialize generator from config dictionary or yaml file path.
        
        Parameters:
        config (dict or str): Dictionary or path to yaml file containing:
            - variables: List of variable names
            - variable_objects: Dict mapping variable names to distribution descriptions
            - dependencies: Dict describing relationships between variables
        """
        # Load config from yaml if string path provided
        if isinstance(config, str):
            self.name = config.split('/')[-1].split('.')[0]
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            self.name = name
                
        # Validate config has required keys
        required_keys = ['variables', 'variable_objects', 'dependencies']
        if not all(key in config for key in required_keys):
            raise ValueError(f"Config must contain keys: {required_keys}")
            
        self.variables = config['variables']
        
        # Validate all variables have distribution objects
        missing_vars = set(self.variables) - set(config['variable_objects'].keys())
        if missing_vars:
            raise ValueError(f"Missing distribution objects for variables: {missing_vars}")
            
        # Parse distribution strings into objects; store numeric and categorical variables separately
        self.distributions = {}
        self.numeric_vars = []
        self.categorical_vars = []
        self.sequential_vars = []
        for var, dist_str in config['variable_objects'].items():
            self.distributions[var] = self._parse_distribution(dist_str)
            if self.distributions[var].type == "numeric":
                self.numeric_vars.append(var)
            elif self.distributions[var].type == "categorical":
                self.categorical_vars.append(var)

                if self.distributions[var].is_sequential():
                    self.sequential_vars.append(var)
        self.n_numeric = len(self.numeric_vars)
        self.n_categorical = len(self.categorical_vars)
        self.n_sequential = len(self.sequential_vars)
            
        # Create dependency matrix
        self.dependency_matrix = DependencyMatrix(self.variables, config['dependencies'])
        self.dependency_tree = self.dependency_matrix.get_tree()
        
    def _parse_distribution(self, dist_str):
        """
        Parse distribution string into Distribution object.
        
        Parameters:
        dist_str (str): String describing distribution (e.g. "normal(mean=0, std=1)")
        
        Returns:
        Distribution object
        """
        dist_name = dist_str.split('(')[0].lower()
        # Extract parameters between parentheses and evaluate
        params_str = dist_str[dist_str.find('(')+1:dist_str.rfind(')')]
        params = eval(f"dict({params_str})")
        
        dist_map = {
            'uniform': UniformDistribution,
            'normal': NormalDistribution,
            'multivariate_normal': MultivariateNormalDistribution,
            'exponential': ExponentialDistribution,
            'poisson': PoissonDistribution,
            'categorical': CategoricalDistribution,
            'beta': BetaDistribution
        }
        
        if dist_name not in dist_map:
            raise ValueError(f"Unknown distribution: {dist_name}")
        
        return dist_map[dist_name](**params)
        
    def __call__(self, n):
        """
        Generate n random points according to specified distributions and dependencies.
        
        Parameters:
        n (int): Number of points to generate
        
        Returns:
        pandas.DataFrame: Generated points with columns labeled by variable names
        """
        # Get dependency ordering from tree
        levels, _ = self.dependency_matrix.get_tree()
            
        # Generate data level by level
        data = {}
        for level in levels:
            for var in level:
                # Get parent values if they exist
                var_idx = self.variables.index(var)
                parent_indices = self.dependency_matrix.tree.parents[var_idx]
                parent_vars = [self.variables[i] for i in parent_indices]
                
                if not parent_vars:
                    # No parents - generate from base distribution
                    data[var] = self.distributions[var](n)
                else:
                    # Get relationship function
                    rel_func = None
                    for parent in parent_vars:
                        rel = self.dependency_matrix.get_relationship(parent, var)
                        if rel is not None:
                            rel_func = rel
                            break
                            
                    if rel_func is None:
                        raise ValueError(f"No relastionship found for {var}")
                        
                    # Apply relationship to parent values
                    parent_vals = data[parent_vars[0]]  # Currently assumes single parent

                    if isinstance(rel_func, BooleanFunction):
                        # Gather context variables for boolean function
                        if len(rel_func.context_vars) == 1 and rel_func.context_vars[0] == "x":
                            func_input = {"x": parent_vals}
                        else:
                            if "x" in rel_func.context_vars:
                                func_input = {var: data[var] for var in rel_func.extra_context_vars()}
                                func_input["x"] = parent_vals
                            else:
                                func_input = {var: data[var] for var in rel_func.context_vars}
                        
                        # Apply boolean function
                        if rel_func.output_type == "value":
                            data[var] = rel_func(func_input).astype(self.distributions[var].dtype)
                        else:
                            param = rel_func(func_input)  # Fix the logic for array parameters, e.g. yaml 8
                            data[var] = self.distributions[var](n, **{rel_func.output_type: param})
                    else:
                        data[var] = rel_func(parent_vals).astype(self.distributions[var].dtype)
                    
                    if not isinstance(rel_func, BooleanFunction) or rel_func.output_type == "value":
                        # Add noise from distribution if Polynomial/Trigonomic or a BooleanFunction over values
                        noise = self.distributions[var](n)
                        data[var] += noise.astype(data[var].dtype)
        
        data = Data(data, generator_name=self.name, name=f"dataset_{self.name.split('_')[-1]}")
        data.set_numeric_vars(self.numeric_vars)
        data.set_categorical_vars(self.categorical_vars)
        data.set_sequential_vars(self.sequential_vars)
        
        return data
