import yaml
from collections import OrderedDict
import re

import pandas as pd
import numpy as np
from scipy.optimize import fsolve


class DependencyTree:
    """
    A class representing dependencies between variables as a tree structure.
    
    The dependencies are specified by an adjacency matrix where entry (i,j) = 1 
    indicates that variable i depends on variable j. Diagonal entries are ignored.
    """
    def __init__(self, dependency_matrix, variable_names):
        """
        Initialize the dependency tree from an adjacency matrix.
        
        Parameters:
        dependency_matrix (numpy.ndarray): n x n binary matrix of dependencies
        variable_names (list): List of n variable names corresponding to matrix indices
        
        Raises:
        ValueError: If a cycle is detected in the dependencies
        """
        if len(variable_names) != len(dependency_matrix):
            raise ValueError("Number of variable names must match matrix dimensions")
            
        # Convert diagonal entries to 0
        self.matrix = np.array(dependency_matrix)
        np.fill_diagonal(self.matrix, 0)
            
        self.variables = variable_names
        self.n = len(variable_names)
        
        # Build adjacency lists representation
        self.children = {i: [] for i in range(self.n)}
        self.parents = {i: [] for i in range(self.n)}
        
        for i in range(self.n):
            for j in range(self.n):
                if self.matrix[i,j] == 1:
                    self.children[j].append(i)
                    self.parents[i].append(j)

        # Find roots (nodes with no parents)
        self.roots = [i for i in range(self.n) if not self.parents[i]]
                    
        # Check for cycles
        visited = set()
        temp = set()
        
        def has_cycle(node):
            if node in temp:
                return True
            if node in visited:
                return False
                
            temp.add(node)
            for child in self.children[node]:
                if has_cycle(child):
                    return True
            temp.remove(node)
            visited.add(node)
            return False
            
        for i in range(self.n):
            if i not in visited:
                if has_cycle(i):
                    raise ValueError("Cycle detected in dependency graph")
                    
    def get_tree(self):
        """
        Get the dependency tree as a list of levels and dependencies.
        
        Returns:
        tuple: (list, dict) where:
            - list contains lists of variable names at each level of the tree
            - dict maps variable names to tuples of (parent_names, parent_indices)
              where parent_names is a list of variable names this variable depends on
              and parent_indices is a list of indices in the level above, or (None, None)
              if the variable has no dependencies
        """
        # Find roots (nodes with no parents)
        roots = self.roots
        
        # Build levels through BFS
        levels = []
        visited = set()
        current_level = roots
        dependencies = {}
        
        # Add root nodes with no dependencies
        for root in roots:
            dependencies[self.variables[root]] = None
        
        while current_level:
            # Add current level to results
            level_vars = [self.variables[i] for i in current_level]
            levels.append(level_vars)
            visited.update(current_level)
            
            # Find all children of current level nodes that have all parents visited
            next_level = []
            for node in current_level:
                for child in self.children[node]:
                    if child not in visited and all(p in visited for p in self.parents[child]):
                        next_level.append(child)
                        # Get parent names and find their indices in all previous levels
                        parent_names = [self.variables[p] for p in self.parents[child]]
                        parent_indices = []
                        for p in self.parents[child]:
                            # Search through all previous levels to find the parent
                            for level_idx, level in enumerate(levels):
                                if self.variables[p] in level:
                                    parent_indices.append((level_idx, level.index(self.variables[p])))
                                    break
                        dependencies[self.variables[child]] = (parent_names, parent_indices)
            
            current_level = next_level
            
        return levels, dependencies
    
    def get_roots(self):
        """
        Get the roots of the dependency tree.
        """
        return self.roots

    def get_children(self, var):
        """
        Get the children of a variable.
        """
        return self.children[self.variables.index(var)]

    def get_parents(self, var):
        """
        Get the parents of a variable.
        """
        return self.parents[self.variables.index(var)]

    def is_upstream(self, upstream_var, downstream_var):
        """
        Check if one variable is upstream of another in the dependency tree.
        
        Parameters:
        upstream_var (str): Name of potential upstream variable
        downstream_var (str): Name of downstream variable
        
        Returns:
        bool: True if upstream_var is a parent or ancestor of downstream_var
        """
        # Convert variable names to indices
        try:
            upstream_idx = self.variables.index(upstream_var)
            downstream_idx = self.variables.index(downstream_var)
        except ValueError:
            raise ValueError("Both variables must exist in the dependency tree")
        
        # Helper function to recursively check parents
        def check_ancestors(current_idx, target_idx, visited=None):
            if visited is None:
                visited = set()
                
            if current_idx in visited:
                return False
                
            visited.add(current_idx)
            
            # Check direct parents
            for parent_idx in self.parents[current_idx]:
                if parent_idx == target_idx:
                    return True
                # Recursively check parents of parents
                if check_ancestors(parent_idx, target_idx, visited):
                    return True
                    
            return False
        
        return check_ancestors(downstream_idx, upstream_idx)

    def __str__(self):
        """
        Create a string representation of the dependency tree.
        """
        levels, dependencies = self.get_tree()  # Get both return values from get_tree()
        result = []
        
        # Track which variables we've seen to show dependencies
        seen_vars = {}
        for level_idx, level in enumerate(levels):
            # Store the level number for each variable
            for var in level:
                # Remove the print statement that was causing the error
                seen_vars[var] = level_idx
                
            # Rest of the method remains the same...
            level_str = f"Level {level_idx}: {', '.join(level)}"
            
            deps = []
            for var in level:
                var_idx = self.variables.index(var)
                parents = [self.variables[p] for p in self.parents[var_idx]]
                if parents:
                    deps.append(f"{var} ← {', '.join(parents)}")
            
            if deps:
                level_str += f"  (Dependencies: {'; '.join(deps)})"
            
            result.append(level_str)
            
        return "\n".join(result)


class DependencyMatrix:
    """
    A class representing relationships between variables in a matrix form.
    
    The relationships can be correlations (floats) or functional relationships
    (Polynomial or BooleanFunction instances).
    """
    def __init__(self, variable_names, relationships):
        """
        Initialize the dependency matrix from variable names and relationships.
        
        Parameters:
        variable_names (list): List of variable names in desired order
        relationships (dict): Nested dictionary describing relationships between variables
            First level keys are source variables
            Second level keys are target variables
            Values are either floats (correlations) or strings (functional relationships)
        """
        self.variables = variable_names
        
        # Initialize ordered dictionary for relationships
        self.map = OrderedDict()
        for var1 in self.variables:
            self.map[var1] = OrderedDict()
            for var2 in self.variables:
                self.map[var1][var2] = None
        
        # Fill in relationships from dictionary
        for source, targets in relationships.items():
            for target, relationship in targets.items():
                if isinstance(relationship, (int, float)):
                    self.map[source][target] = float(relationship)
                elif isinstance(relationship, str):
                    # Parse relationship string to create appropriate object
                    if relationship.startswith("Polynomial"):
                        self.map[source][target] = Polynomial.from_string(relationship)
                    elif relationship.startswith("BooleanFunction"):
                        self.map[source][target] = BooleanFunction.from_string(relationship)

                    elif relationship.startswith("Trigonometric"):
                        pattern = r"Trigonometric\(\[(.*?)\],\s*\[(.*?)\]\)"
                        match = re.match(pattern, relationship)
                        if not match:
                            raise ValueError(f"Invalid Trigonometric string format: {relationship}")
                        
                        # Parse the functions and coefficients
                        funcs_str = match.group(1)
                        coeffs_str = match.group(2)
                        
                        # Convert string lists to Python lists
                        funcs = [f.strip().strip("'\"") for f in funcs_str.split(',')]
                        coeffs = [float(c.strip()) for c in coeffs_str.split(',')]
                        self.map[source][target] = Trigonometric(funcs, coeffs)
                        
                    else:
                        raise ValueError(f"Unknown relationship type: {relationship}")
        
        # Create binary dependency matrix
        n = len(self.variables)
        dependency_matrix = np.zeros((n, n), dtype=int)
        for i, source in enumerate(self.variables):
            for j, target in enumerate(self.variables):
                if self.map[source][target] is not None:
                    dependency_matrix[j][i] = 1  # j depends on i
                    
        # Create dependency tree
        self.tree = DependencyTree(dependency_matrix, self.variables)

    def __str__(self):
        """
        Create a string representation of the dependency matrix showing relationships between variables.
        """
        # Get max height of multiline relationships
        max_height = 1
        cell_widths = {}  # Store width needed for each cell
        cell_contents = {}  # Store actual content for each cell
        
        # Calculate cell dimensions
        max_width = max(len(var) for var in self.variables)  # Minimum width matches longest variable name
        for source in self.variables:
            for target in self.variables:
                rel = self.map[source][target]
                if rel is not None:
                    # Format relationship string with source variable name
                    if isinstance(rel, (int, float)):
                        content = str(rel)
                    else:
                        content = str(rel).replace('x', source)
                    lines = content.split('\n')
                    max_height = max(max_height, len(lines))
                    width = max(len(line) for line in lines)
                    max_width = max(max_width, width)
                    cell_widths[(source,target)] = width
                    cell_contents[(source,target)] = lines
                else:
                    cell_widths[(source,target)] = max_width
                    cell_contents[(source,target)] = [" " * max_width] * max_height
        
        # Standardize all cells to max dimensions
        for source in self.variables:
            for target in self.variables:
                content = cell_contents[(source,target)]
                # Pad each line to max width
                content = [line.ljust(max_width) for line in content]
                # Pad to max height with blank lines
                while len(content) < max_height:
                    content.append(" " * max_width)
                cell_contents[(source,target)] = content
                cell_widths[(source,target)] = max_width
        
        # Get max variable name length for row labels
        max_var_len = max(len(var) for var in self.variables)
        
        # Create header row
        header = " " * (max_var_len + 2) + "|"  # Padding for row labels
        header += "|".join(f"{var:^{max_width}}" for var in self.variables) + "|"
        
        # Create separator line
        separator = "-" * (max_var_len + 2) + "+"
        separator += "+".join("-" * max_width for var in self.variables) + "+"
        
        # Create rows
        rows = []
        for source in self.variables:
            # For each line in the maximum cell height
            for h in range(max_height):
                if h == 0:
                    row = f"{source:<{max_var_len + 2}}|"  # Left align row label
                else:
                    row = " " * (max_var_len + 2) + "|"
                    
                for target in self.variables:
                    content = cell_contents[(source,target)]
                    row += f"{content[h]}|"
                rows.append(row)
            
            # Add separator after each variable's rows
            rows.append(separator)
            
        # Combine all parts
        return header + "\n" + separator + "\n" + "\n".join(rows[:-1])  # Remove last separator
        
    def __repr__(self):
        """
        Returns the same string representation as __str__.
        """
        return self.__str__()
    
    def depends_on(self, source, target):
        """
        Check if source variable depends on target variable.
        """
        return self.tree.is_upstream(target, source)
                        
    def get_relationship(self, source, target):
        """
        Get the relationship between two variables.
        
        Parameters:
        source (str): Name of source variable
        target (str): Name of target variable
        
        Returns:
        The relationship (float, Polynomial, or BooleanFunction) or None if no relationship exists
        """
        return self.map[source][target]

    def get_tree(self):
        """
        Get the dependency tree for the variables.
        """
        return self.tree.get_tree()

    def get_independent_variables(self):
        """
        Get the independent variables (roots of the dependency tree).
        """
        return self.tree.get_roots()

    def get_dependent_variables(self, var):
        """
        Get the dependent variables of a variable (var).
        """
        return self.tree.get_children(var)


class Trigonometric:
    """
    A class representing a trigonometric function.
    """
    def __init__(self, funcs, coeffs):
        self.funcs = funcs
        self.coeffs = coeffs
        self.type = "trigonometric"

        if len(funcs) != len(coeffs):
            raise ValueError("Number of functions must match number of coefficients")

    def __call__(self, x):
        if isinstance(x, (int, float, np.int32, np.int64, np.float32, np.float64)):
            result = 0
        else:
            if isinstance(x[0], np.str_):
                vals = sorted(np.unique(x))
                degs = np.linspace(0, 2*np.pi, len(vals))
                x = np.array([degs[vals.index(val)] for val in x]).astype(float)

            result = np.zeros_like(x)
        
        for func, coeff in zip(self.funcs, self.coeffs):
            # Extract the multiplier inside trig function (e.g., 2 from sin(2*x))
            mult_match = re.match(r"(sin|cos|tan)\((\d*\*)?x\)", func)
            if not mult_match:
                raise ValueError(f"Invalid function format: {func}")
            
            # Get the function type (sin/cos) and multiplier
            func_type = mult_match.group(1)
            mult_str = mult_match.group(2)
            multiplier = float(mult_str.rstrip('*')) if mult_str else 1.0

            # Apply the transformation
            if func_type == 'sin':
                result += coeff * np.sin(multiplier * x)
            else:  # cos
                result += coeff * np.cos(multiplier * x)
        
        return result

    @classmethod
    def from_string(cls, trig_str):
        """
        Initialize the trigonometric function from a string representation.
        """
        funcs, coeffs = cls._parse_trig_string(trig_str)
        return cls(funcs, coeffs)

    def _parse_trig_string(trig_str):
        """
        Parse a trigonometric string into its components.
        """
        pattern = r"Trigonometric\(\[(.*?)\],\s*\[(.*?)\]\)"
        match = re.match(pattern, relationship)
        if not match:
            raise ValueError(f"Invalid Trigonometric string format: {relationship}")
        
        # Parse the functions and coefficients
        funcs_str = match.group(1)
        coeffs_str = match.group(2)
        
        # Convert string lists to Python lists
        funcs = [f.strip().strip("'\"") for f in funcs_str.split(',')]
        coeffs = [float(c.strip()) for c in coeffs_str.split(',')]
        return funcs, coeffs

    def __str__(self):
        """
        Create a readable string representation of the trigonometric function.
        
        Examples:
        funcs=['sin(x)', 'cos(2*x)'], coeffs=[1, 2] -> "f(x)=sin(x)+2cos(2*x)"
        funcs=['sin(3*x)', 'cos(x)'], coeffs=[-1, 0.5] -> "f(x)=-sin(3*x)+0.5cos(x)"
        """
        terms = []
        
        for func, coeff in zip(self.funcs, self.coeffs):
            if coeff == 0:
                continue
                
            # Add plus sign if positive and not first term
            if coeff > 0 and terms:
                terms.append("+")
                
            # Handle coefficient of 1 or -1
            if abs(coeff) == 1:
                terms.append("-" if coeff < 0 else "")
            else:
                terms.append(str(coeff))
                
            terms.append(func)
        
        # Handle special cases
        if not terms:
            return "f(x)=0"
        return "f(x)=" + "".join(terms)

    def __repr__(self):
        """
        Create a string representation that could be used to recreate the object.
        """
        return f"Trigonometric({self.funcs}, {self.coeffs})"


class Polynomial:
    """
    A class representing a polynomial function.
    
    The polynomial is represented by its coefficients in ascending order of degree.
    For example, coefficients [1, 2, 3] represents the polynomial 1 + 2x + 3x^2.
    """
    def __init__(self, coefficients):
        """
        Initialize the polynomial with its coefficients.
        
        Parameters:
        coefficients (list): List of coefficients in ascending order of degree
        """
        self.coefficients = coefficients
        # Degree is one less than length of coefficients
        self.degree = len(coefficients) - 1
        self.type = "polynomial"

    @classmethod
    def from_string(cls, poly_str):
        """
        Initialize the polynomial from a string representation.
        """
        coeffs = cls._parse_polynomial_string(poly_str)
        return cls(coeffs)

    def _parse_polynomial_string(poly_str):
        """
        Convert a polynomial string into a list of coefficients.
        
        Parameters:
        poly_str (str): String representation of polynomial (e.g. "2x^2-1" or "1 + 4x + x^2" or "Polynomial([1,2,3])")
        
        Returns:
        list: Coefficients in ascending order of degree
        """
        if "Polynomial" in poly_str or "polynomial" in poly_str:
            # Extract coefficients from string like "Polynomial([1,2,3])"
            try:
                coeffs = eval(poly_str.split("[")[1].split("]")[0])
            except IndexError:
                poly_str = poly_str.replace("((", "([").replace("))", "])")
                coeffs = eval(poly_str.split("[")[1].split("]")[0])
            return coeffs
        
        # Remove spaces and convert minuses to plus-minus for easier splitting
        poly_str = poly_str.replace(" ", "").replace("-", "+-")
        if poly_str.startswith("+"): 
            poly_str = poly_str[1:]
            
        # Split into terms
        terms = poly_str.split("+")
        
        # Dictionary to store coefficient for each degree
        coeffs_dict = {}
        
        for term in terms:
            if not term: # Skip empty terms from double signs
                continue
                
            # Handle special cases
            if term == "x":
                coeffs_dict[1] = 1
                continue
            elif term == "-x":
                coeffs_dict[1] = -1
                continue
            elif "x" not in term:
                coeffs_dict[0] = float(term)
                continue
                
            # Parse terms with x
            parts = term.split("x^" if "^" in term else "x")
            
            # Get coefficient
            coeff = float(parts[0]) if parts[0] and parts[0] != "-" else (-1 if parts[0] == "-" else 1)
            
            # Get degree
            degree = int(parts[1]) if "^" in term else 1
            coeffs_dict[degree] = coeff
        
        # Convert to list, filling in zeros for missing degrees
        max_degree = max(coeffs_dict.keys()) if coeffs_dict else 0
        coeffs = [coeffs_dict.get(i, 0) for i in range(max_degree + 1)]
        
        return coeffs
        
    def __call__(self, x):
        """
        Evaluate the polynomial at point x.
        
        Parameters:
        x (float or array-like): Point(s) at which to evaluate the polynomial
        
        Returns:
        float or array-like: Value of polynomial at x with same dtype as input
        """
        if isinstance(x, (int, float, np.int32, np.int64, np.float32, np.float64)):
            result = sum(coef * (x ** i) for i, coef in enumerate(self.coefficients))
            return type(x)(result)
        else:
            return np.array([self(xi) for xi in x], dtype=x.dtype)

    def degree(self, return_type="int"):
        """
        Get the degree of the polynomial.

        Parameters:
        return_type (str): "int" or "str"
        """
        if return_type == "int":
            return self.degree
        degree_strs = {0: "constant", 1: "linear", 2: "quadratic", 3: "cubic", 4: "quartic", 5: "quintic"}
        return degree_strs[self.degree] if self.degree in degree_strs else f"{self.degree}th degree"
            
    def derivative(self):
        """
        Compute the derivative polynomial.
        
        Returns:
        Polynomial: The derivative polynomial
        """
        if self.degree == 0:
            return Polynomial([0])
        new_coeffs = [i * coef for i, coef in enumerate(self.coefficients)][1:]
        return Polynomial(new_coeffs)
        
    def is_increasing(self, interval):
        """
        Determine where the polynomial is increasing, decreasing, or at inflection points.
        
        Parameters:
        interval (tuple): (start, end) points of interval to analyze
        
        Returns:
        dict: Maps 'increasing', 'decreasing', and 'inflection' to lists of intervals
        """
        start, end = interval
        derivative = self.derivative()
        second_derivative = derivative.derivative()
        
        # Find critical points (where derivative is zero)
        def der_func(x):
            return derivative(x)
        
        # Get rough estimates of critical points
        x_vals = np.linspace(start, end, 100)
        y_vals = derivative(x_vals)
        sign_changes = np.where(np.diff(np.signbit(y_vals)))[0]
        critical_points = []
        
        for idx in sign_changes:
            root = fsolve(der_func, x_vals[idx])[0]
            if start <= root <= end:
                critical_points.append(root)
                
        # Add endpoints
        all_points = sorted([start] + critical_points + [end])
        
        result = {
            "increasing": [],
            "decreasing": [],
            "inflection": []
        }
        
        # Check each subinterval
        for i in range(len(all_points)-1):
            mid = (all_points[i] + all_points[i+1]) / 2
            der_val = derivative(mid)
            if abs(der_val) < 1e-10:  # At inflection point
                result["inflection"].append((all_points[i], all_points[i+1]))
            elif der_val > 0:  # Increasing
                result["increasing"].append((all_points[i], all_points[i+1]))
            else:  # Decreasing
                result["decreasing"].append((all_points[i], all_points[i+1]))
                
        return result
        
    def __min__(self):
        """
        Find local minima of the polynomial.
        
        Returns:
        list: x-coordinates of local minima
        """
        derivative = self.derivative()
        second_derivative = derivative.derivative()
        
        def der_func(x):
            return derivative(x)
            
        # Get rough estimates
        x_vals = np.linspace(-100, 100, 1000)
        y_vals = derivative(x_vals)
        sign_changes = np.where(np.diff(np.signbit(y_vals)))[0]
        
        minima = []
        for idx in sign_changes:
            root = fsolve(der_func, x_vals[idx])[0]
            if second_derivative(root) > 0:  # Check if it's a minimum
                minima.append(root)
                
        return minima
        
    def __max__(self):
        """
        Find local maxima of the polynomial.
        
        Returns:
        list: x-coordinates of local maxima
        """
        derivative = self.derivative()
        second_derivative = derivative.derivative()
        
        def der_func(x):
            return derivative(x)
            
        # Get rough estimates
        x_vals = np.linspace(-100, 100, 1000)
        y_vals = derivative(x_vals)
        sign_changes = np.where(np.diff(np.signbit(y_vals)))[0]
        
        maxima = []
        for idx in sign_changes:
            root = fsolve(der_func, x_vals[idx])[0]
            if second_derivative(root) < 0:  # Check if it's a maximum
                maxima.append(root)
                
        return maxima
    
    def __str__(self):
        """
        Create a readable string representation of the polynomial.
        
        Examples:
        [1, 2, 3] -> "1 + 2x + 3x²"
        [-1, 0, 2] -> "-1 + 2x²"
        [0, -2, 1] -> "-2x + x²"
        """
        terms = []
        
        for i, coef in enumerate(self.coefficients):
            if coef == 0:
                continue
                
            # Handle the coefficient
            if i == 0:  # Constant term
                terms.append(str(coef))
            else:
                # Add plus sign if positive and not first term
                if coef > 0 and terms:
                    terms.append("+")
                
                # Handle coefficient of 1 or -1
                if abs(coef) == 1 and i > 0:
                    terms.append("-" if coef < 0 else "")
                else:
                    terms.append(str(coef))
                
                # Add the variable and exponent
                if i == 1:
                    terms.append("x")
                else:
                    terms.append(f"x²" if i == 2 else f"x^{i}")
        
        # Handle special cases
        if not terms:
            return "0"
        return "f(x)=" + "".join(terms)

    def __repr__(self):
        """
        Create a string representation that could be used to recreate the object.
        """
        return f"Polynomial({self.coefficients})"


class BooleanFunction:
    """
    A class representing a series of boolean expressions and their corresponding outputs.
    When called, evaluates the expressions in order and returns the output corresponding
    to the first True expression, or a default value if no expressions are True.
    
    Parameters:
    conditions (list): List of boolean expressions to evaluate
    outputs (list): List of values to return, where the last value is returned if no conditions are True.
                   Length should be one more than the number of conditions.
    output_type (str): Which parameter of input x to return ("value" for the value; "mean", "std" e.g. for Distribution parameters)
    context_vars (list): Context for the boolean function, e.g. the variable names of the parent variables
    """
    def __init__(self, conditions, outputs, output_type="value", context_vars=["x"]):
        if len(conditions) + 1 != len(outputs) and len(conditions) != len(outputs):
            raise ValueError("Number of outputs must be equal to or one more than number of conditions")
        self.conditions = conditions
        self.outputs = outputs
        self.output_type = output_type
        self.context_vars = context_vars
        self.type = "booleanfunction"

    @classmethod
    def from_string(cls, bool_str):
        """
        Initialize the BooleanFunction from a string representation.
        """
        conditions, outputs, output_type, context_vars = cls._parse_boolean_function_string(bool_str)
        return cls(conditions, outputs, output_type=output_type, context_vars=context_vars)

    def _parse_boolean_function_string(bool_str):
        """
        Parse a BooleanFunction string into its components.
        
        Parameters:
        bool_str (str): String like "BooleanFunction(['cond1', 'cond2'], [val1, val2, val3])" 
                       or "BooleanFunction(['cond1', 'cond2'], [val1, val2, val3], 'output_type')"
        
        Returns:
        tuple: (conditions, outputs, output_type, context_vars)
        """
        # Match the pattern for conditions and outputs
        #pattern = r"BooleanFunction\(\[(.*?)\],\s*\[(.*?)\](?:,\s*'(\w+)')?\)"
        pattern = r"BooleanFunction\(\[(.*?)\],\s*\[((?:\([^)]+\)|[^,\s]+)(?:\s*,\s*(?:\([^)]+\)|[^,\s]+))*)\](?:,\s*'([^']*(?:,[^']*)?)')?\)"
        match = re.match(pattern, bool_str)
        
        if not match:
            raise ValueError(f"Invalid BooleanFunction string format: {bool_str}")
        
        # Extract conditions and outputs
        conditions_str = match.group(1)
        outputs_str = match.group(2)
        output_type = match.group(3) or "value"  # Default to "value" if not specified

        # Parse conditions - handle both quoted and unquoted strings
        conditions = []
        context_vars = set()  # Track unique variable names
        for cond in re.findall(r"'([^']*)'|\"([^\"]*)\"|([^,\s][^,]*[^,\s])", conditions_str):
            # Take the non-empty group from the alternatives
            cond_str = next(c for c in cond if c)
            conditions.append(cond_str)
            
            # Extract variable names from condition
            # Split on logical operators
            for part in re.split(r'\s+(?:and|or|not)\s+', cond_str):
                part = part.strip()
                if '==' in part:
                    var = part.split('==')[0].strip()
                elif 'in' in part:
                    var = part.split('in')[0].strip().replace('not', '')
                elif '<' in part or '>' in part:
                    var = part.split('<')[0].strip() if '<' in part else part.split('>')[0].strip()
                if "(" in var:
                    var = var.split("(")[1].replace(")", "")
                context_vars.add(var)
        context_vars = sorted(list(context_vars))
        
        # Parse outputs - handle numbers, strings, and tuples
        outputs = []
        # First check if the entire string represents a list of tuples
        if output_type == "dependency":
            # Split on commas that are not inside parentheses
            output_strs = re.split(r',\s*(?![^()]*\))', outputs_str)
            outputs = [Polynomial.from_string(o.strip()) for o in output_strs]
        else:
            tuple_pattern = r'\(([^)]+)\)'
            tuple_matches = re.findall(tuple_pattern, outputs_str)
        
            if tuple_matches:
                # Handle list of tuples
                for match in tuple_matches:
                    tuple_values = []
                    for val in match.split(','):
                        val = val.strip()
                        if val.isdigit():
                            tuple_values.append(int(val))
                        elif val.replace(".", "", 1).isdigit():
                            tuple_values.append(float(val))
                        else:
                            tuple_values.append(val)
                    outputs.append(tuple(tuple_values))
            else:
                # Handle regular comma-separated values
                for o in outputs_str.split(","):
                    o = o.strip().replace(" ", "")
                    if o.isdigit():
                        outputs.append(int(o))
                    elif o.replace(".", "", 1).isdigit():
                        outputs.append(float(o))
                    else:
                        outputs.append(o)
        
        return conditions, outputs, output_type, context_vars
        
    def __call__(self, x):
        """
        Evaluates the boolean expressions for the given input and returns the corresponding output.
        
        Parameters:
        x (dict): Input value(s) to evaluate conditions against. Must contain all context variables.
        
        Returns:
        The output corresponding to the first True condition for each input, or the last output if no conditions are True
        """
        assert all(var in x for var in self.context_vars), f"Input must contain all context variables: {self.context_vars}"

        # Convert all inputs to numpy arrays
        for var in self.context_vars:
            x[var] = np.asarray(x[var])
        
        # Get the length of the input arrays
        n = len(next(iter(x.values())))
        
        # Initialize with default value, ensuring proper shape
        if isinstance(self.outputs[-1], (list, np.ndarray, tuple)):
            result = np.tile(self.outputs[-1], (n, 1))
        else:
            result = np.full(n, self.outputs[-1])
        
        # Evaluate each condition and update result where condition is True
        for i, condition in enumerate(self.conditions):
            # Create evaluation environment with all context variables
            def eval_condition(idx):
                env = {var: x[var][idx] for var in self.context_vars}
                try:
                    return eval(condition, env)
                except NameError:
                    # Handle string literals in conditions
                    modified_condition = condition
                    
                    # Split on logical operators first
                    logical_parts = re.split(r'\s+and\s+|\s+or\s+|\s+not\s+|\s+in\s+|\s+not\s+in\s+', condition)
                    
                    for part in logical_parts:
                        # Find operator and split around it
                        match = re.search(r'(==|<|>)', part)
                        if match:
                            operator = match.group()
                            left, right = part.split(operator)
                            left = left.strip()
                            right = right.strip()
                            
                            # Only modify right side if it's not a context variable
                            if not any(var in right for var in self.context_vars):
                                modified_condition = modified_condition.replace(right, f"'{right}'")
                    
                    return eval(modified_condition, env)
            
            # Create mask of True conditions
            mask = np.array([eval_condition(idx) for idx in range(n)])
            
            # Update result where condition is True
            result[mask] = self.outputs[i]
        
        return result

    def __str__(self):
        """
        Create a readable string representation of the boolean function.
        
        Examples:
        conditions: ['x>0', 'x<5'], outputs: [1, 2, 0] ->
            "if x>0: return 1
             elif x<5: return 2
             else: return 0"
             
        conditions: ['Group==G1 and x==A', 'Group==G1 and x==B'], outputs: [2, 3, 1] ->
            "if Group==G1 and x==A: return 2
             elif Group==G1 and x==B: return 3
             else: return 1"
        """
        lines = []
        
        # Handle first condition with 'if'
        if self.conditions:
            lines.append(f"if {self.conditions[0]}: return {self.outputs[0]}")
            
            # Handle remaining conditions with 'elif'
            for condition, output in zip(self.conditions[1:], self.outputs[1:-1]):
                lines.append(f"elif {condition}: return {output}")
        
        # Add else clause with default output
        lines.append(f"else: return {self.outputs[-1]}")
        
        return f"\n".join(lines)

    def __repr__(self):
        """
        Create a string representation that could be used to recreate the object.
        """
        return (f"BooleanFunction({self.conditions}, {self.outputs}, "
                f"output_type='{self.output_type}', context_vars={self.context_vars})")

    def extra_context_vars(self):
        """
        Get any additional context variables required for the boolean function.
        """
        return [var for var in self.context_vars if var != "x"]