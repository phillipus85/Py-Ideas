from dataclasses import dataclass, field
from typing import Callable, List, Dict     # , Any, Optional, Union, Tuple
import numpy as np
import random


@dataclass
class DependentMonteCarloSimulation:
    """
    Monte Carlo simulation with support for dependent variables where the output
    of one distribution can be used as input to another.
    """
    iterations: int                               # Number of simulation runs
    model_function: Callable[..., float]          # Function to simulate
    # Dictionary mapping variable names to their distribution functions
    variable_distributions: Dict[str, Callable[..., float]]
    # Dictionary specifying dependencies between variables
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    # Store intermediate variable values for each iteration
    intermediate_values: Dict[str, List[float]] = field(default_factory=dict)
    # Store final simulation results
    results: List[float] = field(default_factory=list)

    def __post_init__(self):
        """Initialize the intermediate values storage for all variables"""
        for var_name in self.variable_distributions.keys():
            self.intermediate_values[var_name] = []

    def _generate_input(self, var_name: str, iteration_values: Dict[str, float]) -> float:
        """
        Generate input for a specific variable, considering its dependencies.

        Args:
            var_name: The name of the variable to generate.
            iteration_values: Dictionary of values already generated in this iteration.

        Returns:
            Generated value for the variable.
        """
        # Check if this variable depends on others
        if var_name in self.dependencies:
            # Get the dependent variable values for this iteration
            dep_values = [iteration_values[dep] for dep in self.dependencies[var_name]]
            # Generate value using these dependencies
            return self.variable_distributions[var_name](*dep_values)
        else:
            # Independent variable with no dependencies
            return self.variable_distributions[var_name]()

    def run(self):
        """Run the Monte Carlo simulation for the specified number of iterations"""
        self.results.clear()
        # Clear previous intermediate values
        for var_name in self.intermediate_values:
            self.intermediate_values[var_name].clear()

        # For each iteration
        for _ in range(self.iterations):
            # Dictionary to store values for this iteration
            iteration_values = {}

            # Process variables in a way that respects dependencies
            processed_vars = set()
            remaining_vars = set(self.variable_distributions.keys())

            # Continue until all variables are processed
            while remaining_vars:
                # Find variables whose dependencies are all processed
                for var in list(remaining_vars):
                    deps = self.dependencies.get(var, [])
                    # If all dependencies have been processed or no dependencies
                    if all(dep in processed_vars for dep in deps):
                        # Generate value for this variable
                        value = self._generate_input(var, iteration_values)
                        iteration_values[var] = value
                        self.intermediate_values[var].append(value)

                        # Mark as processed
                        processed_vars.add(var)
                        remaining_vars.remove(var)

            # Calculate model output using all variable values
            input_values = [iteration_values[var] for var in self.variable_distributions.keys()]
            result = self.model_function(*input_values)
            self.results.append(result)

    def mean(self) -> float:
        """Calculate the mean of the simulation results"""
        return np.mean(self.results)

    def variance(self) -> float:
        """Calculate the variance of the simulation results"""
        return np.var(self.results)

    def get_variable_stats(self, var_name: str) -> dict:
        """
        Get statistics for a specific intermediate variable.

        Args:
            var_name: The name of the variable.

        Returns:
            Dictionary of statistics for the variable.
        """
        if var_name not in self.intermediate_values:
            raise ValueError(f"Variable '{var_name}' not found in simulation")

        values = self.intermediate_values[var_name]
        return {
            "mean": np.mean(values),
            "variance": np.var(values),
            "min": min(values),
            "max": max(values),
            "std": np.std(values)
        }

    def summary(self) -> dict:
        """Get a summary of the simulation results"""
        return {
            "mean": self.mean(),
            "variance": self.variance(),
            "min": min(self.results),
            "max": max(self.results),
            "std": np.std(self.results)
        }

    def full_summary(self) -> dict:
        """Get a comprehensive summary including all variables and results"""
        summary = {
            "results": self.summary(),
            "variables": {}
        }

        for var_name in self.variable_distributions.keys():
            summary["variables"][var_name] = self.get_variable_stats(var_name)

        return summary


# Example usage
if __name__ == "__main__":
    # Example 1: Independent variables
    print("Example 1: Independent variables")

    def model_indep(x, y):
        return x * y + 2

    # Define independent distributions
    def uniform_dist():
        return random.uniform(0, 1)

    def normal_dist():
        return random.gauss(5, 1)

    sim_indep = DependentMonteCarloSimulation(
        iterations=1000,
        model_function=model_indep,
        variable_distributions={
            "x": uniform_dist,
            "y": normal_dist
        }
    )

    sim_indep.run()
    print("Results:", sim_indep.summary())
    print()

    # Example 2: Dependent variables
    print("Example 2: Dependent variables")

    def model_dep(x, y):
        return x * y + 2

    # Define a distribution that depends on another variable
    def dependent_normal_dist(x_value):
        # Mean depends on x_value
        mean = 5 + x_value * 2
        return random.gauss(mean, 1)

    sim_dep = DependentMonteCarloSimulation(
        iterations=1000,
        model_function=model_dep,
        variable_distributions={
            "x": uniform_dist,
            "y": dependent_normal_dist
        },
        dependencies={
            "y": ["x"]  # y depends on x
        }
    )

    sim_dep.run()
    print("Results:", sim_dep.summary())
    print("Full summary:", sim_dep.full_summary())

    # Example 3: Multiple dependencies
    print("\nExample 3: Multiple dependent variables")

    def complex_model(x, y, z):
        return x * y * z + 2

    def dependent_uniform(x_value):
        # Range depends on x_value
        return random.uniform(0, x_value * 10)

    def double_dependent_dist(x_value, y_value):
        # Distribution parameters depend on both x and y
        mean = x_value * 5 + y_value / 2
        std_dev = max(0.5, y_value / 10)
        return random.gauss(mean, std_dev)

    sim_complex = DependentMonteCarloSimulation(
        iterations=1000,
        model_function=complex_model,
        variable_distributions={
            "x": uniform_dist,
            "y": dependent_uniform,
            "z": double_dependent_dist
        },
        dependencies={
            "x": [],           # x is independent
            "y": ["x"],          # y depends on x
            "z": ["x", "y"]      # z depends on both x and y
        }
    )

    sim_complex.run()
    print("Results:", sim_complex.summary())

    # Show statistics for each variable
    for var_name in ["x", "y", "z"]:
        print(f"{var_name} stats:", sim_complex.get_variable_stats(var_name))
