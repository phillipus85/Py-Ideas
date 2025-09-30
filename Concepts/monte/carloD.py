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


# Example usage with lambda functions
if __name__ == "__main__":
    # Example 1: Independent variables with lambda functions
    print("Example 1: Independent variables (lambda functions)")

    # Model function as lambda
    model_indep = lambda x, y: x * y + 2

    sim_indep = DependentMonteCarloSimulation(
        iterations=1000,
        model_function=model_indep,
        variable_distributions={
            # Independent distributions as lambda functions
            "x": lambda: random.uniform(0, 1),
            "y": lambda: random.gauss(5, 1)
        }
    )

    sim_indep.run()
    print("Results:", sim_indep.summary(), "\n")

    # Example 2: Dependent variables with lambda functions
    print("Example 2: Dependent variables (lambda functions)")

    sim_dep = DependentMonteCarloSimulation(
        iterations=1000,
        model_function=lambda x, y: x * y + 2,
        variable_distributions={
            # Independent distribution as lambda
            "x": lambda: random.uniform(0, 1),
            # Dependent distribution as lambda - note the parameter
            "y": lambda x_value: random.gauss(5 + x_value * 2, 1)
        },
        dependencies={
            "y": ["x"]  # y depends on x
        }
    )

    sim_dep.run()
    print("Results:", sim_dep.summary())
    print("Full summary:", sim_dep.full_summary())

    # Example 3: Multiple dependencies with lambda functions
    print("\nExample 3: Multiple dependent variables (lambda functions)")

    sim_complex = DependentMonteCarloSimulation(
        iterations=1000,
        model_function=lambda x, y, z: x * y * z + 2,
        variable_distributions={
            # All distributions as lambda functions
            "x": lambda: random.uniform(0, 1),
            # Lambda with one dependency
            "y": lambda x_value: random.uniform(0, x_value * 10),
            # Lambda with two dependencies
            "z": lambda x_value, y_value: random.gauss(
                x_value * 5 + y_value / 2,  # mean
                max(0.5, y_value / 10)      # std_dev
            )
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

    # Example 4: More complex model with lambda functions
    print("\nExample 4: Complex financial model (lambda functions)")

    # A more complex financial model simulation
    sim_financial = DependentMonteCarloSimulation(
        iterations=10000,
        # Financial model: portfolio return calculation
        model_function=lambda market_return, risk_free_rate, stock_return, bond_return, stock_weight: 
            stock_weight * stock_return + (1 - stock_weight) * bond_return,
        variable_distributions={
            # Market return (independent)
            "market_return": lambda: random.gauss(0.08, 0.16),  # 8% mean, 16% std dev

            # Risk-free rate (independent)
            "risk_free_rate": lambda: random.gauss(0.03, 0.01),  # 3% mean, 1% std dev

            # Stock weight (independent but constrained to 0-1)
            "stock_weight": lambda: min(max(random.gauss(0.6, 0.2), 0.0), 1.0),  # 60% mean, constrained to [0,1]

            # Stock return (depends on market return and risk-free rate)
            "stock_return": lambda market, rf: rf + 1.2 * (market - rf) + random.gauss(0, 0.05),  # CAPM with beta=1.2

            # Bond return (depends on risk-free rate)
            "bond_return": lambda rf: rf + random.gauss(0.01, 0.03)  # Slight premium over risk-free
        },
        dependencies={
            "stock_return": ["market_return", "risk_free_rate"],
            "bond_return": ["risk_free_rate"]
        }
    )

    sim_financial.run()
    print("Portfolio results:", sim_financial.summary())

    # Show statistics for key variables
    print("\nAsset class statistics:")
    for var_name in ["market_return", "risk_free_rate", "stock_return", "bond_return"]:
        print(f"{var_name}:", sim_financial.get_variable_stats(var_name))
