from dataclasses import dataclass, field
from typing import Callable, List, Dict, Union, Any
import numpy as np
import random


@dataclass
class DependentMonteCarloSimulation:
    """
    Monte Carlo simulation with support for dependent variables where the output
    of one distribution can be used as input to another.

    Supports both named functions and lambda functions interchangeably.
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


# Example usage with mixed named functions and lambda functions
if __name__ == "__main__":
    # Define named functions for distributions
    def uniform_01():
        """Generate a uniform random number between 0 and 1"""
        return random.uniform(0, 1)

    def normal_5_1():
        """Generate a normal random number with mean 5 and std dev 1"""
        return random.gauss(5, 1)

    def dependent_normal(x_value):
        """Generate a normal random number with mean depending on x"""
        return random.gauss(5 + x_value * 2, 1)

    # Define a named model function
    def multiply_add_2(x, y):
        """Multiply x and y and add 2"""
        return x * y + 2

    # Example 1: Mixed named and lambda functions for independent variables
    print("Example 1: Mixed named and lambda functions (independent variables)")

    sim_mixed_indep = DependentMonteCarloSimulation(
        iterations=1000,
        model_function=multiply_add_2,  # Using named function
        variable_distributions={
            "x": uniform_01,           # Using named function
            "y": lambda: random.gauss(5, 1)  # Using lambda function
        }
    )

    sim_mixed_indep.run()
    print("Results:", sim_mixed_indep.summary(), "\n")

    # Example 2: Mixed with dependent variables
    print("Example 2: Mixed named and lambda functions (dependent variables)")
    
    sim_mixed_dep = DependentMonteCarloSimulation(
        iterations=1000,
        model_function=lambda x, y: x * y + 2,  # Using lambda
        variable_distributions={
            "x": uniform_01,                    # Using named function
            "y": dependent_normal               # Using named function with parameter
        },
        dependencies={
            "y": ["x"]  # y depends on x
        }
    )

    sim_mixed_dep.run()
    print("Results:", sim_mixed_dep.summary())
    print("Full summary:", sim_mixed_dep.full_summary())

    # Example 3: Complex financial model with mixed functions
    print("\nExample 3: Complex financial model (mixed functions)")

    # Define some named functions for the financial model
    def market_return_generator():
        """Generate market returns with 8% mean and 16% std dev"""
        return random.gauss(0.08, 0.16)

    def risk_free_rate_generator():
        """Generate risk-free rates with 3% mean and 1% std dev"""
        return random.gauss(0.03, 0.01)

    def stock_return_generator(market, rf):
        """Generate stock returns using CAPM with beta=1.2"""
        return rf + 1.2 * (market - rf) + random.gauss(0, 0.05)

    # A more complex financial model simulation
    sim_financial_mixed = DependentMonteCarloSimulation(
        iterations=10000,
        # Named function for model
        model_function=lambda market_return, risk_free_rate, stock_return, bond_return, stock_weight: 
            stock_weight * stock_return + (1 - stock_weight) * bond_return,
        variable_distributions={
            # Named functions
            "market_return": market_return_generator,
            "risk_free_rate": risk_free_rate_generator,
            "stock_return": stock_return_generator,

            # Lambda functions
            "stock_weight": lambda: min(max(random.gauss(0.6, 0.2), 0.0), 1.0),
            "bond_return": lambda rf: rf + random.gauss(0.01, 0.03)
        },
        dependencies={
            "stock_return": ["market_return", "risk_free_rate"],
            "bond_return": ["risk_free_rate"]
        }
    )

    sim_financial_mixed.run()
    print("Portfolio results:", sim_financial_mixed.summary())
    
    # Show statistics for key variables
    print("\nAsset class statistics:")
    for var_name in ["market_return", "risk_free_rate", "stock_return", "bond_return"]:
        print(f"{var_name}:", sim_financial_mixed.get_variable_stats(var_name))