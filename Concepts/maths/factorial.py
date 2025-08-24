"""
Factorial module that provides factorial calculation for both integers and floats.

For integers n ≥ 0: factorial(n) = n!
For floats x: factorial(x) uses the gamma function where factorial(x) = Γ(x+1)
"""

import math
from typing import Union, Optional


def factorial(x: Union[int, float],
              precision: Optional[int] = None) -> Union[int, float]:
    """
    Calculate the factorial of a number, including support for floats less than 1.0.

    For integers n ≥ 0: Returns n! (n factorial)
    For floats x: Returns Γ(x+1) (gamma function)

    Args:
        x: The number to compute factorial for. Can be integer or float.
        precision: Optional number of decimal places for rounding the result. If None, no rounding is performed.

    Returns:
        The factorial of x. Returns an integer for integer inputs ≥ 0, and a float for float inputs or integers < 0.

    Raises:
        ValueError: If x is a negative integer.

    Examples:
        >>> factorial(5)
        120
        >>> factorial(0)
        1
        >>> factorial(0.5)  # Equivalent to Γ(1.5) = 0.5 * Γ(0.5) = 0.5 * √π
        0.8862269254527579
        >>> factorial(-0.5)  # Equivalent to Γ(0.5) = √π
        1.7724538509055159
    """
    if isinstance(x, int) and x >= 0:
        # Standard factorial for non-negative integers
        result = math.factorial(x)
    elif isinstance(x, int) and x < 0:
        # Factorial is not defined for negative integers
        raise ValueError("Factorial is not defined for negative integers")
    else:
        # For floats, use the gamma function: Γ(x+1)
        result = math.gamma(x + 1)

    # Apply precision if specified
    if precision is not None:
        result = round(result, precision)

    return result


if __name__ == "__main__":
    # Demo of the factorial function
    test_values = [0, 1, 5, 10, 0.5, 1.5, -0.5, 2.0, 10.0, 10.1]

    print("Factorial Examples:")
    print("-" * 40)
    print(f"{'Value':<10} {'Factorial':<25}")
    print("-" * 40)

    for val in test_values:
        try:
            result = factorial(val)
            print(f"{val:<10} {result:<25}")
        except ValueError as e:
            print(f"{val:<10} Error: {str(e)}")
