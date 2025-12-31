from functools import wraps


# WITHOUT @wraps(func)
# Level 1: The decorator factory - takes configuration parameters
def validate_min_length_bad(min_len: int) -> callable:
    """*validate_min_length_bad()* "This is the outer function - it receives decorator parameters.

    Args:
        min_len (int): minimum length

    Returns:
        callable: the actual decorator
    """

    # Level 2: The actual decorator - receives the function to be decorated
    def decorator(func):
        """This receives the function we're decorating."""

        # Level 3: The wrapper - receives the actual arguments when called
        def wrapper(self, value):
            """This runs when the decorated function is called."""

            # Validation logic using the parameter from level 1
            if len(value) < min_len:
                raise ValueError(f"Too shor, {len(value)} < {min_len}")

            # Call the original function
            return func(self, value)
        return wrapper  # Return the wrapper function
    return decorator    # Return the decorator function


# WITH @wraps(func)
# Level 1: The decorator factory - takes configuration parameters
def validate_min_length_good(min_len: int) -> callable:
    """*validate_min_length_good()* "This is the outer function - it receives decorator parameters.

    Args:
        min_len (int): minimum length

    Returns:
        callable: the actual decorator
    """

    # Level 2: The actual decorator - receives the function to be decorated
    def decorator(func):
        """This receives the function we're decorating."""

        # Level 3: The wrapper - receives the actual arguments when called
        @wraps(func)  # ← This preserves func's metadata
        def wrapper(self, value):
            """This runs when the decorated function is called."""

            # Validation logic using the parameter from level 1
            if len(value) < min_len:
                raise ValueError(f"Too shor, {len(value)} < {min_len}")

            # Call the original function
            return func(self, value)
        return wrapper  # Return the wrapper function
    return decorator    # Return the decorator function


# Example usage with a class
class User:
    @property
    def gname(self):
        return self._name

    @property
    def bname(self):
        return self._name

    @gname.setter
    @validate_min_length_good(3)
    def gname(self, value):
        """Sets the user's name."""
        self._name = value

    @bname.setter
    @validate_min_length_bad(3)
    def bname(self, value):
        """Sets the user's name."""
        self._name = value


if __name__ == "__main__":
    print("=== Decorator example ===\n")
    # WITHOUT @wraps:
    print("WITHOUT @wraps:")
    print(User.bname.fset.__name__)    # 'wrapper' ❌ (lost original name!)
    print(User.bname.fset.__doc__)     # None ❌ (lost docstring!)

    # WITH @wraps:
    print("\nWITH @wraps:")
    print(User.gname.fset.__name__)    # 'name' ✅ (preserved!)
    print(User.gname.fset.__doc__)     # 'Sets the user's name.' ✅ (preserved!)
