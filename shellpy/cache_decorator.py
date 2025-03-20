import hashlib
from functools import wraps
import numpy as np

# Global dictionary to store cached results for functions
cache_global = {}


def generate_hash(value):
    """Generates a unique hash for a given input value.

    - If the value is a number (int, float) or a NumPy array, a hash is generated based on its content.
    - For NumPy arrays, the content is converted to bytes before hashing to ensure uniqueness.
    - For other data types, the hash is based only on the type of the object.
    """
    if isinstance(value, (int, float, np.ndarray)):
        if isinstance(value, np.ndarray):
            value = value.tobytes()  # Convert arrays to bytes for hashing
        return hashlib.sha256(str(value).encode("utf-8")).hexdigest()

    # For other types, return a hash based only on the object's type
    return hashlib.sha256(str(type(value)).encode("utf-8")).hexdigest()


def make_cache_key(func_name, *args, **kwargs):
    """Creates a unique cache key for a function based on its arguments.

    - The key includes the function name, a hash of positional arguments, and a hash of keyword arguments.
    - Keyword arguments (kwargs) are sorted to ensure consistency in hash generation.
    """
    args_hash = tuple(generate_hash(arg) for arg in args)
    kwargs_hash = tuple((k, generate_hash(v)) for k, v in sorted(kwargs.items()))
    return (func_name, args_hash, kwargs_hash)


def cache_method(func):
    """Decorator to store the cache of instance methods.

    - The cache is stored within the instance itself (`self.cache`).
    - If the result is already in the cache, it is returned directly.
    - Otherwise, the function is executed, and the result is stored.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        cache_key = make_cache_key(func.__name__, *args, **kwargs)

        # Check if the result is already cached
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Compute the result and store it in the instance's cache
        result = func(self, *args, **kwargs)
        self.cache[cache_key] = result
        print("Caching method result: ", len(self.cache), func.__name__)
        return result

    return wrapper


def cache_function(func):
    """Decorator to store the cache of global functions.

    - Uses the global dictionary `cache_global` to store function results.
    - If the function has been called with the same arguments before, the cached result is returned.
    - Otherwise, the function is executed, and its result is stored.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        cache_key = make_cache_key(func.__name__, *args, **kwargs)

        # Check if the result is already in the global cache
        if cache_key in cache_global:
            return cache_global[cache_key]

        # Execute the function and store the result in the global cache
        result = func(*args, **kwargs)
        cache_global[cache_key] = result
        print("Caching function result: ", len(cache_global), func.__name__)

        return result

    return wrapper


def clear_cache(shell = None):
    cache_global = {}
    if not shell is None:
        shell.mid_surface_geometry.cache = {}
        shell.material.cache = {}

