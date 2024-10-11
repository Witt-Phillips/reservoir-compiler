from prnn.reservoir import Reservoir
from typing import Any, Callable, Dict, TypeVar
from functools import wraps

registry: Dict[str, str] = {}  # name, path to load

# Type variable to preserve the original function signature
F = TypeVar("F", bound=Callable[..., Any])


def stdlib_function(name: str, path: str) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        if name in registry:
            raise KeyError(f"Function '{name}' is already registered in stdlib.")
        registry[name] = path

        @wraps(func)
        def wrapper(*args, **kwargs):
            raise NotImplementedError(
                f"{name} was executed by the Python interpreter. "
                "Be sure to use the Reservoir Compiler for .pyres files!"
            )

        return func

    return decorator


@stdlib_function("nand", "nand")
def nand(i1, i2) -> Reservoir:
    """Logical nand, where .1 -> True and -.1 -> False"""
    pass
