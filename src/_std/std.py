from typing import Any, Callable, Dict, TypeVar
from functools import wraps
from _prnn.reservoir import Reservoir

registry: Dict[str, str] = {}  # name, path to load

# Type variable to preserve the original function signature
F = TypeVar("F", bound=Callable[..., Any])


def std_function(name: str, path: str) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        if name in registry:
            raise KeyError(f"Function '{name}' is already registered in std.")
        registry[name] = path

        @wraps(func)
        def wrapper(*args, **kwargs):
            raise NotImplementedError(
                f"{name} was executed by the Python interpreter. "
                "Be sure to use the Reservoir Compiler for .pyres files!"
            )

        return func

    return decorator


@std_function("nand", "nand")
def nand(x, y) -> Reservoir:
    """logical nand: .1 -> True and -.1 -> False"""
    pass


@std_function("fan", "fan")
def fan(x) -> Reservoir:
    """fanout: o1 == o2 == x. TODO: broken!"""
    pass


@std_function("nor3", "nor_triple")
def nor3(x, y) -> Reservoir:
    """logical nor: .1 -> True and -.1 -> False. Tripled output"""
    pass

@std_function("lorenz", "lorenz")
def lorenz(x, y) -> Reservoir:
    """lorenz series"""
    pass
