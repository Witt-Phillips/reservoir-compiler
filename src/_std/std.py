from typing import Any, Callable, Dict, TypeVar
from functools import wraps
from _prnn.reservoir import Reservoir


class stdFnInfo:
    def __init__(self, path: str, inp_dim: int, out_dim: int):
        self.path = path
        self.inp_dim = inp_dim
        self.out_dim = out_dim


registry: Dict[str, stdFnInfo] = {}  # name -> stdFnInfo

# Type variable to preserve the original function signature
F = TypeVar("F", bound=Callable[..., Any])


def std_function(name: str, path: str, inp_dim: int, out_dim: int) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        if name in registry:
            raise KeyError(f"Function '{name}' is already registered in std.")
        # Register the function with its info
        registry[name] = stdFnInfo(path, inp_dim, out_dim)

        @wraps(func)
        def wrapper(*args, **kwargs):
            raise NotImplementedError(
                f"{name} was executed by the Python interpreter. "
                "Be sure to use the Reservoir Compiler for .pyres files!"
            )

        return func

    return decorator


@std_function("nand", "nand", inp_dim=2, out_dim=1)
def nand(x, y) -> Reservoir:
    """logical nand: .1 -> True and -.1 -> False"""
    pass


@std_function("std_and", "and", inp_dim=2, out_dim=1)
def std_and(x, y) -> Reservoir:
    """logical nand: .1 -> True and -.1 -> False"""
    pass


@std_function("fan", "fan", inp_dim=1, out_dim=2)
def fan(x) -> Reservoir:
    """fanout: o1 == o2 == x. TODO: broken!"""
    pass


@std_function("nor2", "nor_double", inp_dim=2, out_dim=2)
def nor2(x, y) -> Reservoir:
    """logical nor: .1 -> True and -.1 -> False. Doubled output"""
    pass


@std_function("nor3", "nor_triple", inp_dim=2, out_dim=3)
def nor3(x, y) -> Reservoir:
    """logical nor: .1 -> True and -.1 -> False. Tripled output"""
    pass


@std_function("lorenz", "lorenz", inp_dim=0, out_dim=3)
def lorenz(x, y) -> Reservoir:
    """lorenz series"""
    pass
