""" 
Language primitives for the IR used by reservoir compiler. Every program
consists of a list of expressions. An expression is a tuple of an
opcode (either predefined in enum or custom string to be lookup up in 
library) followed by a string of operands. An operand is either an expression,
variable, or list of variables.
"""

from enum import Enum, auto
from typing import List, Union
from dataclasses import dataclass


class Opc(Enum):
    """
    Reserved opcodes
    """

    LET = auto()
    RET = auto()
    INPUT = auto()

@dataclass
class Operand:
    """
    Operand: arguments passed to opcodes. Note their recursive
    structre; they can take the form of expressions with their own operands.
    """

    operand: Union[list[str], str, "Expr"]


@dataclass
class Expr:
    """
    Expression: basic evaluation unit for our IR. Combines an opcode
    with its arguments and evaluates. All top-level exprs must evaluate to none
    (indicating reserved opcode) to None.
    """

    op: Union[Opc, str]  # op for special cases, str for gates
    operands: List[Operand]


@dataclass
class Prog:
    """
    Program: list of expressions to be evaluated.
    """

    exprs: List[Expr]
