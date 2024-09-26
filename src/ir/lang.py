from enum import Enum, auto
from typing import List, Union
from dataclasses import dataclass

class Opc(Enum):
    LET = auto()
    RET = auto()
    INPUT = auto()
    REC = auto()

@dataclass
class Operand:
    Union[list[str], str, 'Expr']

# Define Expressions and Program
@dataclass
class Expr:
    op: Union[Opc, str] #op for special cases, str for gates
    operands: List[Operand]

@dataclass
class Prog:
    exprs: List[Expr]


