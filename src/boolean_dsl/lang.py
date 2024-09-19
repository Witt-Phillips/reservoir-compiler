from enum import Enum
from dataclasses import dataclass
from typing import List, Union, Dict, Tuple
from fn_library import *

class Op(Enum):
    AND = "AND"
    NAND = "NAND"
    OR = "OR"
    NOR = "NOR"
    XOR = "XOR"
    XNOR = "XNOR"
    LET = "LET"
    RET = "RET"

# Map Op to corresponding functions
op_funcs = {
    Op.AND: AND,
    Op.NAND: NAND,
    Op.OR: OR,
    Op.NOR: NOR,
    Op.XOR: XOR,
    Op.XNOR: XNOR,
}

Operand = Union[bool, str, 'Expr']

@dataclass
class Expr:
    opcode: Op
    operands: List[Operand]

@dataclass
class Prog:
    exprs: List[Expr]