def AND(a: bool, b: bool) -> bool:
    return a and b

def NAND(a: bool, b: bool) -> bool:
    return not (a and b)

def OR(a: bool, b: bool) -> bool:
    return a or b

def NOR(a: bool, b: bool) -> bool:
    return not (a or b)

def XOR(a: bool, b: bool) -> bool:
    return a != b

def XNOR(a: bool, b: bool) -> bool:
    return a == b