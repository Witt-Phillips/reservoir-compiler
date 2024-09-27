""" 
Validates and transforms a set of user-defined equations:
Input: user-defined system of equations.
Outputs:
    * transformed_eqs: List of right hand sides of user functions, with symbols replaced by x(t)
    * recurrences: list of tuples (o#, x#(t)) s.t. o# = x#(t)
"""

import sympy as sp


def process(eqs: list[sp.Eq]):
    transformed_eqs: list[sp.Eq] = []
    recurrences: list[(int, int)] = []
    rhs_symbols: list[sp.Symbol] = []
    lhs_symbols: list[sp.Symbol] = []

    # extract symbols
    for eq in eqs:
        lhs = eq.lhs.free_symbols
        rhs = eq.rhs.free_symbols

        if len(lhs) != 1:
            raise ValueError(
                "Equation must have exactly one symbol on the left-hand side."
            )

        transformed_eqs.append(eq.rhs)
        lhs_symbols.extend(symbol for symbol in lhs if symbol not in lhs_symbols)
        rhs_symbols.extend(symbol for symbol in rhs if symbol not in rhs_symbols)

    # find recurrences
    for o_idx, output in enumerate(lhs_symbols):
        if output in rhs_symbols:
            recurrences.append((o_idx, rhs_symbols.index(output)))

    # init/ substitute new symbols
    t = sp.symbols("t")
    x = sp.symbols("x:{0}(t)".format(len(rhs_symbols)))

    for eq_idx, eq in enumerate(transformed_eqs):
        for sym_idx, sym in enumerate(rhs_symbols):
            transformed_eqs[eq_idx] = transformed_eqs[eq_idx].subs(sym, x[sym_idx])

    return transformed_eqs, recurrences
