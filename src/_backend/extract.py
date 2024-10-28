import sympy as sp
from sympy.core.relational import Equality
from collections import OrderedDict
from typing import Tuple, Optional

# (coef, var_idx, exp)
PackedExpr = Tuple[float, int, float]
# [(o_var_id, expr_wise_list[PackedExpr])]
PackedEqs = list[Tuple[int, list[PackedExpr]]]


def extract(eqs: list[sp.Eq]) -> Tuple[PackedEqs, list[str]]:
    # var_name, found on lhs
    ctxt: OrderedDict[str, bool] = {}
    result: PackedEqs = []

    def lookup(
        var: str, used_cond: Optional[bool] = None, add_if_absent=False
    ) -> Optional[int]:
        """
        * Returns the index of the var in ctxt, None if not present.
        * If used_cond (to specify whether a var has appeared on the lhs)
        * add_if_absent inserts teh var/ used_cond pair if not found, returning index
        is set, lookup only if ctxt value matches.
        """
        if var in ctxt and (used_cond is None or ctxt[var] == used_cond):
            return list(ctxt.keys()).index(var)
        elif add_if_absent:
            ctxt[var] = False if used_cond is None else used_cond
            return list(ctxt.keys()).index(var)

        return None

    def parse_expr(expr: sp.Expr) -> PackedExpr:
        match expr:
            case sp.Number():
                i = lookup("1", add_if_absent=True)
                return float(expr), i, 1.0

            case sp.Symbol():
                i = lookup(str(expr), add_if_absent=True)
                return 1.0, i, 1.0

            case sp.Expr():
                coef, rest = expr.as_coeff_Mul()
                base, exp = rest.as_base_exp()

                # TODO: how do we want to handle cases of multiplied bases? For now, treat them as compound symbols
                base = lookup(str(base), add_if_absent=True)
                coef = float(coef) if isinstance(coef, sp.Number) else parse_expr(coef)
                exp = float(exp) if isinstance(exp, sp.Number) else parse_expr(exp)

                return coef, base, exp
            case _:
                raise ValueError("Got unrecognized expr type")

    for eq in eqs:
        eq: Equality

        # handle lhs
        lhs = eq.lhs
        if isinstance(lhs, sp.Symbol):
            lhs = str(lhs)
            if lookup(lhs, used_cond=True) is not None:
                raise ValueError(f"Output {lhs} defined more than once.")
            else:
                out_idx = lookup(lhs, used_cond=True, add_if_absent=True)
        else:
            raise ValueError(f"LHS must be a single variable. Invalid LHS: {lhs}")

        # handle rhs
        pkd_eqs: list[PackedExpr] = []
        rhs: sp.Expr = eq.rhs
        terms = rhs.as_ordered_terms()
        for term in terms:
            coef, i, exp = parse_expr(term)
            pkd_eqs.append((coef, i, exp))

        result.append((out_idx, pkd_eqs))

    return result, list(ctxt.keys())


# Sandbox ----------

o1, o2, o3, o4, i1, i2, i3 = sp.symbols("o1 o2 o3 o4 i1 i2 i3")

""" TODO: odd equality issue happening here. If sp knows an Equation evals
 false, it simplifies to false. Need evaluate=False to suppress """
eqs = [
    sp.Eq(o1, 10 + 2 * i1**3 + i2**4),  # a + ax^b + x^b
    sp.Eq(o2, o2 + 4, evaluate=False),  # reccurence
    sp.Eq(o3, i2 * i3),  # x * y
    sp.Eq(
        o4, (2 * i3**2) * i2**4
    ),  # these cases still handled poorly. creates a var 'i2**4*i3**2'
]

peqs, syms = extract(eqs)

# some pretty printing

print("SymPy Equations: ")
for eq in eqs:
    print(sp.octave_code(eq))

print("\nSymbol indices: ")
print("idx | var")
for i, sym in enumerate(syms):
    print(i, sym)

print("\nProcessed equations:")
print("* format: coef | var | exp\n")
for peq in peqs:
    idx, partials = peq
    name = syms[idx]
    print(f"{idx} (maps to {name}) = ...")
    for partial in partials:
        print(partial)

""" 
Meeting notes:

get B, W, bias. I do A = BW, everything else is passed directly

* for now, pseudo-arbitarily "enough" latents
next
assume: linearization abt 0
* non-polynomical funcs: 1) try to symbolically diff. 

* from eqs (invariant: polynomials) -> 1) monomial basis 2) pow on monomial / coef --*jason magic*--> SNP/DNP O mat
    syms: x, y, z -> uids rng(1 -> len(syms))
    o1 = ax**4 + byz**2 --> {o1 : (a, x, 4), ((b, y, 1), z, 2)}

* somewhat trivial to get RNN approx (R) w/ no connectivity
* super duper trivial O -> R via W;
once I have a res and W, I take it from there
 """
