from _prnn.reservoir import Reservoir
from _utils import inputs, plotters
import sympy as sp

s, o1, o2 = sp.symbols("s o1 o2")

logic_eqs = [sp.Eq(o1, s), sp.Eq(o2, o1)]

reservoir = Reservoir.solve(logic_eqs)

if 1:
    reservoir.save(f"fan")
