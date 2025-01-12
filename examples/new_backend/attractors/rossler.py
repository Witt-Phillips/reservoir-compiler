from pyres import Reservoir
import numpy as np
import sympy as sp
from _utils import plotters

# example
o1, o2, o3, o4, o5, o6 = sp.symbols("o1 o2 o3 o4 o5 o6")

rossler_eqs = [
    (o1, (-5 * o2) - (5 * o3) - (5 * 3 / 5)),
    (o2, (5 * o1) + o2),
    (o3, (50 * 5 / 3 * o1) * o3 + (50 * o1) - 28.5 * o3 - 28.4 * 3 / 5),
]

# o2 o3 o1
R = Reservoir.solve(rossler_eqs)

t = 15000
o = R.run(time=t)
print("Output at t = 0:\n", o[:, 0])
plotters.three_d(R, o, "rossler")
