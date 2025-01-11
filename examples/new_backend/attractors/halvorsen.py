from pyres import Reservoir
import numpy as np
import sympy as sp
from _utils import plotters


# example
o1, o2, o3, o4, o5, o6 = sp.symbols("o1 o2 o3 o4 o5 o6")

halvorsen_eqs = [
    (o1, -2.1 * o1 - 6 * o2 - 6 * o3 - 15 * o2**2),
    (o2, -2.1 * o2 - 6 * o3 - 6 * o1 - 15 * o3**2),
    (o3, -2.1 * o3 - 6 * o1 - 6 * o2 - 15 * o1**2),
]

R = Reservoir.solve(halvorsen_eqs)
t = 30000
o = R.run(time=t)
plotters.three_d(R, o, "halvorsen")
