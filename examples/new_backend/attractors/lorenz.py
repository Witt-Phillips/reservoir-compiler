from pyres import Reservoir
import numpy as np
import sympy as sp
from _utils import plotters


# example
o1, o2, o3, o4, o5, o6 = sp.symbols("o1 o2 o3 o4 o5 o6")

lorenz_eqs = [
    (o1, 10 * (o2 - o1)),
    (o2, o1 * (1 - o3) - o2),
    (o3, o1 * o2 - 8 * o3 / 3 - 8 / 3 * 27),
]

R = Reservoir.solve(lorenz_eqs)
R.gamma *= 15
t = int(30000 / 15)
o = R.run(time=t)
plotters.three_d(R, o, f"Lorenz: gam = {R.gamma}, t = {t}", path=".")
