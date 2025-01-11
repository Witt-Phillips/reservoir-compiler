from pyres import Reservoir
import numpy as np
import sympy as sp
from _utils import plotters


# example
o1, o2, o3, o4, o5, o6 = sp.symbols("o1 o2 o3 o4 o5 o6")

sprott_eqs = [
    (o1, -8 * o2),
    (o2, 25 / 4 * o1 + 5 * (o3**2)),
    (o3, 5 / 4 + 20 * o2 - 10 * o3),
]

R = Reservoir.solve(sprott_eqs)
t = 30000
o = R.run(time=t)
plotters.three_d(R, o, "sprott")
