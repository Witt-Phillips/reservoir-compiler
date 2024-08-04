import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import sympy as sp
from reservoir import *
from prnn_method import circuit, prnn_method
from utils import inputs, plotters

o1, o2, o3 = sp.symbols('o1 o2 o3')

rho = 28
sigma = 10
beta = 8/3
scale = 1

lorenz_eqs = [
    sp.Eq(o1, scale * (sigma * (o2 - o1))),
    sp.Eq(o2, scale * (o1 * (rho - o3) - o2)),
    sp.Eq(o3, scale * (o1 * o2 - beta * o3))
]

for eq in lorenz_eqs:
    print(sp.octave_code(eq))

time = 10000
lorenz_inputs = inputs.zeros(time)

reservoir = Reservoir.solveReservoir(lorenz_eqs)
reservoir.x_init = np.array([0, 1, 1.05])
reservoir: Reservoir

outputs = reservoir.run4input(lorenz_inputs)
plotters.threeD(outputs, "Lorenz Attractor")


