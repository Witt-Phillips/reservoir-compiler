import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import sympy as sp
from reservoir import *
from prnn import circuit, solve
from utils import inputs, plotters

o1, o2, o3 = sp.symbols('o1 o2 o3')

lorenz_eqs = [
    sp.Eq(o1, o2 - o1),
    sp.Eq(o2, 1/10*o1 - 1/10*o2 - 20*o1*o3),
    sp.Eq(o3, 20*o1*o2 - 4/15*o3 - 0.036)
]

if 0:
    for eq in lorenz_eqs:
        print(sp.octave_code(eq))

time = 5000
lorenz_inputs = inputs.zeros(time)

reservoir = Reservoir.solve(lorenz_eqs)
reservoir: Reservoir

outputs = reservoir.run4input(lorenz_inputs)
plotters.threeD(outputs, "Lorenz Attractor")

# save preset
if 1:
    name = "lorenz"
    reservoir.save(name)  
    res = Reservoir.load(name)
    res: Reservoir



