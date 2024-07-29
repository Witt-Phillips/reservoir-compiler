import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import sympy as sp
from reservoir import *
from prnn_method import circuit, prnn_method
from utils import inputs, plotters

verbose = False
# define symbolic equations (naming is not constrained) -----------------
o1, s1, s2 = sp.symbols('o1 s1 s2')

# parameters and pitchfork base
xw = 0.025
xf = 0.1
cx = 3/13
ax = -cx / (3 * xw**2)
pitchfork_bifurcation = (ax * (o1 ** 3)) + (cx * o1)

# shifting logic for each gate
logic = {
    'and': -0.1 + (s1 + xf) * (s2 + xf) / (2 * xf),
    'nand': 0.1 + (s1 + xf) * (-s2 - xf) / (2 * xf),
    'or': 0.1 + (s1 - xf) * (-s2 + xf) / (2 * xf),
    'nor': -0.1 + (s1 - xf) * (s2 - xf) / (2 * xf),
    'xor': 0.0 + (-s1) * (s2) / xf,
    'xnor': 0.0 + (s1) * (s2) / xf
}

currentlyRunning = 'nor'

# list of symbolic output equations
logic_eqs = [
    sp.Eq(o1, pitchfork_bifurcation + logic[currentlyRunning])
]

# display octave code (matlab readable equations)
if verbose:
    for eq in logic_eqs:
        print(sp.octave_code(eq))

# generate inputs
logic_inputs = inputs.high_low_inputs(1000)

# solve for W/ internalize recurrencies (solveReservoir can also take & run inputs)
reservoir = Reservoir.solveReservoir(logic_eqs)
reservoir: Reservoir

# run network forward
outputs = reservoir.run4input(logic_inputs)
plotters.InOutSplit(logic_inputs, outputs, currentlyRunning + " Gate")

# save preset
if 0:
    reservoir.saveFile(currentlyRunning)  
    # validate
    res = Reservoir.loadFile(currentlyRunning)
    res: Reservoir
