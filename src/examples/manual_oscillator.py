# add reservoir.py to import path
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import sympy as sp
from reservoir import *
from prnn_method import circuit, prnn_method
from utils import inputs, plotters

# define symbolic equations (naming is not constrained) -----------------
o1, s1, s2 = sp.symbols('o1 s1 s2')

# parameters and pitchfork base
xw = 0.025
xf = 0.1
cx = 3/13
ax = -cx / (3 * xw**2)
pitchfork_bifurcation = (ax * (o1 ** 3)) + (cx * o1)

# shifting logic for each gate
nand_logic = 0.1 + (s1 + xf) * (-s2 - xf) / (2 * xf)

# list of symbolic output equations
logic_eqs = [
    sp.Eq(o1, pitchfork_bifurcation + nand_logic)
]

nand_res, _ = Reservoir.solveReservoir(logic_eqs)

# Manual circuit: Oscillator
n = 30
A = nand_res.A
B = nand_res.B
W = nand_res.W

O = np.zeros((n, n))
OB = np.zeros((n, 1))
b1 = B[:, 0]
b2 = B[:, 1]

# Construct circuit
AC = np.block([
    [A,                  O,                  np.outer(b1+b2, W)],
    [np.outer(b1+b2, W), A,                  O],
    [O,                  np.outer(b1+b2, W), A]
])

BC = np.tile(OB, (3, 1))


RAD = Reservoir(AC, 
                BC, 
                np.tile(nand_res.r_init, (3, 1)), 
                0, 
                nand_res.global_timescale, 
                nand_res.gamma, 
                np.tile(nand_res.d, (3, 1)), 
                None)
W = nand_res.W

# Run for no input.
time = 7000
input_data = np.zeros((1, time, 4))
radp = RAD.run4input(input_data, np.identity(3*n))

# Find o=Wr
outputs = np.vstack([
    W @ radp[0:30, :],  # first nand
    W @ radp[30:60, :],  # second nand
    W @ radp[60:90, :]   # third nand
])

# plot using 'outputs' to see all three signals
exposed_output = outputs[2, :].reshape(1, -1)
print(exposed_output.shape)

plotters.InOutSplit(input_data, exposed_output, "Oscillator")