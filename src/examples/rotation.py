import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import sympy as sp
from reservoir import *
from prnn_method import circuit, prnn_method
from utils import inputs, plotters

#TODO: Make 'x#' valid symbol name for sympy eqs
o1, o2, o3, s1, s2, s3 = sp.symbols('o1 o2 o3 s1 s2 s3')

logic_eqs = [
    sp.Eq(o1, -s2),
    sp.Eq(o2, s1),
    sp.Eq(o3, s3)
]

rotation_res = Reservoir.solveReservoir(logic_eqs)
input_data = inputs.lorenz(5000)
outputs = rotation_res.run4input(input_data)
plotters.threeDInputOutput(input_data, outputs, 'Rotation')

#plotters.threeD(input_data, 'Lorenz Attractor')

