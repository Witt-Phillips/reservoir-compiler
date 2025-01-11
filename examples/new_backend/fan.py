from pyres import Reservoir
from _utils import inputs, plotters
import sympy as sp
import numpy as np

s, o1, o2 = sp.symbols("s o1 o2")
a = -1
# logic_eqs = [(o1, -1000 * (o1 - s) ** 3)]
logic_eqs = [(o1, -20 * (o1 - s))]

R = Reservoir.solve(logic_eqs)

print(R.input_names)
print(R.output_names)

inp = np.full((1, 4000), 0.1)
o = R.run(inp)
plotters.in_out_split(R, inp, o, "fan")

if False:
    R.save(f"fan")
