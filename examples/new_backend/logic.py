from pyres import Reservoir
from _utils import inputs, plotters
import sympy as sp

o1, o2, s1, s2 = sp.symbols("o1 o2 s1 s2")

# parameters and pitchfork base
xw = 0.025
xf = 0.1
cx = 3 / 13
ax = -cx / (3 * xw**2)
pitchfork_bifurcation = (ax * (o1**3)) + (cx * o1)

# shifting logic for each gate
logic = {
    "and": -0.1 + (s1 + xf) * (s2 + xf) / (2 * xf),
    "nand": 0.1 + (s1 + xf) * (-s2 - xf) / (2 * xf),
    "or": 0.1 + (s1 - xf) * (-s2 + xf) / (2 * xf),
    "nor": -0.1 + (s1 - xf) * (s2 - xf) / (2 * xf),
    "xor": 0.0 + (-s1) * (s2) / xf,
    "xnor": 0.0 + (s1) * (s2) / xf,
}

currentlyRunning = "nor"
logic_eqs = [(o1, 10 * (pitchfork_bifurcation + logic[currentlyRunning]))]

R = Reservoir.solve(logic_eqs, fold_recurrent_outputs=False)
print(R.input_names)
print(R.output_names)

high_low_inps = inputs.high_low_inputs(4000)
o = R.run(high_low_inps)
plotters.in_out_split(R, high_low_inps, o, currentlyRunning)

if 1:
    R.save(f"{currentlyRunning}")
